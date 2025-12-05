/*
 * Copyright 2020-2025 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample shows the usage of fcntl locks with cuFile for
 * unaligned writes to achieve atomic transactions. We initialize
 * three threads that do the following:
 * 1. Thread 0 writes to file from offset 10 - write size 100 bytes
 * 2. Thread 1 writes to file from offset 50 - write size 200 bytes
 * 3. Thread 2 reads from file from offset 1000 - read size 100 bytes
 *
 * All three threads have a overlapping region between offset 0 and offset 4K.
 * 
 * ./file_control_locks <testfile> <gpuid>
 *
 * | Output |
 * Write lock acquired from offset 0 size 4096. Submit write at offset 10 size 100
 * Write lock acquired from offset 0 size 4096. Submit write at offset 50 size 200
 * Read lock acquired from offset 0 size 4096. Submit read at offset 1000 size 100
 * Write success ret = 200 at offset 50 size 200 on tid=1
 * Write success ret = 100 at offset 10 size 100 on tid=0
 * Read success ret = 100 at offset 1000 size 100 on tid=2
 *
 * Required: testfile must be already created and filled with random data
 * dd if=/dev/urandom of=testfile bs=1G count=1 (NOTE: only 4 KiB is used here)
 */

#include <iostream>
#include <future>
#include <thread>
#include <array>
#include <mutex>
#include <sys/types.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <cstdlib>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

static constexpr int MAX_RETRY = 3;
static constexpr int NUM_THREADS = 3;
static constexpr int GPU_MEMORY_SIZE = MiB(1);
static constexpr int NVFS_BLOCK_SIZE = KiB(4);
// Mutex to help format thread-specific output
static std::mutex finalCoutMutex;

static size_t alignUp(size_t x, size_t align_to) {
	return (x + (align_to - 1)) & ~(align_to - 1);
}

static size_t alignDown(size_t x, size_t align_to) {
	return x & ~(align_to - 1);
}

struct ThreadData {
	int tid;
	int gpuid;
	void* devPtr;
	int fd;
	CUfileHandle_t cfr_handle;
	loff_t offset;
	size_t size;

	ThreadData() = default;
	ThreadData(int tid, int gpuid, void* devPtr, int fd, CUfileHandle_t cfr_handle, loff_t offset, size_t size)
		: tid(tid), gpuid(gpuid), devPtr(devPtr), fd(fd), cfr_handle(cfr_handle), offset(offset), size(size) {}
};

// Acquire a lock on the file using fcntl, fails if max_retry is reached
static bool acquire_lock(int fd, struct flock& fl, int max_retry, std::string lock_type) {
	int cnt = 0;
	while (true) {
		cnt++;
		if (fcntl(fd, F_SETLKW, &fl) == -1) {
			std::cout << "Failed to acquire " << lock_type << " lock from offset " << fl.l_start << " size "
				<< fl.l_len << " errno " << cuFileGetErrorString(errno) << std::endl;
			if (cnt == max_retry) {
				std::cerr << "Failed to acquire " << lock_type << " lock after " << max_retry << " retries" << std::endl;
				return false;
			} else {
				std::cout << "Retrying fcntl for " << lock_type << ".." << std::endl;
			}
		} else {
			break;
		}
	}

	return true;
}

static void read_thread_fn(std::promise<int>&& funcRet, ThreadData& threadData) {
	bool success = true;
	int ret = 0;

	/*
	 * We need to set the CUDA device; threads will not inherit main thread's
	 * CUDA context. In this case, since main thread allocated memory on GPU 0,
	 * we set it explicitly. However, threads have to ensure that they are in
	 * same cuda context as devPtr was allocated.
	 */
	check_cudaruntimecall_thread(cudaSetDevice(threadData.gpuid), funcRet);

	// Initialize advisory lock (l_type, l_whence, l_start, l_len, l_pid)
	struct flock fl = { F_RDLCK, SEEK_SET, 0, 0, 0 };
	fl.l_pid = getpid();
	fl.l_type = F_RDLCK;
	
	// Attempt to acquire lock at 4K boundary
	fl.l_start = alignDown(threadData.offset, NVFS_BLOCK_SIZE);
	fl.l_len = alignUp(threadData.size, NVFS_BLOCK_SIZE);
	if (!acquire_lock(threadData.fd, fl, MAX_RETRY, "read")) {
		funcRet.set_value(-1);
		return;
	}

	// Lock acquired, submit read
	std::cout << "Read lock acquired from offset " << fl.l_start 
		<< " size " << fl.l_len << ". Submit read at offset " 
		<< threadData.offset << " size " << threadData.size << std::endl;
	ret = cuFileRead(threadData.cfr_handle, threadData.devPtr, threadData.size, threadData.offset, 0);
	if (ret < 0) {
		std::cerr << "cuFileRead failed with ret=" << ret << std::endl;
		success = false;
	}

	// Unlock the region the advisory lock was acquired on
	fl.l_type = F_UNLCK;
	if (fcntl(threadData.fd, F_SETLKW, &fl) == -1) {
		std::cerr << "fcntl unlock failed" << std::endl;
		success = false;
	}

	if (success) {
		std::lock_guard<std::mutex> lock(finalCoutMutex);
		std::cout << "Read success ret = " << ret << " at offset " << threadData.offset
			<< " size " << threadData.size << " on tid=" << threadData.tid << std::endl;
		funcRet.set_value(0);
	} else {
		funcRet.set_value(-1);
	}
}

static void write_thread_fn(std::promise<int>&& funcRet, ThreadData& threadData) {
	bool success = true;
	int ret = 0;

	/*
	 * We need to set the CUDA device; threads will not inherit main thread's
	 * CUDA context. In this case, since main thread allocated memory on GPU 0,
	 * we set it explicitly. However, threads have to ensure that they are in
	 * same cuda context as devPtr was allocated.
	 */
	check_cudaruntimecall_thread(cudaSetDevice(threadData.gpuid), funcRet);

	// Initialize advisory lock (l_type, l_whence, l_start, l_len, l_pid)
	struct flock fl = { F_WRLCK, SEEK_SET, 0, 0, 0 };
	fl.l_pid = getpid();
	fl.l_type = F_WRLCK;

	// Attempt to acquire lock at 4K boundary
	fl.l_start = alignDown(threadData.offset, NVFS_BLOCK_SIZE);
	fl.l_len = alignUp(threadData.size, NVFS_BLOCK_SIZE);
	if (!acquire_lock(threadData.fd, fl, MAX_RETRY, "write")) {
		funcRet.set_value(-1);
		return;
	}

	// Lock acquired, submit write
	std::cout << "Write lock acquired from offset " << fl.l_start 
		<< " size " << fl.l_len << ". Submit write at offset " 
		<< threadData.offset << " size " << threadData.size << std::endl;
	ret = cuFileWrite(threadData.cfr_handle, threadData.devPtr, threadData.size, threadData.offset, 0);
	if (ret < 0) {
		std::cerr << "cuFileWrite failed with ret=" << ret << std::endl;
		success = false;
	}

	// Unlock the region the advisory lock was acquired on
	fl.l_type = F_UNLCK;
	if (fcntl(threadData.fd, F_SETLKW, &fl) == -1) {
		std::cerr << "fcntl unlock failed" << std::endl;
		success = false;
	}

	if (success) {
		std::lock_guard<std::mutex> lock(finalCoutMutex);
		std::cout << "Write success ret = " << ret << " at offset " << threadData.offset
			<< " size " << threadData.size << " on tid=" << threadData.tid << std::endl;
		funcRet.set_value(0);
	} else {
		funcRet.set_value(-1);
	}
}

int main(int argc, char *argv[]) {
	std::array<std::promise<int>, NUM_THREADS> threadPromises;
	std::array<std::future<int>, NUM_THREADS> threadFutures;
	std::array<std::thread, NUM_THREADS> threadPool;
	std::array<ThreadData, NUM_THREADS> threadData;
	int fd = -1, gpuid = -1, ret = EXIT_SUCCESS;
	void *devPtr = nullptr;

	// cuFile specific variables
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle = nullptr;

	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <testfile> <gpuid>" << std::endl;
		return EXIT_FAILURE;
	}

	// Parse the GPU ID and open the file
	gpuid = parseInt(argv[2]);
	fd = open(argv[1], O_CREAT | O_RDWR | O_DIRECT);
	if (fd < 0) {
		std::cerr << "Error: Unable to open file " << argv[1] << " with error "
			<< cuFileGetErrorString(errno) << std::endl;
		return EXIT_FAILURE;
	}

	// Register the file
	std::memset(static_cast<void*>(&cfr_descr), 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: " << cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto close_file;
	}

	// Allocate memory on the GPU
	check_cudaruntimecall(cudaSetDevice(gpuid));
	check_cudaruntimecall(cudaMalloc(&devPtr, GPU_MEMORY_SIZE));
	check_cudaruntimecall(cudaMemset(devPtr, 0xab, GPU_MEMORY_SIZE));
	check_cudaruntimecall(cudaStreamSynchronize(0));
	
	// Thread 0 will write to file from offset 10 - write size 100 bytes
	// This is an unaligned write as offset is not 4K aligned. cuFile will
	// convert this write to Read-Modify-Write
	threadPromises[0] = std::promise<int>();
	threadFutures[0] = threadPromises[0].get_future();
	threadData[0] = ThreadData(0, gpuid, devPtr, fd, cfr_handle, 10, 100);

	// Thread 1 will write to file from offset 50 - write size 200 bytes
	// This is an unaligned write as offset is not 4K aligned. cuFile will
	// convert this write to Read-Modify-Write
	threadPromises[1] = std::promise<int>();
	threadFutures[1] = threadPromises[1].get_future();
	threadData[1] = ThreadData(1, gpuid, devPtr, fd, cfr_handle, 50, 200);

	// Thread 2 will read from file from offset 1000 - read size 100 bytes
	// This is an unaligned read as offset is not 4K aligned. cuFile will
	// convert this read to Read-Modify-Write
	threadPromises[2] = std::promise<int>();
	threadFutures[2] = threadPromises[2].get_future();
	threadData[2] = ThreadData(2, gpuid, devPtr, fd, cfr_handle, 1000, 100);

	/*
	 * Thread 0 and Thread 1 are unaligned writes in a overlapping region.
	 * Thread 2 is a read but the range is not overlapping between writes.
	 *
	 * However, all three threads have a overlapping region between offset 0 and offset 4K.
	 * cuFile does READ-MODIFY-WRITE on a 4K boundary. Hence, in the aforementioned case, 
	 * it is necessary for all three threads to acquire lock in 4k boundary even through
	 * thread 2 doesn't have a direct overlap.
	 */
	threadPool[0] = std::thread(write_thread_fn, std::move(threadPromises[0]), std::ref(threadData[0]));
	threadPool[1] = std::thread(write_thread_fn, std::move(threadPromises[1]), std::ref(threadData[1]));
	threadPool[2] = std::thread(read_thread_fn, std::move(threadPromises[2]), std::ref(threadData[2]));

	// Join all threads
	for (int i = 0; i < NUM_THREADS; i++) {
		threadPool[i].join();
		if (threadFutures[i].get() != 0) {
			std::cerr << "Error: Thread " << i << " exited with failure" << std::endl;
			ret = EXIT_FAILURE;
		}
	}

	// Clean up
	if (cfr_handle) cuFileHandleDeregister(cfr_handle);
	if (devPtr) check_cudaruntimecall(cudaFree(devPtr));

// Cleanup labels
close_file:
	if (fd >= 0) close(fd);

	return ret;
}
