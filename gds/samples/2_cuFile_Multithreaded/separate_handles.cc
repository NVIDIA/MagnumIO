/*
 * Copyright 2020-2025 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

/*
 * This sample shows how two threads independently open two
 * (different or same) files and have separate copies of 
 * CUfileHandle_t, CUfileDescr_t, and CUDA device pointer.
 *
 * ./separate_handles <testfile1> <testfile2> <gpuid>
 *
 * | Output |
 * Successful read of 1073741824 bytes on thread 0 from fd 3 to GPU id <gpuid>
 * Successful read of 1073741824 bytes on thread 1 from fd 4 to GPU id <gpuid>
 */

#include <iostream>
#include <future>
#include <algorithm>
#include <cstring>
#include <thread>
#include <array>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

static constexpr size_t NUM_THREADS = 2;
static constexpr size_t CHUNK_SIZE = MiB(1);

/*
 * Each thread allocates GPU memory and invokes cuFileBufRegister
 * on the entire buffer. This is done once; after registering the
 * buffer, the thread reads data in chunks into their own buffer.
 * The GPU buffer acts as a streaming buffer, allowing data to be
 * directly DMA'ed into the registered memory. This approach is
 * optimal for performance.
 */
static void thread_fn(std::promise<int>&& funcRet, int tid, int gpuid, int fd) {
	size_t totalBytesRead = 0, fileSize = 0;
	void *devPtr = nullptr;
	bool success = true;
	loff_t offset = 0;
	int nr_ios = 0;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error for thread " << tid << ": " 
			<< cuFileGetErrorString(status) << std::endl;
		funcRet.set_value(-1);
		return;
	}

	// Allocate GPU memory and register it with cuFile
	check_cudaruntimecall_thread(cudaSetDevice(gpuid), funcRet);
	check_cudaruntimecall_thread(cudaMalloc(&devPtr, CHUNK_SIZE), funcRet);
	status = cuFileBufRegister(devPtr, CHUNK_SIZE, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed for thread " << tid << ": " 
			<< cuFileGetErrorString(status) << std::endl;
		success = false;
		goto deregister_filehandle;
	}

	// Determine the number of IOs to perform
	fileSize = GetFileSize(fd);
	nr_ios = fileSize / CHUNK_SIZE;
	if (fileSize % CHUNK_SIZE)
		nr_ios++;

	// Read the file in chunks
	// NOTE: cuFileRead will chunk down to the file size (or remainder) if it's smaller than CHUNK_SIZE
	for (int i = 0; i < nr_ios; i++) {
		int ret = cuFileRead(cf_handle, devPtr, CHUNK_SIZE, offset, 0);
		if (ret < 0) {
			std::cerr << "cuFile Read failed for thread " << tid << ": " 
				<< cuFileGetErrorString(ret) << std::endl;
			success = false;
			goto deregister_bufferhandle;
		}
		offset += ret;
		totalBytesRead += ret;
	}

// Cleanup labels
deregister_bufferhandle:
	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer Deregister failed for thread " << tid << ": " 
			<< cuFileGetErrorString(status) << std::endl;
		success = false;
	}

deregister_filehandle:
	if (cf_handle) cuFileHandleDeregister(cf_handle);

	// Free the device memory
	if (devPtr) check_cudaruntimecall_thread(cudaFree(devPtr), funcRet);

	if (success) {
		funcRet.set_value(0);
		std::cout << "Successful read of " << totalBytesRead << " bytes on thread "
			<< tid << " from fd " << fd << " to GPU id " << gpuid << std::endl;
	} else {
		funcRet.set_value(-1);
	}
}

int main(int argc, char *argv[]) {
	std::array<std::future<int>, NUM_THREADS> threadFutures;
	std::array<std::thread, NUM_THREADS> threadPool;
	std::array<int, NUM_THREADS> fds;
	int ret = EXIT_SUCCESS, gpuid = 0;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testfile1> <testfile2> <gpuid> " << std::endl;
		return EXIT_FAILURE;
	}

	gpuid = parseInt(argv[3]);

	// Close file descriptors helper function
	auto closeFds = [](std::array<int, NUM_THREADS>& fds) {
		std::for_each(fds.begin(), fds.end(), [](int fd) {
			if (fd > 0) close(fd);
		});
	};

	// Open files and get file descriptors
	for (size_t tid = 0; tid < NUM_THREADS; tid++) {
		fds[tid] = open(argv[tid + 1], O_RDWR | O_DIRECT);
		if (fds[tid] < 0) {
			std::cerr << "Error: Unable to open file " << argv[tid + 1] << " for fd " << tid << std::endl;
			// Close all file descriptors that were opened prior to this failure
			closeFds(fds);
			return EXIT_FAILURE;
		}
	}

	// Spawn out each thread with a separate file descriptor
	for (size_t tid = 0; tid < NUM_THREADS; tid++) {
		std::promise<int> funcRet;
		threadFutures[tid] = funcRet.get_future();
		threadPool[tid] = std::thread(thread_fn, std::move(funcRet), tid, gpuid, fds[tid]);
	}

	// Wait for all threads to finish
	for (size_t tid = 0; tid < NUM_THREADS; tid++) {
		threadPool[tid].join();
		if (threadFutures[tid].get() != 0) {
			std::cerr << "Error: Thread " << tid << " exited with failure" << std::endl;
			ret = EXIT_FAILURE;
		}
	}

	// Close all file descriptors
	closeFds(fds);
	return ret;
}
