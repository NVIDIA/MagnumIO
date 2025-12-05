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
 * This sample shows two threads with separate copies of CUfileHandle_t 
 * performing READs with 64 different GPU buffers and sets maximum buffer 
 * space that is pinned and mapped to the GPU BAR space of 128 KiB.
 *
 * NOTE: This sample can run for awhile if file size is >= 1 GiB
 *
 * ./many_buffers_set_bar <testfile1> <testfile2> <gpuid>
 *
 * | Output |	
 * Setting max pinned memory size to: 131072 bytes
 * Successful read of 68719476736 bytes on thread 0 from fd 73 to GPU id <gpuid>
 * Successful read of 68719476736 bytes on thread 1 from fd 74 to GPU id <gpuid>
 */
#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <thread>
#include <array>
#include <iostream>
#include <future>
#include <algorithm>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

static constexpr size_t NUM_THREADS = 2;
static constexpr size_t NUM_GPU_BUFFERS = 64;
static constexpr size_t MAX_PINNED_MEM_SIZE = KiB(128);
static constexpr size_t CHUNK_SIZE = MiB(1);

/*
 * Each thread allocates GPU memory and invokes cuFileBufRegister
 * on the entire buffer. This is done once; after registering the
 * buffer, the thread reads data in chunks into their own set of 64
 * buffers. The GPU buffer acts as a streaming buffer, allowing data to be
 * directly DMA'ed into the registered memory. This approach is optimal for
 * performance.
 */
static void thread_fn(std::promise<int>&& funcRet, int tid, int gpuid, int fd) {
	std::array<void*, NUM_GPU_BUFFERS> devPtrs;
	size_t fileSize = 0, totalBytesRead = 0;
	size_t allocated_buffers = 0;
	devPtrs.fill(nullptr);
	bool success = true;
	loff_t offset = 0;
	size_t nr_ios = 0;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle = nullptr;

	std::memset(static_cast<void*>(&cfr_descr), 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Error: File register error: " << cuFileGetErrorString(status) << std::endl;
		funcRet.set_value(-1);
		return;
	}

	// Set to the same PCI hierarchy of the peer device
	check_cudaruntimecall_thread(cudaSetDevice(gpuid), funcRet);

	// Register the 64 GPU buffers with cuFile
	for (size_t i = 0; i < NUM_GPU_BUFFERS; i++) {
		check_cudaruntimecall_thread(cudaMalloc(&devPtrs[i], CHUNK_SIZE), funcRet);

		status = cuFileBufRegister(devPtrs[i], CHUNK_SIZE, 0);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer register failed on buffer[" << 
				i << "]: " << cuFileGetErrorString(status) << std::endl;
			// Attempt clean up, free all allocated 
			// GPU buffers up to the current buffer[i]
			success = false;
			allocated_buffers = i;
			// Free the current buffer
			if (devPtrs[i]) check_cudaruntimecall_thread(cudaFree(devPtrs[i]), funcRet);
			goto deregister_bufferhandles;
		}
	}
	allocated_buffers = NUM_GPU_BUFFERS;

	// Determine the number of IOs to perform
	fileSize = GetFileSize(fd);
	nr_ios = fileSize / CHUNK_SIZE;
	if (fileSize % CHUNK_SIZE)
		nr_ios++;

	// Read the file in chunks of CHUNK_SIZE for NUM_GPU_BUFFERS times
	for (size_t i = 0; i < NUM_GPU_BUFFERS && success; i++) {
		offset = 0;
		for (size_t j = 0; j < nr_ios; j++) {
			int ret = cuFileRead(cfr_handle, devPtrs[i], CHUNK_SIZE, offset, 0);
			if (ret < 0) {
				std::cerr << "cuFile Read failed for thread " << tid << ": " 
					<< cuFileGetErrorString(ret) << std::endl;
				success = false;
				break;
			}
			offset += ret;
			totalBytesRead += ret;
		}
	}

// Cleanup labels
deregister_bufferhandles:
	for (size_t i = 0; i < allocated_buffers; i++) {
		status = cuFileBufDeregister(devPtrs[i]);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer deregister failed on buffer[" << 
				i << "]: " << cuFileGetErrorString(status) << std::endl;
			success = false;
		}

		if (devPtrs[i]) check_cudaruntimecall_thread(cudaFree(devPtrs[i]), funcRet);
	}

	if (cfr_handle) cuFileHandleDeregister(cfr_handle);

	if (success) {
		std::cout << "Successful read of " << totalBytesRead << " bytes on thread "
			<< tid << " from fd " << fd << " to GPU id " << gpuid << std::endl;
		funcRet.set_value(0);
	} else {
		funcRet.set_value(-1);
	}
}

int main(int argc, char **argv) {
	std::array<std::future<int>, NUM_THREADS> threadFutures;
	std::array<std::thread, NUM_THREADS> threadPool;
	std::array<int, NUM_THREADS> fds;
	int ret = EXIT_SUCCESS, gpuid = 0;
	CUfileDrvProps_t props = {};
	CUfileError_t status;

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

	// Set the maximum pinned memory size to 128 KiB
	status = cuFileDriverSetMaxPinnedMemSize(MAX_PINNED_MEM_SIZE);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Error: cuFileDriverSetMaxPinnedMemSize failed" << std::endl;
		return EXIT_FAILURE;
	}
	status = cuFileDriverGetProperties(&props);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Error: cuFileDriverGetProperties failed" << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "Setting max pinned memory size to: " << props.max_device_pinned_mem_size << " bytes" << std::endl;

	// Open files and get file descriptors
	for (size_t tid = 0; tid < NUM_THREADS; tid++) {
		fds[tid] = open(argv[tid + 1], O_RDONLY | O_DIRECT);
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
