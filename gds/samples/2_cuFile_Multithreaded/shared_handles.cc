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
 * This sample shows how two threads can share the same CUfileHandle_t.
 * This program opens the file in the main thread and then the same file
 * descriptor is shared by both threads.
 *
 * ./shared_handles <testfile> <gpuid1> <gpuid2>
 *
 * | Output |	
 * Successful read of 1073741824 bytes for file <testfile> on thread 0 to GPU id <gpuid1>
 * Successful read of 1073741824 bytes for file <testfile> on thread 1 to GPU id <gpuid2>
 */

#include <iostream>
#include <thread>
#include <array>
#include <future>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
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
 * buffer, the thread reads data in chunks using shared file handle
 * into their own buffer. The GPU buffer acts as a streaming buffer,
 * allowing data to be directly DMA'ed into the registered memory.
 * This approach is optimal for performance.
 */
static void thread_fn(std::promise<int>&& funcRet, int tid, int fd, int gpuid, CUfileHandle_t& cf_handle, std::string file) {
	size_t fileSize = 0, totalBytesRead = 0;
	void *devPtr = nullptr;
	CUfileError_t status;
	bool success = true;
	loff_t offset = 0;
	int nr_ios = 0;

	check_cudaruntimecall_thread(cudaSetDevice(gpuid), funcRet);
	check_cudaruntimecall_thread(cudaMalloc(&devPtr, CHUNK_SIZE), funcRet);

	// Register the buffer with cuFile
	status = cuFileBufRegister(devPtr, CHUNK_SIZE, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed for thread " << tid << ": " 
			<< cuFileGetErrorString(status) << std::endl;
		check_cudaruntimecall_thread(cudaFree(devPtr), funcRet);
		funcRet.set_value(-1);
		return;
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

	// Free the device memory
	if (devPtr) check_cudaruntimecall_thread(cudaFree(devPtr), funcRet);

	// Set the return value and print the result
	if (success) {
		std::cout << "Successful read of " << totalBytesRead << " bytes for file " 
			<< file << " on thread " << tid << " to GPU id " << gpuid << std::endl;
		funcRet.set_value(0);
	} else {
		funcRet.set_value(-1);
	}
}
 
int main(int argc, char *argv[]) {
	std::array<std::future<int>, NUM_THREADS> threadFutures;
	std::array<std::thread, NUM_THREADS> threadPool;
	std::array<int, NUM_THREADS> gpuIDs;
	int fd = -1, ret = EXIT_SUCCESS;
	std::string file;

	// cuFile specific variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testfile> <gpuid1> <gpuid2> " << std::endl;
		return EXIT_FAILURE;
	}

	file = argv[1];
	for (size_t i = 0; i < NUM_THREADS; i++)
		gpuIDs[i] = parseInt(argv[i + 2]);

	// Open the file and get the file descriptor
	fd = open(file.c_str(), O_RDWR | O_DIRECT);
	if (fd < 0) {
		std::cerr << "Error: Unable to open file " << file << " fd " << fd << std::endl;
		return EXIT_FAILURE;
	}

	// Register the file handle that will be shared by all threads
	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: " << cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto close_file;
	}

	// Spawn out each thread with the same cuFile Handle
	for (size_t tid = 0; tid < NUM_THREADS; tid++) {
		std::promise<int> funcRet;
		threadFutures[tid] = funcRet.get_future();
		threadPool[tid] = std::thread(thread_fn, std::move(funcRet), tid, fd, gpuIDs[tid], std::ref(cf_handle), file);
	}

	// Wait for all threads to finish
	for (size_t tid = 0; tid < NUM_THREADS; tid++) {
		threadPool[tid].join();
		if (threadFutures[tid].get() != 0) {
			std::cerr << "Error: Thread " << tid << " exited with failure" << std::endl;
			ret = EXIT_FAILURE;
		}
	}

	if (cf_handle) cuFileHandleDeregister(cf_handle);

// Cleanup labels
close_file:
	if (fd >= 0) close(fd);

	return ret;
}
 