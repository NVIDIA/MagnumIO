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
 * In this sample program, main thread will allocate 100 MB of GPU memory where
 * the entire GPU memory will be registered using cuFileBufRegister in the main
 * thread. Each thread will just read the data at different offsets. We simulate
 * 100 reads of 10 MB each per thread.
 *
 * ./shared_handles_offset <testfile> <gpuid>
 *
 * | Output |
 * Successful read of 1048576000 bytes on thread 0 from fd with file offset 0 to GPU id 0 on buffer offset 0
 * Successful read of 1048576000 bytes on thread 7 from fd with file offset 73400320 to GPU id 0 on buffer offset 73400320
 * Successful read of 1048576000 bytes on thread 5 from fd with file offset 52428800 to GPU id 0 on buffer offset 52428800
 * ...
 * Successful read of 1048576000 bytes on thread 8 from fd with file offset 83886080 to GPU id 0 on buffer offset 83886080
 *
 * Required: testfile must be already created and filled with random data
 * dd if=/dev/urandom of=testfile bs=1G count=1 (NOTE: only <= 100M is used here)
 */

#include <iostream>
#include <future>
#include <thread>
#include <array>
#include <cstdlib>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <cstring>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

constexpr int NUM_THREADS = 10;
constexpr int MEMORY_SIZE = MiB(100);
constexpr int READ_SIZE = MiB(10);
constexpr int SIMULATED_READS = 100;

/*
 * Note the usage of devPtr_offset. Every thread has same devPtr handle
 * which was registered using cuFileBufRegister; however all threads are
 * working at different buffer offsets. This is optimal as GPU memory is
 * registered once and no internal caching is used.
 */
static void thread_fn(std::promise<int>&& funcRet, int tid, int gpuid, void *devPtr, CUfileHandle_t cfr_handle, size_t fileOffset, size_t devPtrOffset) {	
	bool success = true;
	int bytesRead = 0, totalBytesRead = 0;
	
	check_cudaruntimecall_thread(cudaSetDevice(gpuid), funcRet);

	for (int i = 0; i < SIMULATED_READS; i++) {
		bytesRead = cuFileRead(cfr_handle, devPtr, READ_SIZE, fileOffset, devPtrOffset);
		if (bytesRead < 0) {
			std::cerr << "cuFileRead failed with ret=" << bytesRead << std::endl;
			success = false;
			break;
		}
		totalBytesRead += bytesRead;
	}

	if (success) {
		std::cout << "Successful read of " << totalBytesRead << " bytes on thread "
			<< tid << " from fd with file offset " << fileOffset << " to GPU id " 
			<< gpuid << " on buffer offset " << devPtrOffset << std::endl;
		funcRet.set_value(0);
	} else {
		funcRet.set_value(-1);
	}
}

int main(int argc, char **argv) {
	std::array<std::future<int>, NUM_THREADS> threadFutures;
	std::array<std::thread, NUM_THREADS> threadPool;
	int ret = EXIT_SUCCESS, gpuid = 0, fd = -1;
	void *devPtr = nullptr;
	size_t offset = 0;

	// cuFile specific variables
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle = nullptr;

	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <testfile> <gpuid> " << std::endl;
		return EXIT_FAILURE;
	}

	// Parse the GPU ID and open the file
	gpuid = parseInt(argv[2]);
	fd = open(argv[1], O_RDONLY | O_DIRECT);
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

	// Allocate memory on the GPU and register it with cuFile
	check_cudaruntimecall(cudaSetDevice(gpuid));
	check_cudaruntimecall(cudaMalloc(&devPtr, MEMORY_SIZE));
	status = cuFileBufRegister(devPtr, MEMORY_SIZE, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed: " << cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}
	
	for (int tid = 0; tid < NUM_THREADS; tid++) {
		std::promise<int> funcRet;
		threadFutures[tid] = funcRet.get_future();

		// Every thread will get same devPtr address and CUfileHandle_t
		threadPool[tid] = std::thread(thread_fn, std::move(funcRet), tid, gpuid, devPtr, cfr_handle, offset, offset);

		// Every thread will work on different offset
		offset += READ_SIZE;
	}

	for (int tid = 0; tid < NUM_THREADS; tid++) {
		threadPool[tid].join();
		if (threadFutures[tid].get() != 0) {
			std::cerr << "Error: Thread " << tid << " exited with failure" << std::endl;
			ret = EXIT_FAILURE;
		}
	}

	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFileBufDeregister failed: " << cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
	}

// Cleanup labels
deregister_filehandle:
	if (cfr_handle) cuFileHandleDeregister(cfr_handle);
	if (devPtr) check_cudaruntimecall(cudaFree(devPtr));
close_file:
	if (fd >= 0) close(fd);

	return ret;
}
