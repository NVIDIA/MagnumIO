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
 * In this sample, the main thread allocates 100 MiB of GPU memory and then spawns 10 threads.
 * Each thread reads 10 MiB of data from a file at a fixed offset (10 MiB × thread_index) 
 * and writes it into its corresponding region of the GPU memory.
 *
 * ./separate_handles_offset <testfile> <gpuid>
 *
 * | Output |
 * Successful read of 10485760 bytes on thread 0 from fd with file offset 0 to GPU id 0 on buffer offset 0
 * Successful read of 10485760 bytes on thread 4 from fd with file offset 41943040 to GPU id 0 on buffer offset 41943040
 * Successful read of 10485760 bytes on thread 6 from fd with file offset 62914560 to GPU id 0 on buffer offset 62914560
 * ...
 * Successful read of 10485760 bytes on thread 5 from fd with file offset 52428800 to GPU id 0 on buffer offset 52428800
 *
 * Required: testfile must be already created and filled with random data
 * dd if=/dev/urandom of=testfile bs=1G count=1 (NOTE: only <= 100M is used here)
 */

#include <iostream>
#include <future>
#include <thread>
#include <array>
#include <algorithm>
#include <stdlib.h>
#include <fcntl.h>
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

static void thread_fn(std::promise<int>&& funcRet, int tid, int gpuid, int fd, void *threadDevPtr, size_t offset) {
	bool success = true;
	int bytesRead = 0;

	// cuFile specific variables
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle = nullptr;
	
	/*
	 * We need to set the CUDA device; threads will not inherit main thread's
	 * CUDA context. In this case, since main thread allocated memory on GPU 0,
	 * we set it explicitly. However, threads have to ensure that they are in
	 * same cuda context as devPtr was allocated.
	 */
	check_cudaruntimecall_thread(cudaSetDevice(gpuid), funcRet);

	std::memset(static_cast<void*>(&cfr_descr), 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File handle register error: " << cuFileGetErrorString(status) << std::endl;
		funcRet.set_value(-1);
		return;
	}

	// Every thread is registering buffer at different devPtr address of size 10 MiB
	status = cuFileBufRegister(threadDevPtr, READ_SIZE, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed: " << cuFileGetErrorString(status) << std::endl;
		success = false;
		goto deregister_filehandle;
	}

	bytesRead = cuFileRead(cfr_handle, threadDevPtr, READ_SIZE, offset, 0);
	if (bytesRead < 0) {
		std::cerr << "cuFileRead failed with ret=" << bytesRead << std::endl;
		success = false;
		goto deregister_bufferhandle;
	}

// Cleanup labels
deregister_bufferhandle:
	status = cuFileBufDeregister(threadDevPtr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer deregister failed on buffer: " << cuFileGetErrorString(status) << std::endl;
		success = false;
	}

deregister_filehandle:
	if (cfr_handle) cuFileHandleDeregister(cfr_handle);

	if (success) {
		std::cout << "Successful read of " << bytesRead << " bytes on thread "
			<< tid << " from fd with file offset " << offset << " to GPU id " 
			<< gpuid << " on buffer offset " << offset << std::endl;
		funcRet.set_value(0);
	} else {
		funcRet.set_value(-1);
	}
}

int main(int argc, char *argv[]) {
	std::array<std::future<int>, NUM_THREADS> threadFutures;
	std::array<std::thread, NUM_THREADS> threadPool;
	std::array<int, NUM_THREADS> fds;
	int ret = EXIT_SUCCESS, gpuid = 0;
	void *devPtr = nullptr;
	size_t offset = 0;
	fds.fill(-1);

	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <testfile> <gpuid>" << std::endl;
		return EXIT_FAILURE;
	}

	gpuid = parseInt(argv[2]);

	// Cleanup function for file descriptors
	auto cleanupFds = [](std::array<int, NUM_THREADS>& fds) {
		std::for_each(fds.begin(), fds.end(), [](int fd) {
			if (fd > 0) close(fd);
		});
	};

	check_cudaruntimecall(cudaSetDevice(gpuid));
	check_cudaruntimecall(cudaMalloc(&devPtr, MEMORY_SIZE));
	
	for (int i = 0; i < NUM_THREADS; i++) {
		fds[i] = open(argv[1], O_RDONLY | O_DIRECT);
		if (fds[i] < 0) {
			std::cerr << "Error: Unable to open file " << argv[1] << " fd " << fds[i] << std::endl;
			// Close all file descriptors that were opened prior to this failure
			cleanupFds(fds);
			if (devPtr) check_cudaruntimecall(cudaFree(devPtr));
			return EXIT_FAILURE;
		}
	}

	for (int tid = 0; tid < NUM_THREADS; tid++) {
		std::promise<int> funcRet;
		threadFutures[tid] = funcRet.get_future();
		threadPool[tid] = std::thread(thread_fn, std::move(funcRet), tid, gpuid, fds[tid], static_cast<char*>(devPtr) + offset, offset);
		offset += READ_SIZE;
	}

	for (int tid = 0; tid < NUM_THREADS; tid++) {
		threadPool[tid].join();
		if (threadFutures[tid].get() != 0) {
			std::cerr << "Error: Thread " << tid << " exited with failure" << std::endl;
			ret = EXIT_FAILURE;
		}
	}

	cleanupFds(fds);
	if (devPtr) check_cudaruntimecall(cudaFree(devPtr));
	return ret;
}
