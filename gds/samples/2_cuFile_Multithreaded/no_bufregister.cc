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
 * This sample shows how two threads who share the same CUFileHandle
 * perform READs at incremental GPU memory offsets without buffer
 * registration. NOTE: CHUNK_SIZE is 100 MB and MAX_FILE_SIZE is 1 GiB.
 * 
 * ./no_bufregister <testfile> <gpuid1> <gpuid2>
 *
 * | Output |	
 * Successful read of 1073741824 bytes for file <testfile> on thread 0 to GPU id <gpuid1>
 * Successful read of 1073741824 bytes for file <testfile> on thread 1 to GPU id <gpuid2>
 */

#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

#include <iostream>
#include <thread>
#include <array>
#include <future>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

static constexpr size_t NUM_THREADS = 2;
static constexpr size_t MAX_FILE_SIZE = GiB(1);
static constexpr size_t CHUNK_SIZE = MiB(100);

static void thread_fn(std::promise<int>&& funcRet, int tid, int fd, int gpuid, CUfileHandle_t& cf_handle, std::string file) {
	size_t fileSize = 0, totalBytesRead = 0;
	void *devPtr = nullptr;
	bool success = true;
	loff_t offset = 0;
	int nr_ios = 0;

	check_cudaruntimecall_thread(cudaSetDevice(gpuid), funcRet);

	/*
	 * Each thread allocates 1 GB of GPU Memory;
	 * However, there is no cuFileBufRegister now.
	 */
	check_cudaruntimecall_thread(cudaMalloc(&devPtr, MAX_FILE_SIZE), funcRet);

	// Assume the file size is less than allocated GPU memory size
	fileSize = GetFileSize(fd);
	nr_ios = fileSize / CHUNK_SIZE;
	if (fileSize % CHUNK_SIZE)
		nr_ios++;

	for (int i = 0; i < nr_ios; i++) {
		/*
		 * We are filling up the entire 1GB of GPU memory by reading file
		 * in chunks of 100 MB. When we have to do such incremental reads
		 * at different GPU Buffer offsets, we should avoid registering
		 * buffers using cuFileBufRegister in the loop. 
		 *
		 *  Ex:
		 *
		 *  for (...) {
		 *  	cuFileBufRegister(static_cast<char*>(devPtr) + offset, CHUNK_SIZE);
		 *  	cuFileRead(cf_handle, static_cast<char*>(devPtr) + offset, CHUNK_SIZE, offset, 0);
		 *  	offset += CHUNK_SIZE;
		 *  	cuFileBufDeregister(static_cast<char*>(devPtr) + offset);
		 *  }
		 *
		 * The above code snippet Registers and Deregisters buffer in a loop at different
		 * GPU Buffer offsets. This is a very expensive operation and should be avoided.
		 * If this is the use case, we can skip the cuFileBufRegister altogether and 
		 * invoke cuFileRead directly to avoid the overhead. GDS Will use internal caching
		 * to transfer the data to the GPU Buffer.
		 */
		int ret = cuFileRead(cf_handle, static_cast<char*>(devPtr) + offset, CHUNK_SIZE, offset, 0);
		if (ret < 0) {
			std::cerr << "cuFile Read failed for thread " << tid << ": " 
				<< cuFileGetErrorString(ret) << std::endl;
			success = false;
			break;
		}

		offset += ret;
		totalBytesRead += ret;
	}

	check_cudaruntimecall_thread(cudaFree(devPtr), funcRet);

	// Set the return value and print the result
	if (success) {
		std::cout << "Successful read of " << totalBytesRead << " bytes for file " 
			<< file << " on thread " << tid << " to GPU id " << gpuid << std::endl;
		funcRet.set_value(0);
	} else {
		funcRet.set_value(-1);
	}
}

int main(int argc, char **argv) {
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

	// Open the GDS driver
	status = cuFileDriverOpen();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "GDS Driver open failed: " << cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	// Open the file and get the file descriptor
	fd = open(file.c_str(), O_RDWR | O_DIRECT);
	if (fd < 0) {
		std::cerr << "Error: Unable to open file " << file << " fd " << fd << std::endl;
		ret = EXIT_FAILURE;
		goto close_driver;
	}

	// Verify that the file size is less than what is allocated for the GPU memory
	if (GetFileSize(fd) > MAX_FILE_SIZE) {
		std::cerr << "Error: File size is greater than the maximum file size of 1 GiB or " 
			<< MAX_FILE_SIZE << " bytes" << std::endl;
		ret = EXIT_FAILURE;
		goto close_file;
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
close_driver:
	status = cuFileDriverClose();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver close failed: " 
			<< cuFileGetErrorString(status) << std::endl;
	}

	return ret;
}
