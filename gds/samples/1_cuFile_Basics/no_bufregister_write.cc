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
 * Sample cuFileWrite Test WITHOUT explicit device memory registration.
 * Note: DriverOpen/Close is not needed and implicitly done.
 * User can verify the output file's data after write using:
 * hexdump -C <testFile>
 * 0000000 abab abab abab abab abab abab abab abab  |................|
 *
 * ./no_bufregister_write <testFile> <gpuid>
 *
 * | Output |
 * cuFileWrite without device memory registration
 * Open file <testFile> for writing
 * Allocate device memory of size: 1048576 on GPU ID: <gpuid>
 * Write from device memory to file <testFile>
 * Written bytes: 1048576
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <fcntl.h>

// Include CUDA runtime header file
#include <cuda_runtime.h>
// Include cuFile header file
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

static constexpr size_t SIZE = MiB(1);

int main(int argc, char *argv[]) {
	int fd = -1;
	int gpuid = -1;
	ssize_t ret = EXIT_SUCCESS;
	std::string testFile;
	void *devPtr = nullptr;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <testFile> <gpuid>" << std::endl;
		return EXIT_FAILURE;
	}

	testFile = argv[1];
	gpuid = parseInt(argv[2]);
	check_cudaruntimecall(cudaSetDevice(gpuid));

	std::cout << "cuFileWrite without device memory registration" << std::endl;

	// Open a file to write
	std::cout << "Open file " << testFile << " for writing" << std::endl;
	fd = open(testFile.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0664);
	if (fd < 0) {
		std::cerr << "File open error: "
			<< cuFileGetErrorString(errno) << std::endl;
		return EXIT_FAILURE;
	}

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto close_file;
	}

	// Allocates device memory
	check_cudaruntimecall(cudaGetDevice(&gpuid));
	std::cout << "Allocate device memory of size: " 
		<< SIZE << " on GPU ID: " << gpuid << std::endl;
	check_cudaruntimecall(cudaMalloc(&devPtr, SIZE));

	// Fill pattern to device memory
	check_cudaruntimecall(cudaMemset(devPtr, 0xab, SIZE));
	check_cudaruntimecall(cudaStreamSynchronize(0));
	check_cudaruntimecall(cudaGetDevice(&gpuid));

	// Writes device memory contents to a file
	// Note we skipped device memory registration using cuFileBufRegister
	std::cout << "Write from device memory to file " << testFile << std::endl;
	ret = cuFileWrite(cf_handle, devPtr, SIZE, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Write failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Write failed: " << cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
	} else {
		std::cout << "Written bytes: " << ret << std::endl;
		ret = EXIT_SUCCESS;
	}
	
	// Deregister the cuFile handle
	cuFileHandleDeregister(cf_handle);

// Cleanup labels
close_file:
	if (fd >= 0) close(fd);

	// Free the device memory
	if (devPtr) check_cudaruntimecall(cudaFree(devPtr));

	return ret;
}
