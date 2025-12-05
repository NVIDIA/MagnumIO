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
 * Sample cuFileWrite Test without O_DIRECT MODE.
 * This sample program writes data from GPU memory to a file.
 * For verification, input data has a pattern.
 * User can verify the output file's data after write using:
 * hexdump -C <testFile>
 * 0000000 abab abab abab abab abab abab abab abab  |................|
 *
 * ./no_odirect_write <testFile> <gpuid>
 *
 * | Output |
 * Opening file without O_DIRECT flag: <testFile>
 * Registering device memory of size: 131072
 * Writing from device memory
 * Written bytes: 131072
 * Deregistering device memory
 */

#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <unistd.h>

// Include CUDA runtime header file
#include <cuda_runtime.h>
// Include cuFile header file
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

static constexpr size_t SIZE = KiB(128);

int main(int argc, char *argv[]) {
	int fd = -1;
	ssize_t ret = EXIT_SUCCESS;
	std::string testFile;
	void *devPtr = nullptr;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <testFile> <gpuid> " << std::endl;
		return EXIT_FAILURE;
	}

	testFile = std::string(argv[1]);
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[2])));

	// Open the cuFile driver
	status = cuFileDriverOpen();
	if (status.err != CU_FILE_SUCCESS) {
		return EXIT_FAILURE;
	}

	// Opens a file to write
	std::cout << "Opening file without O_DIRECT flag: " << testFile << std::endl;
	fd = open(testFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0664);
	if (fd < 0) {
		std::cerr << "File open error: " << cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto close_driver;
	}

	// Initialize the cuFile descriptor
	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: " << cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto close_file;
	}

	// Allocate device memory and fill with 0xab
	check_cudaruntimecall(cudaMalloc(&devPtr, SIZE));
	check_cudaruntimecall(cudaMemset(static_cast<void*>(devPtr), 0xab, SIZE));
	check_cudaruntimecall(cudaStreamSynchronize(0));

	// Registers device memory
	std::cout << "Registering device memory of size: " << SIZE << std::endl;
	status = cuFileBufRegister(devPtr, SIZE, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}

	// Writes device memory contents to a file
	std::cout << "Writing from device memory" << std::endl;
	ret = cuFileWrite(cf_handle, devPtr, SIZE, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Write failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Write failed: " << cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
	} else {
		std::cout << "written bytes: " << ret << std::endl;
		ret = EXIT_SUCCESS;
	}

	// Deregister the device memory
	std::cout << "Deregistering device memory" << std::endl;
	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer deregister failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
	}

// Cleanup labels
deregister_filehandle:
	if (cf_handle) cuFileHandleDeregister(cf_handle);
close_file:
	if (fd >= 0) close(fd);
close_driver:
	status = cuFileDriverClose();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver close failed:"
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
	}

	// Free the device memory
	if (devPtr) check_cudaruntimecall(cudaFree(devPtr));

	return ret;
}
