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
 * This is a data-integrity test for cuFileRead/Write APIs 
 * where CUDA runtime APIs are utilized.
 * The test does the following:
 * 1. Creates a Test file with pattern
 * 2. Test file is loaded to device memory (cuFileRead)
 * 3. From device memory, data is written to a new file (cuFileWrite)
 * 4. Test file and new file are compared for data integrity
 *
 * ./runtime_rw_integrity <testWriteReadFile> <testWriteFile> <gpuid>
 *
 * | Output |
 * Reading file to device memory: <testWriteReadFile>
 * Bytes read to device memory: 32505856
 * Writing device memory to file: <testWriteFile>
 * Bytes written to file: 32505856
 * <iDigest>
 * <oDigest>
 * SHA256 SUM Match
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <array>
#include <unistd.h>
#include <fcntl.h>
#include <openssl/sha.h>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t SIZE = MiB(31);

int main(int argc, char *argv[]) {
	int fd = -1;
	ssize_t ret = EXIT_SUCCESS;
	void *devPtr = nullptr;
	std::string testWriteReadFile, testWriteFile;
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testWriteReadFile> <testWriteFile> <gpuid>" << std::endl;
		return EXIT_FAILURE;
	}

	testWriteReadFile = std::string(argv[1]);
	testWriteFile = std::string(argv[2]);
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[3])));

	if (testWriteReadFile == testWriteFile) {
		std::cerr << "Test file and write file cannot be the same for verification" << std::endl;
		return EXIT_FAILURE;
	}

	// Create a Test file using standard POSIX File IO calls
	if (!createTestFile(testWriteReadFile, SIZE)) {
		std::cerr << "Test file creation failed" << std::endl;
		return EXIT_FAILURE;
	}

	// Read the test file to device memory
	fd = open(testWriteReadFile.c_str(), O_RDONLY | O_DIRECT);
	if (fd < 0) {
		std::cerr << "Read file open error: " << testWriteReadFile << " "
			<< cuFileGetErrorString(errno) << std::endl;
		return EXIT_FAILURE;
	}

	// Register the file handle
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

	// Allocate device memory and zero it out in case of holes
	check_cudaruntimecall(cudaMalloc(&devPtr, SIZE));
	check_cudaruntimecall(cudaMemset(devPtr, 0, SIZE));
	check_cudaruntimecall(cudaStreamSynchronize(0));
	std::cout << "Reading file to device memory: " << testWriteReadFile << std::endl;

	ret = cuFileRead(cf_handle, devPtr, SIZE, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Read failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Read failed: " << cuFileGetErrorString(errno) << std::endl;

		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	} else {
		std::cout << "Bytes read to device memory: " << ret << std::endl;
		ret = EXIT_SUCCESS;
	}

	cuFileHandleDeregister(cf_handle);
	close(fd);
	cf_handle = nullptr, fd = -1;

	// Write read data from device memory to a new file
	fd = open(testWriteFile.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0664);
	if (fd < 0) {
		std::cerr << "Write file open error: " << cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto cuda_cleanup;
	}

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: " 
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto cuda_cleanup;
	}

	std::cout << "Writing device memory to file: " << testWriteFile << std::endl;
	ret = cuFileWrite(cf_handle, devPtr, SIZE, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Write failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Write failed: " << cuFileGetErrorString(errno) << std::endl;

		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	} else {
		std::cout << "Bytes written to file: " << ret << std::endl;
		ret = EXIT_SUCCESS;
	}

	// Compare file signatures
	iDigest.fill(0);
	if (SHASUM256(testWriteReadFile, iDigest, SIZE) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}
	DumpSHASUM(iDigest);

	oDigest.fill(0);
	if (SHASUM256(testWriteFile, oDigest, SIZE) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}
	DumpSHASUM(oDigest);

	if (iDigest != oDigest) {
		std::cerr << "SHA256 SUM Mismatch" << std::endl;
		ret = EXIT_FAILURE;
	} else {
		std::cout << "SHA256 SUM Match" << std::endl;
		ret = EXIT_SUCCESS;
	}

// Cleanup labels
deregister_filehandle:
	if (cf_handle) cuFileHandleDeregister(cf_handle);
close_file:
	if (fd >= 0) close(fd);
cuda_cleanup:
	if (devPtr) check_cudaruntimecall(cudaFree(devPtr));

	return ret;
}
