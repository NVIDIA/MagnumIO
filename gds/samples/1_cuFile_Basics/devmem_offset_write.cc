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
 * Sample cuFileWrite Test involving device buffer offsets.
 * The test program reads from a file using cuFileRead
 * and writes the contents from an offset inside device memory at some offset.
 * For validation, we match the SHASUM signature of device memory
 * contents from the respective buffer offset and the newly created file contents
 *
 * ./devmem_offset_write <testRandomFile> <testFile> <gpuid>
 *
 * | Output |
 * Reading file to device memory: <testRandomFile>
 * Bytes read to device memory: 1048576
 * Writing from device memory, buffer OFFSET: 131072 to file: <testFile>
 * Bytes written to file: 917504
 * <iDigest>
 * <oDigest>
 * SHA256 SUM Match
 *
 * Required: testRandomFile must be already created and filled with random data
 * dd if=/dev/urandom of=testRandomFile bs=1G count=1 (NOTE: only <= 1M is used here)
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <array>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <openssl/sha.h>
#include <assert.h>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t MAX_BUFFER_SIZE = MiB(1);
static constexpr size_t TEST_BUFF_OFFSET = KiB(128);
static constexpr loff_t FILE_OFFSET = 0;

int main(int argc, char *argv[]) {
	int fd = -1;
	size_t size = 0;
	ssize_t ret = EXIT_SUCCESS;
	void *devPtr = nullptr;
	std::string testRandomFile, testFile;
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testRandomFile> <testFile> <gpuid>" << std::endl;
		return EXIT_FAILURE;
	}

	testRandomFile = argv[1];
	testFile = argv[2];
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[3])));

	fd = open(testRandomFile.c_str(), O_RDONLY | O_DIRECT, 0);
	if (fd < 0) {
		std::cerr << "File open error: " << testRandomFile << " "
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

	size = GetFileSize(fd);
	if (size == 0) {
		std::cerr << "File size is empty: " << testRandomFile << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}
	size = std::min(size, MAX_BUFFER_SIZE);
	assert(size > TEST_BUFF_OFFSET);
	check_cudaruntimecall(cudaMalloc(&devPtr, size));

	status = cuFileBufRegister(devPtr, size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}

	// Read a file with random data to device memory	
	std::cout << "Reading file to device memory: " << testRandomFile << std::endl;
	ret = cuFileRead(cf_handle, static_cast<void*>(devPtr), size, FILE_OFFSET, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Read failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Read failed: " << cuFileGetErrorString(errno) << std::endl;

		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
	} else {
		std::cout << "Bytes read to device memory: " << ret << std::endl;
		ret = EXIT_SUCCESS;
	}

	cuFileHandleDeregister(cf_handle);
	close(fd);
	cf_handle = nullptr, fd = -1;

	// Write random data from device memory to file
	fd = open(testFile.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0664);
	if (fd < 0) {
		std::cerr << "Error opening test file " << testFile
			<< " : " << cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
	}

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
	}

	std::cout << "Writing from device memory, buffer OFFSET: " << TEST_BUFF_OFFSET << " to file: " << testFile << std::endl;
	ret = cuFileWrite(cf_handle, static_cast<void*>(devPtr), size - TEST_BUFF_OFFSET, FILE_OFFSET, TEST_BUFF_OFFSET);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Write failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Write failed: " << cuFileGetErrorString(errno) << std::endl;

		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
	} else {
		std::cout << "Bytes written to file: " << ret << std::endl;
		ret = EXIT_SUCCESS;
	}

	// Device memory shasum with bufOffset
	iDigest.fill(0);
	if (SHASUM256_MEM(OSMemoryType::DEVICE, static_cast<char*>(devPtr), size, iDigest, TEST_BUFF_OFFSET) < 0) {
		std::cerr << "SHA256 Device mem compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
	}
	DumpSHASUM(iDigest);

	// File shasum
	oDigest.fill(0);
	if (SHASUM256(testFile, oDigest, size - TEST_BUFF_OFFSET) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
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
deregister_bufferhandle:
	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer deregister failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
	}
deregister_filehandle:
	if (cf_handle) cuFileHandleDeregister(cf_handle);
close_file:
	if (fd >= 0) close(fd);

	// Free the device memory
	if (devPtr) check_cudaruntimecall(cudaFree(devPtr));

	return ret;
}
