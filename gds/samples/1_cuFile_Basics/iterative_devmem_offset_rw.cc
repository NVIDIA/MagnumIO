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
 * This sample test reads up to 1 MiB of a file iteratively into GPU device memory.
 * In each iteration, GPU device memory offsets are passed using cuFileRead 
 * until the whole file is consumed. Then, the device memory is written to a
 * file for verification.
 *
 * ./iterative_devmem_offset_rw <testfile1> <testWriteFile> <gpuid>
 *
 * | Output |	
 * Opening file: <testFile>
 * Registering device memory of size: 1048576
 * Reading file sequentially: <testFile> chunk size: 65536
 * Total chunks read: 16
 * Writing device memory to file: <testWriteFile>
 * Wrote bytes: 1048576
 * <iDigest>
 * <oDigest>
 * SHA256 SUM Match
 *
 * Required: testFile must be already created and filled with random data
 * dd if=/dev/urandom of=testFile bs=16M count=1 (NOTE: only <= 1M is used here)
 */

#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <algorithm>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t CHUNK_SIZE = KiB(64);
static constexpr size_t MAX_BUFFER_SIZE = MiB(1);

int main(int argc, char *argv[]) {
	size_t size = 0, totalBytes = 0, nbytes = 0, bufOff = 0, fileOff = 0;
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;
	std::string testFile, testWriteFile;
	ssize_t ret = EXIT_SUCCESS, count = 0;
	void *devPtr = nullptr;
	int fd = -1;

	// cuFile specific variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testFile> <testWriteFile> <gpuid>" << std::endl;
		return EXIT_FAILURE;
	}

	testFile = argv[1];
	testWriteFile = argv[2];
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[3])));

	std::cout << "Opening file: " << testFile << std::endl;
	fd = open(testFile.c_str(), O_RDONLY | O_DIRECT);
	if (fd < 0) {
		std::cerr << "File open error: " << testFile << " "
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
		std::cerr << "File size is zero: " << testFile << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}

	// Allocate device memory
	size = std::min(size, MAX_BUFFER_SIZE);
	check_cudaruntimecall(cudaMalloc(&devPtr, size));
	check_cudaruntimecall(cudaMemset(static_cast<void*>(devPtr), 0x00, size));
	check_cudaruntimecall(cudaStreamSynchronize(0));

	// Register the device memory with cuFile
	std::cout << "Registering device memory of size: " << size << std::endl;
	status = cuFileBufRegister(devPtr, size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}

	// Read the file in chunks
	std::cout << "Reading file sequentially: " << testFile
		<< " chunk size: " << CHUNK_SIZE << std::endl;
	do {
		nbytes = std::min(size, CHUNK_SIZE);
		int bytesRead = cuFileRead(cf_handle, devPtr, nbytes, fileOff, bufOff);
		if (bytesRead < 0) {
			if (IS_CUFILE_ERR(bytesRead))
				std::cerr << "Read failed: " << cuFileGetErrorString(bytesRead) << std::endl;
			else
				std::cerr << "Read failed: " << cuFileGetErrorString(errno) << std::endl;
			ret = EXIT_FAILURE;
			goto deregister_bufferhandle;
		} else {
			std::cout << "Bytes read to device memory: " << bytesRead << std::endl;
			ret = EXIT_SUCCESS;
		}

		totalBytes += bytesRead;
		bufOff += bytesRead;
		fileOff += bytesRead;
		count++;
	} while (totalBytes < size);

	std::cout << "Total chunks read: " <<  count << std::endl;

	cuFileHandleDeregister(cf_handle);
	close(fd);
	cf_handle = nullptr, fd = -1;

	fd = open(testWriteFile.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (fd < 0) {
		std::cerr << "File open error: " << testWriteFile << " "
			<< cuFileGetErrorString(errno) << std::endl;
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

	std::cout << "Writing device memory to file: " << testWriteFile << std::endl;
	ret = cuFileWrite(cf_handle, static_cast<void*>(devPtr), size, 0, 0);
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

	iDigest.fill(0);
	if (SHASUM256(testFile.c_str(), iDigest, size) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
	}
	DumpSHASUM(iDigest);

	oDigest.fill(0);
	if (SHASUM256(testWriteFile.c_str(), oDigest, size) < 0) {
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
