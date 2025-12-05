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
 * This sample test reads a file iteratively till a given size (16MB).
 * In each iteration, GPU device memory offsets are passed
 * using cuFileRead until the desired length is read. This does
 * not use cuFileBufRegister, because of which the buffer offset parameter
 * is always kept zero.
 * For verification, we write the device memory to a file
 * and compare the signatures
 *
 * ./iterative_read <testFile> <testWriteFile> <gpuid>
 * 
 * | Output |
 * Allocated 16777216 bytes of device memory
 * Reading file sequentially: <testFile> with chunk size: 65536
 * Total chunks read: 256
 * Writing device memory to file: <testWriteFile>
 * <iDigest>
 * <oDigest>
 * SHA256 SUM Match
 *
 * Required: testFile must be already created and filled with random data
 * dd if=/dev/urandom of=testFile bs=16M count=1 (NOTE: only <= 1M is used here)
 */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <fcntl.h>
#include <assert.h>
#include <unistd.h>

// Include CUDA runtime header file
#include <cuda_runtime.h>
// Include cuFile header file
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t MAX_DEVICE_MEMORY_SIZE = MiB(16);
static constexpr size_t CHUNK_SIZE = KiB(64);

int main(int argc, char *argv[]) {
	int fd = -1;
	ssize_t ret = EXIT_SUCCESS, count = 0;
	void *devPtr = nullptr;
	size_t size = 0, nbytes = 0, bufOff = 0, fileOff = 0;
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;
	std::string testFile, testWriteFile;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testFile> <testWriteFile> <gpuid> " << std::endl;
		return EXIT_FAILURE;
	}

	testFile = argv[1];
	testWriteFile = argv[2];
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[3])));

	// Read the file into device memory
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

	// Allocate device memory based on the file size
	size = GetFileSize(fd);
	if (size == 0) {
		std::cerr << "File size is empty: " << testFile << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}

	// Allocate device memory based on the file size
	size = std::min(size, MAX_DEVICE_MEMORY_SIZE);
	check_cudaruntimecall(cudaMalloc(&devPtr, size));
	check_cudaruntimecall(cudaMemset(devPtr, 0x00, size));
	check_cudaruntimecall(cudaStreamSynchronize(0));
	std::cout << "Allocated " << size << " bytes of device memory" << std::endl;

	std::cout << "Reading file sequentially: " << testFile
			<< " with chunk size: " << CHUNK_SIZE << std::endl;
	do {
		nbytes = std::min((size - fileOff), CHUNK_SIZE);
		ret = cuFileRead(cf_handle, static_cast<char*>(devPtr) + bufOff, nbytes, fileOff, 0);
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

		bufOff += nbytes;
		fileOff += nbytes;
		count++;
	} while (fileOff < size);
	std::cout << "Total chunks read: " <<  count << std::endl;

	// Close the file and deregister the file handle for future registration
	cuFileHandleDeregister(cf_handle);
	close(fd);
	cf_handle = nullptr, fd = -1;

	/* Write from device memory to a file */
	fd = open(testWriteFile.c_str(), O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, 0664);
	if (fd < 0) {
		std::cerr << "File open error: " << testWriteFile << " "
			<< cuFileGetErrorString(errno) << std::endl;
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
		goto close_file;
	}

	std::cout << "Writing device memory to file: " << testWriteFile << std::endl;
	ret = cuFileWrite(cf_handle, devPtr, size, 0, 0);
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

	iDigest.fill(0);
	if (SHASUM256(testFile, iDigest, size) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}
	DumpSHASUM(iDigest);

	oDigest.fill(0);
	if (SHASUM256(testWriteFile, oDigest, size) < 0) {
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
