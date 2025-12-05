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
 * This is a simple data-integrity test for cuFileReadAsync/WriteAsync 
 * APIs using default stream.
 * The test does the following:
 * 1. Creates a Test file with a random pattern
 * 2. Test file is loaded to device memory using cuFileReadAsync
 * 3. From device memory, data is written to a new file using cuFileWriteAsync
 * 4. Test file and new file are compared for data integrity using SHA-256
 *
 * ./async_rw_default <testWriteReadFile> <testWriteFile> <gpuid>
 *
 * | Output |
 * Using default stream for cuFileAsync I/O
 * Submit read to device memory from file: <testWriteReadFile> of size: 1048576
 * Reading submit done to file: <testWriteReadFile>
 * Submit write from dev memory to file: <testWriteFile> of size: 1048576
 * Writing submit done to file: <testWriteFile>
 * <iDigest>
 * <oDigest>
 * SHA256 SUM Match
 */

#include <iostream>
#include <array>
#include <string>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

// Include CUDA runtime header file
#include <cuda_runtime.h>
// Include cuFile header file
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t MAX_BUF_SIZE = MiB(1);

struct IOArgs {
	void *devPtr;
	size_t max_size;
	off_t offset;
	off_t buf_off;
	ssize_t read_bytes_done;
	ssize_t write_bytes_done;
};

int main(int argc, char *argv[]) {
	int wfd = -1, rfd = -1;
	ssize_t ret = EXIT_SUCCESS;
	Prng prng(255);
	std::string testWriteReadFile, testWriteFile;
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;
	IOArgs args = {};

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_rhandle = nullptr;
	CUfileHandle_t cf_whandle = nullptr;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testWriteReadFile> <testWriteFile> <gpuid> " << std::endl;
		return EXIT_FAILURE;
	}

	testWriteReadFile = argv[1];
	testWriteFile = argv[2];
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[3])));

	std::cout << "Using default stream for cuFileAsync I/O" << std::endl;

	// Initialize IOArgs
	args.devPtr = nullptr;
	args.max_size = MAX_BUF_SIZE;
	args.offset = 0;
	args.buf_off = 0;
	args.read_bytes_done = 0;
	args.write_bytes_done = 0;

	if (testWriteReadFile == testWriteFile) {
		std::cerr << "Test file and write file cannot be the same for verification" << std::endl;
		return EXIT_FAILURE;
	}

	// Create testWriteReadFile file using standard Posix File IO calls and fill it with random data
	if (!createTestFile(testWriteReadFile, args.max_size)) {
		std::cerr << "Failed to create test file: " << testWriteReadFile << std::endl;
		return EXIT_FAILURE;
	}

	// Allocate device Memory (fill to remove any holes) and register with cuFile
	check_cudaruntimecall(cudaMalloc(&args.devPtr, args.max_size));
	check_cudaruntimecall(cudaMemsetAsync(args.devPtr, 0, args.max_size, 0));
	
	// Register buffers. For unregistered buffers, this call is not required.
	status = cuFileBufRegister(args.devPtr, args.max_size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Buffer register failed: "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	// Open testWriteReadFile in O_DIRECT and read-only mode
	rfd = open(testWriteReadFile.c_str(), O_RDONLY | O_DIRECT);
	if (rfd < 0) {
		std::cerr << "Read file open error: " << testWriteReadFile << " "
			<< cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandle;
	}

	// Open testWriteFile in O_DIRECT and write-only mode
	wfd = open(testWriteFile.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0664);
	if (wfd < 0) {
		std::cerr << "Write file open error: " << testWriteFile << " "
			<< cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto close_files;
	}

	// Register the filehandles
	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = rfd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_rhandle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto close_files;
	}

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = wfd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_whandle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}

	// Read data from file to device memory
	std::cout << "Submit read to device memory from file: " << testWriteReadFile 
		<< " of size: " << args.max_size << std::endl;
	status = cuFileReadAsync(
		cf_rhandle, static_cast<unsigned char*>(args.devPtr),
		&args.max_size, &args.offset, &args.buf_off,
		&args.read_bytes_done, 0
	);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Read failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}
	std::cout << "Reading submit done to file: " << testWriteReadFile << std::endl;

	// Write data from device memory to file
	std::cout << "Submit write from dev memory to file: " << testWriteFile
		<< " of size: " << args.max_size << std::endl;
	status = cuFileWriteAsync(
		cf_whandle, static_cast<unsigned char*>(args.devPtr),
		&args.max_size, &args.offset, &args.buf_off,
		&args.write_bytes_done, 0
	);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "Write failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}
	std::cout << "Writing submit done to file: " << testWriteFile << std::endl;

	// Wait for the GPU to finish all async operations
	check_cudaruntimecall(cudaStreamSynchronize(0));

	if ((args.read_bytes_done < static_cast<ssize_t>(args.max_size)) ||
		(args.write_bytes_done < args.read_bytes_done)) {
		std::cerr << "IO error issued size: " << args.max_size
			<< ", read: " << args.read_bytes_done
			<< ", write: " << args.write_bytes_done << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}

	// Compare file signatures
	if (SHASUM256(testWriteReadFile, iDigest, args.max_size) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}
	DumpSHASUM(iDigest);

	if (SHASUM256(testWriteFile, oDigest, args.max_size) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
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
deregister_filehandles:
	if (cf_rhandle != nullptr) cuFileHandleDeregister(cf_rhandle);
	if (cf_whandle != nullptr) cuFileHandleDeregister(cf_whandle);
close_files:
	if (wfd >= 0) close(wfd);
	if (rfd >= 0) close(rfd);
deregister_bufferhandle:
	if (args.devPtr != nullptr) {
		status = cuFileBufDeregister(args.devPtr);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer deregister failed: "
				<< cuFileGetErrorString(status) << std::endl;
			ret = EXIT_FAILURE;
		}
		check_cudaruntimecall(cudaFree(args.devPtr));
	}

	return ret;
}
