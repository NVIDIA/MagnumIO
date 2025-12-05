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
 * This is a data-integrity test for cuFileRead/Write APIs with different
 * memory types. Managed memory is a special case where it does not need
 * cuFileBufRegister as internal pool buffers are used for actual IO, which
 * is then copied to MallocManaged memory via cuMemcpyPeer.
 *
 * The test does the following:
 * 1. Creates a Test file with pattern
 * 2. Test file is loaded to device memory (cuFileRead)
 * 3. From device memory, data is written to a new file (cuFileWrite)
 * 4. Test file and new file are compared for data integrity
 *
 * ./various_mem_rw <testReadFile> <testWriteFile> <gpuid> <mode(1:DeviceMemory, 2:ManagedMemory, 3:HostMemory)>
 *
 * | Output |
 * Using <memorytype>
 * Reading file to device memory: <testReadFile>
 * Writing device memory to file: <testWriteFile>
 * <iDigest>
 * <oDigest>
 * SHA256 SUM Match
 */
 
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <array>

#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <openssl/sha.h>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t MAX_BUF_SIZE = MiB(1);

enum MemoryType {
	DeviceMemory = 1,
	ManagedMemory = 2,
	HostMemory = 3,
};

static void *allocMemory(size_t size, MemoryType memType) {
	void *devPtr = nullptr;
	switch (memType) {
		case MemoryType::DeviceMemory:
			check_cudaruntimecall(cudaMalloc(&devPtr, size));
			std::cout << "Using cudaMalloc" << std::endl;
			break;
		case MemoryType::ManagedMemory:
			check_cudaruntimecall(cudaMallocManaged(&devPtr, size, cudaMemAttachHost));
			std::cout << "Using cudaMallocManaged" << std::endl;
			break;
		case MemoryType::HostMemory:
			check_cudaruntimecall(cudaMallocHost(&devPtr, size));
			std::cout << "Using cudaMallocHost" << std::endl;
	}
	return devPtr;
}

static void freeMemory(void *devPtr, enum MemoryType memType) {
	switch (memType) {
		case MemoryType::DeviceMemory:
		case MemoryType::ManagedMemory:
			check_cudaruntimecall(cudaFree(devPtr));
			break;
		case MemoryType::HostMemory:
			check_cudaruntimecall(cudaFreeHost(devPtr));
	}
}

int main(int argc, char *argv[]) {
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;
	std::string testReadFile, testWriteFile;
	const size_t size = MAX_BUF_SIZE;
	void *devPtr = nullptr;
	int fd = -1, mode = -1;
	MemoryType mem_type;
	ssize_t ret = EXIT_SUCCESS;

	// cuFile specific variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle = nullptr;

	if (argc < 5) {
		std::cerr << "Usage: " << argv[0] << " <testReadFile> <testWriteFile> <gpuid> "
			<< "<mode(1:DeviceMemory, 2:ManagedMemory, 3:HostMemory)> "
			<< std::endl;
		return EXIT_FAILURE;
	}

	testReadFile = argv[1];
	testWriteFile = argv[2];
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[3])));

	// Parse memory allocation mode from user
	mode = parseInt(argv[4]);
	if (mode < DeviceMemory || mode > HostMemory) {
		std::cerr << "Invalid mode: " << mode << ", "
			<< "expected (1:DeviceMemory, 2:ManagedMemory, 3:HostMemory)"
			<< std::endl;
		return EXIT_FAILURE;
	}
	mem_type = static_cast<MemoryType>(mode);

	// Create a Test file using standard Posix File IO calls
	if (!createTestFile(testReadFile, size)) {
		std::cerr << "Failed to create test file: " << testReadFile << std::endl;
		return EXIT_FAILURE;
	}

	// Load Test file to GPU memory
	fd = open(testReadFile.c_str(), O_RDONLY | O_DIRECT);
	if (fd < 0) {
		std::cerr << "Read file open error: " << testReadFile << " "
			<< cuFileGetErrorString(errno) << std::endl;
		return EXIT_FAILURE;
	}

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error:" << cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto close_file;
	}

	// Allocate memory for the test file either on device, managed or host
	devPtr = allocMemory(size, mem_type);
	if (devPtr == nullptr) {
		std::cerr << "Memory allocation failed" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}
	if (mem_type != MemoryType::HostMemory) {
		check_cudaruntimecall(cudaMemset(devPtr, 0, size));
	} else {
		std::memset(devPtr, 0, size);
	}
	check_cudaruntimecall(cudaStreamSynchronize(0));

	std::cout << "Reading file to device memory: " << testReadFile << std::endl;
	ret = cuFileRead(cf_handle, devPtr, size, 0, 0);
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

	// Write loaded data from GPU memory to a new file
	fd = open(testWriteFile.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (fd < 0) {
		std::cerr << "Write file open error: " << testWriteFile << " "
			<< cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto cuda_cleanup;
	}

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error: " << cuFileGetErrorString(status) << std::endl;
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

	// Compare file signatures
	if (SHASUM256(testReadFile.c_str(), iDigest, size) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandle;
	}
	DumpSHASUM(iDigest);

	if (SHASUM256(testWriteFile.c_str(), oDigest, size) < 0) {
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
	if (devPtr) freeMemory(devPtr, mem_type);

	return ret;
}
