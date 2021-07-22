/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/*
 *  This is a data-integrity test for cuFileRead/Write APIs.
 *  The test does the following:
 *  1. Creates a Test file with pattern
 *  2. Test file is loaded to device memory (cuFileRead)
 *  3. From device memory, data is written to a new file (cuFileWrite)
 *  4. Test file and new file are compared for data integrity
 *
 * ./CUFileTest005
 *
 * e9d2f73120b2f2b1d2782e8ef5a42a3259b3c2badc5edb6ee04d4bc7b7633a
 * e9d2f73120b2f2b1d2782e8ef5a42a3259b3c2badc5edb6ee04d4bc7b7633a
 * SHA SUM Match
 * API Version :
 * 440-442(us) : 1
 * 510-512(us) : 1
 *
 */
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <openssl/sha.h>

#include <cuda_runtime.h>

// include this header
#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace std;

// copy bytes
#define MAX_BUF_SIZE (31 * 1024 * 1024UL)

int main(int argc, char *argv[]) {
	int fd;
	ssize_t ret = -1;
	void *devPtr = NULL, *hostPtr = NULL;
	const size_t size = MAX_BUF_SIZE;
	CUfileError_t status;
        CUfileDescr_t cf_descr;
        CUfileHandle_t cf_handle;
	unsigned char iDigest[SHA256_DIGEST_LENGTH], oDigest[SHA256_DIGEST_LENGTH];
	Prng prng(255);
	const char *TEST_READWRITEFILE, *TEST_WRITEFILE;

        if(argc < 4) {
                std::cerr << argv[0] << " <readfilepath> <writefilepath> <gpuid> "<< std::endl;
                exit(1);
        }

        TEST_READWRITEFILE = argv[1];
        TEST_WRITEFILE = argv[2];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[3])));

	// Create a Test file using standard Posix File IO calls
	fd = open(TEST_READWRITEFILE, O_RDWR | O_CREAT, 0644);
	if (fd < 0) {
		std::cerr << "test file open error : " << TEST_READWRITEFILE << " "
			<< std::strerror(errno) << std::endl;
		return -1;
	}

	hostPtr = malloc(size);
	if (!hostPtr) {
		std::cerr << "buffer allocation failure : "
			<< std::strerror(errno) << std::endl;
		close(fd);
		return -1;
	}

	memset(hostPtr, prng.next_random_offset(), size);
	ret = write(fd, hostPtr, size);
	if (ret < 0) {
		std::cerr << "write failure : " << std::strerror(errno)
				<< std::endl;
		close(fd);
		free(hostPtr);
		return -1;
	}

	free(hostPtr);

	//fsync(fd);
	close(fd);

	// Load Test file to GPU memory
	fd = open(TEST_READWRITEFILE, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		std::cerr << "read file open error : " << TEST_READWRITEFILE << " "
			<< std::strerror(errno) << std::endl;
		return -1;
	}

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
                close(fd);
                return -1;
        }

	check_cudaruntimecall(cudaMalloc(&devPtr, size));
	// special case for holes
	check_cudaruntimecall(cudaMemset(devPtr, 0, size));

	std::cout << "reading file to device memory :" << TEST_READWRITEFILE
				<< std::endl;

	ret = cuFileRead(cf_handle, devPtr, size, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "read failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "read failed : "
				<< cuFileGetErrorString(errno) << std::endl;
		cuFileHandleDeregister(cf_handle);
		close(fd);
		check_cudaruntimecall(cudaFree(devPtr));
		return -1;
	}

	cuFileHandleDeregister(cf_handle);
	close (fd);

	// Write loaded data from GPU memory to a new file
	fd = open(TEST_WRITEFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (fd < 0) {
		std::cerr << "write file open error : " << std::strerror(errno)
				<< std::endl;
		check_cudaruntimecall(cudaFree(devPtr));
		return -1;
	}

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
                close(fd);
		check_cudaruntimecall(cudaFree(devPtr));
                return -1;
        }
	std::cout << "writing device memory to file :" << TEST_WRITEFILE << std::endl;

	ret = cuFileWrite(cf_handle, devPtr, size, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "write failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "write failed : "
				<< cuFileGetErrorString(errno) << std::endl;
		goto out;
	}

	// Compare file signatures
	ret = SHASUM256(TEST_READWRITEFILE, iDigest, size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto out;
        }

	DumpSHASUM(iDigest);

	ret = SHASUM256(TEST_WRITEFILE, oDigest, size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto out;
        }

	DumpSHASUM(oDigest);

	if (memcmp(iDigest, oDigest, SHA256_DIGEST_LENGTH) != 0) {
		std::cerr << "SHA SUM Mismatch" << std::endl;
		ret = -1;
	} else {
		std::cout << "SHA SUM Match" << std::endl;
		ret = 0;
	}

out:
	check_cudaruntimecall(cudaFree(devPtr));
	cuFileHandleDeregister(cf_handle);
	close(fd);
	return ret;
}
