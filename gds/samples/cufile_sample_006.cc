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
 * This sample test reads a file iteratively till a given size (16MB).
 * In each iteration, GPU device memory offsets are passed
 * using cuFileRead until the desired length is read. This does
 * not use cuFileBufRegister, because of which the buffer offset parameter
 * is always kept zero.
 *
 * For verification, we write the device memory to a file
 * and compare the signatures
 *
 */
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <fcntl.h>
#include <assert.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace std;

#define CHUNK_SIZE (64 * 1024UL)
#define MIN_FILE_SIZE (16 * 1024 * 1024UL)

int main(int argc, char *argv[]) {
	int fd, fdw;
	ssize_t ret = -1, count = 0;
	void *devPtr = NULL;
	size_t size, nbytes, bufOff = 0, fileOff = 0;
	CUfileError_t status;
	unsigned char iDigest[SHA256_DIGEST_LENGTH], oDigest[SHA256_DIGEST_LENGTH];
        const char *TESTFILE, *TESTWRITEFILE;
        CUfileDescr_t cf_descr, cf_descr2;
        CUfileHandle_t cf_handle, cf_handle2;

        if(argc < 4) {
                std::cerr << argv[0] << " <readfilepath> <writefilepath> <gpuid> "<< std::endl;
                exit(1);
        }

        TESTFILE = argv[1];
        TESTWRITEFILE = argv[2];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[3])));

	ret = open(TESTFILE, O_RDONLY | O_DIRECT);
	if (ret < 0) {
		std::cerr << "file open error : " << TESTFILE << " "
			<< cuFileGetErrorString(errno) << std::endl;
		return -1;
	}
	fd = ret;

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

	size = GetFileSize(fd);
	if (!size) {
		ret = -1;
		std::cerr << "file size empty:" << TESTFILE << std::endl;
		goto error;
        }

	size = std::min(size, MIN_FILE_SIZE);
	cout << "reading file size(bytes):" << size << std::endl;

	check_cudaruntimecall(cudaMalloc(&devPtr, size));
	check_cudaruntimecall(cudaMemset(devPtr, 0x00, size));

	std::cout << "reading file sequentially :" << TESTFILE
			  << " chunk size : " << CHUNK_SIZE <<  std::endl;
	do {
		nbytes = std::min((size - fileOff), CHUNK_SIZE);
		ret = cuFileRead(cf_handle, (char *) devPtr + bufOff, nbytes, fileOff, 0);
		if (ret < 0) {
			if (IS_CUFILE_ERR(ret))
				std::cerr << "read failed : "
					<< cuFileGetErrorString(ret) << std::endl;
			else
				std::cerr << "read failed : "
					<< cuFileGetErrorString(errno) << std::endl;
			goto error1;
		}
		bufOff += nbytes;
		fileOff += nbytes;
		count++;
	} while (fileOff < size);

	std::cout << "Total chunks read :" <<  count << std::endl;

	ret = open(TESTWRITEFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (ret < 0) {
		std::cerr << "file open error: " << TESTWRITEFILE << " "
			<< cuFileGetErrorString(errno) << std::endl;
		goto error1;
	}
	fdw = ret;

        memset((void *)&cf_descr2, 0, sizeof(CUfileDescr_t));
        cf_descr2.handle.fd = fdw;
        cf_descr2.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle2, &cf_descr2);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error:"
			<< cuFileGetErrorString(status) << std::endl;
		ret = -1;
		goto error2;
        }
	std::cout << "writing device memory to file :" << TESTWRITEFILE << std::endl;

	ret = cuFileWrite(cf_handle2, devPtr, size, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "write failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "write failed : "
				<< cuFileGetErrorString(errno) << std::endl;
		goto error2;
	}

	ret = SHASUM256(TESTFILE, iDigest, size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto error2;
        }

	DumpSHASUM(iDigest);

	ret = SHASUM256(TESTWRITEFILE, oDigest, size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto error2;
        }

	DumpSHASUM(oDigest);

	if (memcmp(iDigest, oDigest, SHA256_DIGEST_LENGTH) != 0) {
		ret = -1;
		std::cerr << "SHA SUM Mismatch" << std::endl;
	} else {
		ret = 0;
		std::cout << "SHA SUM Match" << std::endl;
	}

error2:
	cuFileHandleDeregister(cf_handle2);
	close(fdw);

error1:
	check_cudaruntimecall(cudaFree(devPtr));
error:
	cuFileHandleDeregister(cf_handle);
	close(fd);
	return ret;
}
