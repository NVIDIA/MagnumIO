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
 * This sample test reads an entire file iteratively.
 * In each iteration, GPU device memory offsets are passed
 * using cuFileRead until the whole file is loaded.
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

#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace std;

#define CHUNK_SIZE (64 * 1024UL)

#define MAX_BUFFER_SIZE (1 * 1024 * 1024UL) // 1 MB

int main(int argc, char *argv[]) {
	int fd = -1;
	void *devPtr = NULL;
	CUfileError_t status;
        CUfileDescr_t cf_descr;
        CUfileHandle_t cf_handle;
	ssize_t ret = -1, count = 0;
	size_t size, total_bytes = 0, nbytes, bufOff = 0, fileOff = 0;
	unsigned char iDigest[SHA256_DIGEST_LENGTH], oDigest[SHA256_DIGEST_LENGTH];
        const char *TESTFILE, *TESTWRITEFILE;

        if (argc < 4) {
                std::cerr << argv[0] << " <readfilepath> <writefilepath> <gpuid>"<< std::endl;
                exit(1);
        }

        TESTFILE = argv[1];
        TESTWRITEFILE = argv[2];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[3])));

	std::cout << "opening file " << TESTFILE << std::endl;
	ret = open(TESTFILE, O_RDONLY | O_DIRECT);
	if (ret < 0) {
		std::cerr << "file open error : " << TESTFILE << " "
			<< cuFileGetErrorString(errno) << std::endl;
		goto exit0;
	}
	fd = ret;

	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
		std::cerr << "file register error:"
			<< cuFileGetErrorString(status) << std::endl;
		goto exit0;
	}

	size = GetFileSize(fd);
	if (!size) {
		ret = -1;
		std::cerr << "file size is zero " << TESTFILE << std::endl;
		cuFileHandleDeregister(cf_handle);
		goto exit0;
	}

	size = std::min(size, MAX_BUFFER_SIZE);
	check_cudaruntimecall(cudaMalloc(&devPtr, size));
	check_cudaruntimecall(cudaMemset((void*)(devPtr), 0x00, size));

	std::cout << "registering device memory of size :" << size << std::endl;
	status = cuFileBufRegister(devPtr, size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
		std::cerr << "buf register failed :"
			<< cuFileGetErrorString(status) << std::endl;
		cuFileHandleDeregister(cf_handle);
		goto exit1;
	}

	std::cout << "reading file sequentially :" << TESTFILE
			  << " chunk size : " << CHUNK_SIZE <<  std::endl;
	do {
		nbytes = std::min(size, CHUNK_SIZE);
		ret = cuFileRead(cf_handle, devPtr, nbytes, fileOff, bufOff);
		if (ret < 0) {
			if (IS_CUFILE_ERR(ret))
				std::cerr << "read failed : "
					<< cuFileGetErrorString(ret) << std::endl;
			else
				std::cerr << "read failed : "
					<< cuFileGetErrorString(errno) << std::endl;
			cuFileHandleDeregister(cf_handle);
			goto exit2;
		}
		total_bytes += nbytes;
		bufOff += nbytes;
		fileOff += nbytes;
		count++;
	} while (total_bytes < size);

	std::cout << "Total chunks read :" <<  count << std::endl;

	cuFileHandleDeregister(cf_handle);
	close (fd);
	fd = -1;

	ret = open(TESTWRITEFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (ret < 0) {
		std::cerr << "file open error: " << TESTWRITEFILE << " "
			<< cuFileGetErrorString(errno) << std::endl;
		goto exit2;
	}
	fd = ret;

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
                std::cerr << "file register error:"
			<< cuFileGetErrorString(status) << std::endl;
		goto exit2;
        }

	std::cout << "writing device memory to file :" << TESTWRITEFILE << std::endl;

	ret = cuFileWrite(cf_handle, (void *)devPtr, size, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "write failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "write failed : "
				<< cuFileGetErrorString(errno) << std::endl;
		cuFileHandleDeregister(cf_handle);
		goto exit2;
	}

	std::cout << "wrote bytes :" << ret << std::endl;

	ret = SHASUM256(TESTFILE, iDigest, size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto exit2;
        }

	DumpSHASUM(iDigest);

	ret = SHASUM256(TESTWRITEFILE, oDigest, size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto exit2;
        }

	DumpSHASUM(oDigest);

	if (memcmp(iDigest, oDigest, SHA256_DIGEST_LENGTH) != 0) {
		ret = -1;
		std::cerr << "SHA SUM Mismatch" << std::endl;
	} else {
		ret = 0;
		std::cout << "SHA SUM Match" << std::endl;
	}

	cuFileHandleDeregister(cf_handle);

exit2:
	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
		std::cerr << "buf deregister failed :"
			<< cuFileGetErrorString(status) << std::endl;
	}
exit1:
	check_cudaruntimecall(cudaFree(devPtr));

exit0:
	if (fd > 0) {
		close(fd);
		fd = -1;
	}
	return ret;
}
