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
 * Sample cuFileWrite Test involving device buffer offsets
 *
 * The test program reads from a file using cuFileBufRegister
 * and writes the contents from an offset inside device memory at some offset.
 *
 * For validation, we match the SHASUM signature of device memory
 * contents from the respective buffer offset and the newly created file contents
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

#define MAX_BUFFER_SIZE (1024 * 1024UL)

#define TESTBUFOFFSET (128 * 1024UL)

using namespace std;

int main(int argc, char *argv[]) {
	int fd;
	ssize_t ret = -1;
	void *devPtr = NULL;
	const size_t size = MAX_BUFFER_SIZE;
	CUfileError_t status;
        CUfileDescr_t cf_descr;
        CUfileHandle_t cf_handle;
	const loff_t fileOff = 0;
	unsigned char iDigest[SHA256_DIGEST_LENGTH], oDigest[SHA256_DIGEST_LENGTH];

        const char *TESTRANDOMFILE, *TESTFILE;

        if(argc < 4) {
                std::cerr << argv[0] << " <readfilepath> <writefilepath> <gpuid>"<< std::endl;
                exit(1);
        }

        TESTRANDOMFILE = argv[1];
        TESTFILE = argv[2];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[3])));

	assert(MAX_BUFFER_SIZE > TESTBUFOFFSET);

	fd = open(TESTRANDOMFILE, O_RDONLY | O_DIRECT, 0);
	if (fd < 0) {
		std::cerr << "file open error : " << TESTRANDOMFILE << " "
			<< cuFileGetErrorString(errno) << std::endl;
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

	status = cuFileBufRegister(devPtr, size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "buf register failed: "
			<< cuFileGetErrorString(status) << std::endl;
		cuFileHandleDeregister(cf_handle);
		close(fd);
		check_cudaruntimecall(cudaFree(devPtr));
		return -1;
	}

	std::cout << "reading file to device memory : "
		<< TESTRANDOMFILE << std::endl;
	// read a file with random data to device memory
	ret = cuFileRead(cf_handle, devPtr, size, fileOff, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "read failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "read failed : "
				<< cuFileGetErrorString(errno) << std::endl;
		goto out;
	}

	std::cout << "read bytes : " << size << std::endl;
	cuFileHandleDeregister(cf_handle);
	close(fd);

	// write random data from device memory to file
	fd = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (fd < 0) {
		std::cerr << "error opening test file " << TESTFILE
			<< " : " << std::strerror(errno) << std::endl;
		goto out;
	}

	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		goto out;
	}

	std::cout << "writing from device memory, buffer OFFSET :"
		<< TESTBUFOFFSET << " to file :" << TESTFILE << std::endl;
	ret = cuFileWrite(cf_handle, (char *)devPtr,
		size - TESTBUFOFFSET, fileOff, TESTBUFOFFSET);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "write failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "write failed : "
				<< cuFileGetErrorString(errno) << std::endl;
		goto out;
	}

	// device memory shasum with bufOffset
	ret = SHASUM256_DEVICEMEM(reinterpret_cast<char *>(devPtr), size, iDigest, TESTBUFOFFSET);
	if(ret < 0) {
                std::cerr << "SHASUM Device mem compute error" << std::endl;
                goto out;
        }
	DumpSHASUM(iDigest);

	// file shasum
	ret = SHASUM256(TESTFILE, oDigest, size - TESTBUFOFFSET);
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
	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "buf deregister failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = -1;
	}
	check_cudaruntimecall(cudaFree(devPtr));

	if (fd > 0) {
		cuFileHandleDeregister(cf_handle);
		close(fd);
	}
	return ret;
}
