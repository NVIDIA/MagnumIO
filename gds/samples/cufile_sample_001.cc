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
 * Sample cuFileWrite Test.
 *
 * This writes data from GPU memory to a file.
 * For verification, input data has a pattern.
 * User can verify the output file-data after write using
 * hexdump -C <filepath>
 * 00000000  ab ab ab ab ab ab ab ab  ab ab ab ab ab ab ab ab  |................|
 */
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

//include this header file
#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace std;

#define MAX_BUFFER_SIZE (128 * 1024UL) // 128 KB

int main(int argc, char *argv[]) {
	int fd = -1;
	ssize_t ret = -1;
	void *devPtr = NULL;
	const size_t size = MAX_BUFFER_SIZE;
	CUfileError_t status;
	const char *TESTFILE;
	CUfileDescr_t cf_descr;
        CUfileHandle_t cf_handle;

	if(argc < 3) {
                std::cerr << argv[0] << " <filepath> <gpuid> "<< std::endl;
                exit(1);
        }

        TESTFILE = argv[1];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[2])));

        status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "cufile driver open error: "
			<< cuFileGetErrorString(status) << std::endl;
                return -1;
        }

	std::cout << "opening file " << TESTFILE << std::endl;

	// opens a file to write
	ret = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (ret < 0) {
		std::cerr << "file open error:"
			<< cuFileGetErrorString(errno) << std::endl;
		goto out1;
	}
	fd = ret;

	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "file register error:"
			<< cuFileGetErrorString(status) << std::endl;
		close(fd);
		fd = -1;
		goto out1;
	}

	check_cudaruntimecall(cudaMalloc(&devPtr, size));
	// filler
	check_cudaruntimecall(cudaMemset((void*)(devPtr), 0xab, size));

	std::cout << "registering device memory of size :" << size << std::endl;
	// registers device memory
	status = cuFileBufRegister(devPtr, size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
		std::cerr << "buffer register failed:"
			<< cuFileGetErrorString(status) << std::endl;
		goto out2;
	}

	std::cout << "writing from device memory" << std::endl;

	// writes device memory contents to a file
	ret = cuFileWrite(cf_handle, devPtr, size, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "write failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "write failed : "
				<< cuFileGetErrorString(errno) << std::endl;
	} else {
		std::cout << "written bytes :" << ret << std::endl;
		ret = 0;
	}

	std::cout << "deregistering device memory" << std::endl;

	// deregister the device memory
	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
		std::cerr << "buffer deregister failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}

out2:
	check_cudaruntimecall(cudaFree(devPtr));
out1:
	// close file
	if (fd > 0) {
		cuFileHandleDeregister(cf_handle);
		close(fd);
	}

	status = cuFileDriverClose();
	if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
		std::cerr << "cufile driver close failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}
	return ret;
}
