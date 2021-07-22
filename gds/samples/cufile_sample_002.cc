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
 * Sample cuFileWrite Test WITHOUT explicit device memory registration.
 * Note DriverOpen/Close is not needed and implicitly done.
 *
 */
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

// include this header
#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace std;

#define MAX_BUFFER_SIZE (1 * 1024 * 1024UL) // 1 MB

int main(int argc, char *argv[]) {
	int fd = -1;
	int idx = -1;
	ssize_t ret = -1;
	void *devPtr = NULL;
	const size_t size = MAX_BUFFER_SIZE;
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle;
	const char *TESTFILE;

	if(argc < 3) {
                std::cerr << argv[0] << " <filepath> <gpuid> "<< std::endl;
                exit(1);
        }

        TESTFILE = argv[1];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[2])));

	std::cout << "opening file " << TESTFILE << std::endl;

        // opens a file to write
        ret = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
        if (ret < 0) {
                std::cerr << "file open error:"
				<< cuFileGetErrorString(errno) << std::endl;
                return -1;
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
                return -1;
        }

	check_cudaruntimecall(cudaGetDevice(&idx));

	std::cout << "allocating device memory of size :" 
			<< size << " gpu id: " << idx << std::endl;

	// allocates device memory
	check_cudaruntimecall(cudaMalloc(&devPtr, size));

	// filler for device memory
	check_cudaruntimecall(cudaMemset(devPtr, 0xab, size));

	check_cudaruntimecall(cudaGetDevice(&idx));

	std::cout << "writing from gpuid: " << idx << std::endl;

	// writes device memory contents to a file
	// Not we skipped device memory registration using cuFileBufRegister
	ret = cuFileWrite(cf_handle, devPtr, size, 0, 0);
	if (ret < 0)
		if (IS_CUFILE_ERR(ret))
			std::cerr << "write failed : "
				<< cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "write failed : "
				<< cuFileGetErrorString(errno) << std::endl;
	else {
		std::cout << "written bytes :" << ret << std::endl;
		ret = 0;
	}

	check_cudaruntimecall(cudaFree(devPtr));

	// close file
        cuFileHandleDeregister(cf_handle);
        close(fd);
	return ret;
}
