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
 * Sample cuFileGetErrorString() usage to obtain obtain readable cuFileAPI errors (c++)
 */

#include <iostream>

//include this header file
#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace  std;

int main(void) {
	int posix_ret; // hold posix style returns
	CUfileError_t status; // hold regular CUFileAPI based returns

	/* Need to load the symbols first. */
	check_cudadrivercall(cuInit(0));

	status.err = CU_FILE_SUCCESS;
	std::cout << "PASS: cufile success status:" << cuFileGetErrorString(status) << std::endl;
	assert(!cuFileGetErrorString(status).compare("Success"));

	status.err = CU_FILE_PLATFORM_NOT_SUPPORTED;
	assert(!cuFileGetErrorString(status).compare("GPUDirect Storage not supported on current platform"));
	std::cerr << "PASS: cufile error status:" << cuFileGetErrorString(status) << std::endl;

	status.err = CU_FILE_INVALID_VALUE;
	assert(!cuFileGetErrorString(status).compare("invalid arguments"));
	std::cerr << "PASS: cufile error status:" << cuFileGetErrorString(status) << std::endl;

	// cuda-driver errors
	status.err = CU_FILE_CUDA_DRIVER_ERROR;
	status.cu_err = CUDA_ERROR_INVALID_VALUE;
	assert(!cuFileGetErrorString(status).compare("CUDA Driver API error.CUDA_ERROR_INVALID_VALUE"));
	std::cerr << "PASS: cufile error status:" << cuFileGetErrorString(status) << std::endl;

	// read/write APIs return posix style errors
	posix_ret = 0;
	assert(!cuFileGetErrorString(posix_ret).compare("Success"));
	std::cerr << "PASS: cufile error status:" << cuFileGetErrorString(posix_ret) << std::endl;

	posix_ret = -22;
	assert(!cuFileGetErrorString(posix_ret).compare("Invalid argument"));
	std::cerr << "PASS: cufile error status:" << cuFileGetErrorString(posix_ret) << std::endl;

	posix_ret = -CU_FILE_INVALID_FILE_TYPE; // CUFile base errors
	assert(!cuFileGetErrorString(posix_ret).compare("unsupported file type"));
	std::cerr << "PASS: cufile error status:" << cuFileGetErrorString(posix_ret) << std::endl;

	return 0;
}
