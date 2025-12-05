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
 * Sample for cuFileGetErrorString() usage to obtain readable
 * cuFileAPI errors (C++) and cuFileDriver Get/SetProperties
 *
 * ./error_and_properties
 *
 * | Output |
 * cuFileGetErrorString Usage
 * PASS: cuFile success status: Success
 * PASS: cuFile error status: GPUDirect Storage not supported on current platform
 * ...
 * cuFileDriver Get/SetProperties Usage
 * cuFile driver properties before using setters:
 *   Poll mode bitmask: 00000010
 * ...
 */

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <bitset>

// Include CUDA runtime header file
#include <cuda_runtime.h>
// Include cuFile header file
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

// If poll mode is set, this threshold defines lower limit of request size
// before we start polling for request completion
static constexpr size_t POLL_THRESH_KB = 32768;

// Limit application to use a portion of GPU memory for IO usage
static constexpr size_t LIMIT_BAR_USAGE_KB = 512;

// For internal use, max mmaped buffer size
static constexpr size_t LIMIT_DIO_SIZE_KB = 64;

// For internal use, allocation of internal buffers
static constexpr size_t LIMIT_CACHE_SIZE_KB = 64;

int main(void) {
	CUfileError_t status = {CU_FILE_SUCCESS, CUDA_SUCCESS};
	CUfileDrvProps_t props = {};
	int posix_ret = -1; // hold posix style returns

	/* cuFile API errors showcase */
	std::cout << "cuFileGetErrorString Usage" << std::endl;

	// Need to load the symbols first.
	check_cudadrivercall(cuInit(0));

	status.err = CU_FILE_SUCCESS;
	std::cout << "PASS: cuFile success status: " << cuFileGetErrorString(status) << std::endl;
	assert(!cuFileGetErrorString(status).compare("Success"));

	status.err = CU_FILE_PLATFORM_NOT_SUPPORTED;
	assert(!cuFileGetErrorString(status).compare("GPUDirect Storage not supported on current platform"));
	std::cerr << "PASS: cuFile error status: " << cuFileGetErrorString(status) << std::endl;

	status.err = CU_FILE_INVALID_VALUE;
	assert(!cuFileGetErrorString(status).compare("invalid arguments"));
	std::cerr << "PASS: cuFile error status: " << cuFileGetErrorString(status) << std::endl;

	// cuda-driver errors
	status.err = CU_FILE_CUDA_DRIVER_ERROR;
	status.cu_err = CUDA_ERROR_INVALID_VALUE;
	assert(!cuFileGetErrorString(status).compare("CUDA Driver API error.CUDA_ERROR_INVALID_VALUE"));
	std::cerr << "PASS: cuFile error status: " << cuFileGetErrorString(status) << std::endl;

	// read/write APIs return posix style errors
	posix_ret = 0;
	assert(!cuFileGetErrorString(posix_ret).compare("Success"));
	std::cerr << "PASS: cuFile error status: " << cuFileGetErrorString(posix_ret) << std::endl;

	posix_ret = -22;
	assert(!cuFileGetErrorString(posix_ret).compare("Invalid argument"));
	std::cerr << "PASS: cuFile error status: " << cuFileGetErrorString(posix_ret) << std::endl;

	posix_ret = -CU_FILE_INVALID_FILE_TYPE; // CUFile base errors
	assert(!cuFileGetErrorString(posix_ret).compare("unsupported file type"));
	std::cerr << "PASS: cuFile error status: " << cuFileGetErrorString(posix_ret) << std::endl;


	/* cuFileDriver Get/SetProperties showcase */	
	// Reset the status for cuFileDriver Get/SetProperties
	std::cout << std::endl << "cuFileDriver Get/SetProperties Usage" << std::endl;

	status = cuFileDriverOpen();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver open error "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	status = cuFileDriverGetProperties(&props);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver get properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "cuFile driver properties before using setters: " << std::endl;
	std::cout << "  Poll mode bitmask: " << std::bitset<8>(props.nvfs.dcontrolflags) << std::endl;
	std::cout << "  Poll threshold size: " << props.nvfs.poll_thresh_size << " bytes" << std::endl;
	std::cout << "  Max pinned mem size: " << props.max_device_pinned_mem_size << " bytes" << std::endl;
	std::cout << "  Max direct io size: " << props.nvfs.max_direct_io_size << " bytes" << std::endl;
	std::cout << "  Max cache size: " << props.max_device_cache_size << " bytes" << std::endl;

	status = cuFileDriverSetPollMode(true, POLL_THRESH_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	status = cuFileDriverSetMaxPinnedMemSize(LIMIT_BAR_USAGE_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	status = cuFileDriverSetMaxDirectIOSize(LIMIT_DIO_SIZE_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	status = cuFileDriverSetMaxCacheSize(LIMIT_CACHE_SIZE_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	status = cuFileDriverGetProperties(&props);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFile driver get properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "cuFile driver properties after using setters: " << std::endl;
	std::cout << "  Poll mode bitmask: " << std::bitset<8>(props.nvfs.dcontrolflags) << std::endl;
	std::cout << "  Poll threshold size: " << props.nvfs.poll_thresh_size << " bytes" << std::endl;
	std::cout << "  Max pinned mem size: " << props.max_device_pinned_mem_size << " bytes" << std::endl;
	std::cout << "  Max direct io size: " << props.nvfs.max_direct_io_size << " bytes" << std::endl;
	std::cout << "  Max cache size: " << props.max_device_cache_size << " bytes" << std::endl;

	status = cuFileDriverClose();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFile driver close failed "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
