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
 * Sample for cuFileDriver Get/SetProperties
 *
 */

#include <cstdlib>
#include <iostream>

//#include <cuda_runtime.h>

// include this header
#include "cufile.h"

#include "cufile_sample_utils.h"

// if poll mode is set, this threshold defines lower limit of request size
// before we start polling for request completion
#define POLL_THRESH_KB        32768

// Limit application to use a portion of GPU memory for IO usage
#define LIMIT_BAR_USAGE_KB    512

// For internal use, max mmaped buffer size
#define LIMIT_DIO_SIZE_KB     64

// For internal use, allocation of internal buffers
#define LIMIT_CACHE_SIZE_KB   64

using namespace std;

int main(void) {
	CUfileError_t status;
	CUfileDrvProps_t props;

	status = cuFileDriverOpen();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver open error "
			<< cuFileGetErrorString(status) << std::endl;
		return -1;
	}

	status = cuFileDriverGetProperties(&props);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver get properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return -1;
	}

	status = cuFileDriverSetPollMode(true, POLL_THRESH_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return -1;
	}

	status = cuFileDriverSetMaxPinnedMemSize(LIMIT_BAR_USAGE_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return -1;
	}

	status = cuFileDriverSetMaxDirectIOSize(LIMIT_DIO_SIZE_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return -1;
	}

	status = cuFileDriverSetMaxCacheSize(LIMIT_CACHE_SIZE_KB);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver set properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return -1;
	}

	status = cuFileDriverGetProperties(&props);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cufile driver get properties failed "
			<< cuFileGetErrorString(status) << std::endl;
		return -1;
	}

	cuFileDriverClose();
	return 0;
}
