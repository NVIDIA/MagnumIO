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
 * This sample demonstrates how to integrate Thrust's data structures 
 * and algorithms with GDS cuFile APIs. The program allocates device 
 * memory using the cuMemMap API, writes data from device memory to a
 * file using cuFile, zeros out the device memory, and then reads the 
 * data back from the file to device memory. It leverages the thrust::find
 * algorithm to concurrently search for a value in the device memory. 
 * The program outputs the index of the found value or a message indicating
 * that the value was not found.
 * 
 * ./memmap_thrust <testfile>
 * 
 * | Output |
 * Using thrust::find()
 * Device 1 VIRTUAL ADDRESS MANAGEMENT SUPPORTED = true.
 * Device 0 VIRTUAL ADDRESS MANAGEMENT SUPPORTED = true.
 * Total number of elements in each vector: 50331648
 * Size of sysmem vector in bytes: 201326592
 * h_vec(fff2c6a00010) size: 201326592 bytes
 * ...
 * Current value at found index: 517674477
 * Searching for value = 517674477 in device memory using thrust
 * Value found at index: 40868568
 * Current value at found index: 517674477
 * Result = PASS for elements=50331648
 * Cleaning up
 * Freeing device memory
 * unmap address range: 440000000, size: 201326592
 * freeing address range: 440000000, size: 201326592
 */

#include <iostream>
#include <cstring>
#include <random>
#include <vector>
#include <thrust/sequence.h> 
#include <algorithm>

// Include CUDA headers
#include <builtin_types.h>
#include <cuda.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_memmap.hpp"

// Thrust integration Includes
#include "../include/cufile_sample_thrust.hpp"

static constexpr size_t MAX_TEST_LIMIT = MiB(192);
static constexpr size_t NUM_ELEMENTS = 28835840;

int main(int argc, char *argv[]) {
	std::vector<std::vector<CUdevice>> backingDevices;
	std::mt19937 gen(std::random_device{}());
	std::vector<CUdevice> mappingDevices;
	size_t N = NUM_ELEMENTS * 2;
	int visible_devices = 0;
	int attributeVal = 0;
	int num_devices = 0;
	std::string testfile;
	size_t size = 0;

	// CUDA specific variables
	CUdevice cuDevice;
	CUcontext cuContext;

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <testfile>" << std::endl;
		return EXIT_FAILURE;
	}

	testfile = argv[1];
	std::cout << "Using thrust::find()" << std::endl;

	// Get number of devices
	cudaGetDeviceCount(&num_devices);

	// Initialize
	check_cudadrivercall(cuInit(0));
	check_cudadrivercall(cuDeviceGet(&cuDevice, 0));

	// Check that the selected device supports virtual address management
	for (int i = num_devices - 1; i >= 0; i--) {
		check_cudadrivercall(cuDeviceGet(&cuDevice, i));
		check_cudadrivercall(cuDeviceGetAttribute(
			&attributeVal,
			CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
			cuDevice
		));
		std::cout << "Device " << cuDevice << " VIRTUAL ADDRESS MANAGEMENT SUPPORTED = " << (attributeVal ? "true" : "false") << "." << std::endl;
		if (attributeVal == 0) {
			std::cerr << "Device " << cuDevice << " doesn't support VIRTUAL ADDRESS MANAGEMENT." << std::endl;
			return EXIT_FAILURE;
		}

		// Collect devices accessible by the mapping device (cuDevice) into the backingDevices vector.
		mappingDevices.push_back(cuDevice);
	}
	
	backingDevices.push_back(getBackingDevices(cuDevice));
	visible_devices = backingDevices[0].size();

	// Calculate the total number of elements in the vector
	N = N * visible_devices;
	if (N > (MAX_TEST_LIMIT / sizeof (int))) {
		N = MAX_TEST_LIMIT / sizeof(int);
	}
	size = N * sizeof(int);
	std::cout << "Total number of elements in each vector: " << N << std::endl;
	std::cout << "Size of sysmem vector in bytes: " << size << std::endl;

	// Create context
#if CUDA_VERSION >= 13000
	check_cudadrivercall(cuCtxCreate(&cuContext, NULL, 0, cuDevice));
#else
	check_cudadrivercall(cuCtxCreate(&cuContext, 0, cuDevice));
#endif

	// Allocate input host vectors h_vec in host memory using thrust
	thrust::host_vector<int> h_vec(N);
	std::cout << "h_vec(" << std::hex << reinterpret_cast<uint64_t>(thrust::raw_pointer_cast(&h_vec[0])) 
		<< ") size: " << std::dec << size << " bytes" << std::endl;

	// Initialize input host vector with sequence 1, 2, ... N
	thrust::sequence(h_vec.begin(), h_vec.end(), 1);
	
	// Shuffle the vector
	std::shuffle(h_vec.begin(), h_vec.end(), gen);
 
	// Allocate vector in device memory
	CuFileThrustVector<int> dA(visible_devices);

	// Get pointer to device vector
	thrust::device_ptr<int> dA_ptr = dA.cufile_thrust_device_pointer(N);
	std::cout << "Rounded size of each vector in bytes: dA(" << std::hex << reinterpret_cast<uint64_t>(dA.get_raw_pointer())
		<< "):" << std::dec << dA.allocationSize << " bytes" << std::endl;

	// Copy input vector from host memory to device memory
	dA = h_vec;

	// Write device vector to file
	dA.write_to_file(testfile);

	// Clearing device vector to 0
	std::cout << "Clearing device memory" << std::endl;
	dA = 0;
	check_cudadrivercall(cuStreamSynchronize(0));
	
	// Read device memory content from file
	std::cout << "Reading to device memory dA from file:" << testfile << std::endl;
	dA.read_from_file(testfile);

	// Get random element to find
	std::uniform_int_distribution<size_t> dist(0, N - 1);
	size_t index = dist(gen);
	std::cout << "Finding h_vec[" << std::dec << index << "] = " << h_vec[index] << std::endl;

	// Call concurrent implementation of thrust::find
	// Returns index if value is found, else returns -1
	std::cout << "Searching for value = " << std::dec << h_vec[index] << " in device memory using thrust" << std::endl;
	long long int index_found = thrust_concurrent_find(
		dA_ptr, dA_ptr + N, h_vec[index], visible_devices
	);

	if (index_found == -1) {
		std::cerr << "Value not found" << std::endl;
	} else {
		std::cout << "Value found at index: " << std::dec << index_found << std::endl; 
		std::cout << "Current value at found index: " << std::dec << dA_ptr[index_found] << std::endl;
	}

	std::cout << ((index_found == index) ? "Result = PASS" : "Result = FAIL") << " for elements=" << N << std::endl;
	std::cout << "Cleaning up" << std::endl;
	check_cudadrivercall(cuStreamSynchronize(0));
	check_cudadrivercall(cuCtxDestroy(cuContext));

	return (index_found == index) ? EXIT_SUCCESS : EXIT_FAILURE;
}
