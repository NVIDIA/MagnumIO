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

/* Integration of Thrust data structure and algorithm (find) with GDS cuFile APIs.
 *
 * This sample replaces the device allocation of Thrust vector with
 * cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api
 * allows the user to specify the physical properties of their memory while
 * retaining the contiguous nature of their access which can be accessed
 * using thrust device vector and device pointers, thus not requiring a change
 * in their program structure. Thrust find algorithm is implemented 
 * concurrently on the multiple cuMemMap-ed allocations.
 *
 */

// Includes
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cufile.h>
#include "cufile_sample_utils.h"

// includes, CUDA
#include <builtin_types.h>
#include <cuda.h>
#include <vector>

//includes, thrust integration
#include "cufile_sample_thrust.h"

using namespace std;

// Variables
CUdevice cuDevice;
CUcontext cuContext;

// Functions
int CleanupNoFailure();

static const char *_cudaGetErrorEnum(cudaError_t error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  ret = cudaGetErrorName(error);
  return ret ? ret : unknown;
}

// Collect all of the devices whose memory can be mapped from cuDevice.
vector<CUdevice> getBackingDevices(CUdevice cuDevice)
{
    int num_devices;
    checkCudaErrors(cuDeviceGetCount(&num_devices));

    vector<CUdevice> backingDevices;
    backingDevices.push_back(cuDevice);
    for (int dev = 0; dev < num_devices; dev++) {
        int capable = 0;
        int attributeVal = 0;

        // The mapping device is already in the backingDevices vector
        if (dev == cuDevice) {
            continue;
        }

        // Only peer capable devices can map each others memory
        checkCudaErrors(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
        if (!capable) {
            continue;
        }

        // The device needs to support virtual address management for the required apis to work
        checkCudaErrors(cuDeviceGetAttribute(&attributeVal,
                              CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                              cuDevice));
        if (attributeVal == 0) {
            continue;
        }

        backingDevices.push_back(dev);
    }
    return backingDevices;
}

// Host code
int main(int argc, char **argv)
{
    CUdevice cuDevice;
    const char *TESTFILE;

    printf("Using thrust::find()\n");
    size_t N = 28835840 * 2;

    size_t size = 0;
    int attributeVal = 0;

    //  Get number of devices
    int NUM_DEVICES;
    cudaGetDeviceCount(&NUM_DEVICES);

    // Initialize
    checkCudaErrors(cuInit(0));
    cuDeviceGet(&cuDevice, 0);

    vector<CUdevice> mappingDevices;
    vector<vector<CUdevice>> backingDevices;

    // Check that the selected device supports virtual address management
    for (int i=0; i<NUM_DEVICES; i++) {
        cuDeviceGet(&cuDevice, i);

        // Check that the selected device supports virtual address management
        checkCudaErrors(cuDeviceGetAttribute(&attributeVal,
                            CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                            cuDevice));
        printf("Device %d VIRTUAL ADDRESS MANAGEMENT SUPPORTED = %d.\n", cuDevice, attributeVal);
        if (attributeVal == 0) {
            printf("Device %d doesn't support VIRTUAL ADDRESS MANAGEMENT.\n", cuDevice);
            exit(2);
        }
        // Collect devices accessible by the mapping device (cuDevice) into the backingDevices vector.
        mappingDevices.push_back(cuDevice);
    }
    backingDevices.push_back(getBackingDevices(cuDevice));

    N = N * backingDevices[0].size();
    size = N * sizeof(int);

    printf("total number of elements in each vector :%zu \n", N);
    printf("size of sysmem vector in bytes :%ld \n", size);

    // Create context
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

    // Allocate input host vectors h_vec in host memory using thrust
    thrust::host_vector<int> h_vec(N);
    printf("h_vec(%lx) size: %ld\n", (uint64_t)thrust::raw_pointer_cast(&h_vec[0]), size);

    // Initialize input host vector with sequence 1, 2, ... N
    thrust::generate(h_vec.begin(), h_vec.end(), rand);   
 
    // Allocate vector in device memory
    cufile_thrust_vector<int> dA;

    // Get pointer to device vector
    thrust::device_ptr<int> dA_ptr = dA.cufile_thrust_device_pointer(N);

    printf("rounded size of each vector in bytes : dA(%lx):%ld \n",
                (uint64_t) dA.get_raw_pointer(), dA.allocationSize);

    // Copy input vector from host memory to device memory
    dA = h_vec;

    if(argc < 2) {
            std::cerr << argv[0] << " <filepath> "<< std::endl;
            exit(EXIT_FAILURE);
    }

    TESTFILE = argv[1];

    // Write device vector to file
    dA.write_to_file(TESTFILE);

    // Clearing device vector to 0
    std::cout << "clearing device memory" << std::endl;
    dA = 0;
    cuStreamSynchronize(0);
    
    // Read device memory content from file
    std::cout << "reading to device memory dA from file:" << TESTFILE << std::endl;
    dA.read_from_file(TESTFILE);

    // Get random element to find
    size_t index = rand() % N;
    printf("Finding h_vec[%zu] = %d\n", index, h_vec[index]);

    // Call concurrent implementation of thrust::find
    // Returns index if value is found, else returns -1
    long long int index_found = thrust_concurrent_find<int> (dA_ptr, dA_ptr+N, h_vec[index]);

    if (index_found == -1) {
        printf("Value not found\n");
    }
    else {
        printf("Value found at index %lld\n", index_found); 
    }

    cuStreamSynchronize(0);

    CleanupNoFailure();
    printf("%s for elements=%zu \n", (index_found==index) ? "Result = PASS" : "Result = FAIL", N);

    exit((index_found==index) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure() {
    // Destroy context
    checkCudaErrors(cuCtxDestroy(cuContext));
    return EXIT_SUCCESS;
}
