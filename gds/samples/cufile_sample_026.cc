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

/* Vector addition: C = A + B.
 *
 * This sample replaces the device allocation in the vectorAddDrvsample with
 * cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api
 * allows the user to specify the physical properties of their memory while
 * retaining the contiguos nature of their access, thus not requiring a change
 * in their program structure.
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

// includes, project
//#include "helper_cuda_drvapi.h"
//#include "helper_functions.h"

// includes, CUDA
#include <builtin_types.h>
#include <cuda.h>
#include <vector>

// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}


template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static size_t round_up(size_t x, size_t y)
{
    return ((x + y - 1) / y) * y;
}

////////////////////////////////////////////////////////////////////////////
//! Allocate virtually contiguous memory backed on separate devices
//! @return CUresult error code on failure.
//! @param[out] dptr            Virtual address reserved for allocation
//! @param[out] allocationSize  Actual amount of virtual address space reserved.
//!                             AllocationSize is needed in the free operation.
//! @param[in] size             The minimum size to allocate (will be rounded up to accomodate
//!                             required granularity).
//! @param[in] residentDevices  Specifies what devices the allocation should be striped across.
//! @param[in] mappingDevices   Specifies what devices need to read/write to the allocation.
//! @align                      Additional allignment requirement if desired.
//! @note       The VA mappings will look like the following:
//!
//!     v-stripeSize-v                v-rounding -v
//!     +-----------------------------------------+
//!     |      D1     |      D2     |      D3     |
//!     +-----------------------------------------+
//!     ^-- dptr                      ^-- dptr + size
//!
//! Each device in the residentDevices list will get an equal sized stripe.
//! Excess memory allocated will be  that meets the minimum
//! granularity requirements of all the devices.
//!
//! @note uses cuMemGetAllocationGranularity cuMemCreate cuMemMap and cuMemSetAccess
//!   function calls to organize the va space
//!
//! @note uses cuMemRelease to release the allocationHandle.  The allocation handle
//!   is not needed after its mappings are set up.
////////////////////////////////////////////////////////////////////////////
CUresult
simpleMallocMultiDeviceMmap(CUdeviceptr *dptr, size_t va_size, size_t *allocationSize, size_t size,
         const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
         size_t align = 0);


CUresult
simpleMallocMultiDeviceMmapResize(CUdeviceptr dptr, size_t va_size, size_t *allocationSize, size_t size,
         const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
         size_t align = 0);

////////////////////////////////////////////////////////////////////////////
//! Frees resources allocated by simpleMallocMultiDeviceMmap
//! @CUresult CUresult error code on failure.
//! @param[in] dptr  Virtual address reserved by simpleMallocMultiDeviceMmap
//! @param[in] size  allocationSize returned by simpleMallocMultiDeviceMmap
////////////////////////////////////////////////////////////////////////////
CUresult
simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size, size_t va_size);

CUresult
simpleMallocMultiDeviceMmapResize(CUdeviceptr dptr, size_t va_size, size_t *allocationSize, size_t size,
         const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
         size_t align)
{

    CUresult status = CUDA_SUCCESS;
    size_t min_granularity = 0;
    size_t oldStripeSize, stripeSize;
    size_t add_size;

    // Setup the properties common for all the chunks
    // The allocations will be device pinned memory.
    // This property structure describes the physical location where the memory will be allocated via cuMemCreate allong with additional properties
    // In this case, the allocation will be pinnded device memory local to a given device.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    // Get the minimum granularity needed for the resident devices
    // (the max of the minimum granularity of each participating device)
    for (unsigned idx = 0; idx < residentDevices.size(); idx++) {
        size_t granularity = 0;

        //get the minnimum granularity for residentDevices[idx]
        prop.location.id = residentDevices[idx];
        status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            goto done;
        }
        if (min_granularity < granularity) {
            min_granularity = granularity;
        }
    }

    // Get the minimum granularity needed for the accessing devices
    // (the max of the minimum granularity of each participating device)
    for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
        size_t granularity = 0;

        //get the minnimum granularity for mappingDevices[idx]
        prop.location.id = mappingDevices[idx];
        status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            goto done;
        }
        if (min_granularity < granularity) {
            min_granularity = granularity;
        }
    }

    // Round up the size such that we can evenly split it into a stripe size tha meets the granularity requirements
    // Essentially size = N * residentDevices.size() * min_granularity is the requirement,
    // since each piece of the allocation will be stripeSize = N * min_granularity
    // and the min_granularity requirement applies to each stripeSize piece of the allocation.
    size = round_up(size, residentDevices.size() * min_granularity);
    add_size = size - *allocationSize; 
    oldStripeSize = *allocationSize / residentDevices.size(); 
    stripeSize = add_size / residentDevices.size();

    printf("granularity: %ld, oldsize: %ld  add_size: %ld new_size: %ld stripeSize: %ld oldStripeSize: %ld\n",
            min_granularity, *allocationSize, add_size, size, stripeSize, oldStripeSize);


    // Create and map the backings on each gpu
    // note: reusing CUmemAllocationProp prop from earlier with prop.type & prop.location.type already specified.
    for (size_t idx = 0; idx < residentDevices.size(); idx++) {
        CUresult status2 = CUDA_SUCCESS;

        // Set the location for this chunk to this device
        prop.location.id = residentDevices[idx];

        printf("adding physical mem of size %ld to gpu: %ld\n", stripeSize, idx);
        // Create the allocation as a pinned allocation on this device
        CUmemGenericAllocationHandle allocationHandle;
        status = cuMemCreate(&allocationHandle, stripeSize, &prop, 0);
        if (status != CUDA_SUCCESS) {
            goto done;
        }

        // Assign the chunk to the appropriate VA range and release the handle.
        // After mapping the memory, it can be referenced by virtual address.
        // Since we do not need to make any other mappings of this memory or export it,
        // we no longer need and can release the allocationHandle.
        // The allocation will be kept live until it is unmapped.
        printf("adding new mapping to VA space start at %lx new size: %ld gpu: %ld \n", (uint64_t)((char*)dptr + *allocationSize  + (stripeSize * idx)), stripeSize, idx);
        status = cuMemMap((CUdeviceptr)((char*)dptr + *allocationSize  + (stripeSize * idx)),
                          stripeSize, 0, allocationHandle, 0);

        //printf("freeing allocation handle \n");
        // the handle needs to be released even if the mapping failed.
        status2 = cuMemRelease(allocationHandle);
        if (status == CUDA_SUCCESS) {
            // cuMemRelease should not have failed here
            // as the handle was just allocated successfully
            // however return an error if it does.
            status = status2;
        }

        // Cleanup in case of any mapping failures.
        if (status != CUDA_SUCCESS) {
            printf("failed mapping VA \n");
            goto done;
        }

    }

    // Return the rounded up size to the caller for use in the free
    if (allocationSize) {
    	*allocationSize = size;
    }

    {
        // Each accessDescriptor will describe the mapping requirement for a single device
        std::vector<CUmemAccessDesc> accessDescriptors;
        accessDescriptors.resize(mappingDevices.size());

        // Prepare the access descriptor array indicating where and how the backings should be visible.
        for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
            // Specify which device we are adding mappings for.
            accessDescriptors[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDescriptors[idx].location.id = mappingDevices[idx];

            // Specify both read and write access.
            accessDescriptors[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }

        printf("setting access to entire VA space for dptr: %lx size: %ld on %ld GPUs \n", (uint64_t)dptr, size, mappingDevices.size());
        // Apply the access descriptors to the whole VA range.
        status = cuMemSetAccess(dptr, size, &accessDescriptors[0], accessDescriptors.size());
        if (status != CUDA_SUCCESS) {
            goto done;
        }
    }

done:
    if (status != CUDA_SUCCESS) {
        if (dptr) {
            simpleFreeMultiDeviceMmap(dptr, size, va_size);
            *allocationSize = 0;
        }
    }

    return status;

}


CUresult
simpleMallocMultiDeviceMmap(CUdeviceptr *dptr, size_t va_size, size_t *allocationSize, size_t size,
         const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
         size_t align)
{
    CUresult status = CUDA_SUCCESS;
    size_t min_granularity = 0;
    size_t stripeSize;

    // Setup the properties common for all the chunks
    // The allocations will be device pinned memory.
    // This property structure describes the physical location where the memory will be allocated via cuMemCreate allong with additional properties
    // In this case, the allocation will be pinnded device memory local to a given device.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    // Get the minimum granularity needed for the resident devices
    // (the max of the minimum granularity of each participating device)
    for (unsigned idx = 0; idx < residentDevices.size(); idx++) {
        size_t granularity = 0;

        //get the minnimum granularity for residentDevices[idx]
        prop.location.id = residentDevices[idx];
        status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            goto done;
        }
	std::cout <<"resident device idx:" << idx << ",gpuid:" << prop.location.id << ",granularity " << granularity << std::endl;
        if (min_granularity < granularity) {
            min_granularity = granularity;
        }
    }

    // Get the minimum granularity needed for the accessing devices
    // (the max of the minimum granularity of each participating device)
    for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
        size_t granularity = 0;

        //get the minnimum granularity for mappingDevices[idx]
        prop.location.id = mappingDevices[idx];
        status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            goto done;
        }
	std::cout <<"mapping device idx:" << idx << ",gpuid:" << prop.location.id << ",granularity " << granularity << std::endl;
        if (min_granularity < granularity) {
            min_granularity = granularity;
        }
    }

    // Round up the size such that we can evenly split it into a stripe size tha meets the granularity requirements
    // Essentially size = N * residentDevices.size() * min_granularity is the requirement,
    // since each piece of the allocation will be stripeSize = N * min_granularity
    // and the min_granularity requirement applies to each stripeSize piece of the allocation.
    size = round_up(size, residentDevices.size() * min_granularity);
    stripeSize = size / residentDevices.size();
    std::cout << "actual allocation size:" << size  << ",min granularity" << min_granularity << ",total devs:" << residentDevices.size() << std::endl;

    // Return the rounded up size to the caller for use in the free
    if (allocationSize) {
        *allocationSize = size;
    }

    // Reserve the required contiguous VA space for the allocations
    status = cuMemAddressReserve(dptr, va_size, align, 0, 0);
    if (status != CUDA_SUCCESS) {
	std::cout << "Failed to reserve va addr:" << std::hex << (uint64_t)dptr << std::dec << ", size:" << va_size << std::endl;
        goto done;
    }

    // Create and map the backings on each gpu
    // note: reusing CUmemAllocationProp prop from earlier with prop.type & prop.location.type already specified.
    for (size_t idx = 0; idx < residentDevices.size(); idx++) {
        CUresult status2 = CUDA_SUCCESS;

        // Set the location for this chunk to this device
        prop.location.id = residentDevices[idx];

        // Create the allocation as a pinned allocation on this device
        CUmemGenericAllocationHandle allocationHandle;
        status = cuMemCreate(&allocationHandle, stripeSize, &prop, 0);
        if (status != CUDA_SUCCESS) {
            goto done;
        }

        // Assign the chunk to the appropriate VA range and release the handle.
        // After mapping the memory, it can be referenced by virtual address.
        // Since we do not need to make any other mappings of this memory or export it,
        // we no longer need and can release the allocationHandle.
        // The allocation will be kept live until it is unmapped.
        status = cuMemMap(*dptr + (stripeSize * idx), stripeSize, 0, allocationHandle, 0);

        // the handle needs to be released even if the mapping failed.
        status2 = cuMemRelease(allocationHandle);
        if (status == CUDA_SUCCESS) {
            // cuMemRelease should not have failed here
            // as the handle was just allocated successfully
            // however return an error if it does.
            status = status2;
        }

        // Cleanup in case of any mapping failures.
        if (status != CUDA_SUCCESS) {
            goto done;
        }
    }

    {
        // Each accessDescriptor will describe the mapping requirement for a single device
        std::vector<CUmemAccessDesc> accessDescriptors;
        accessDescriptors.resize(mappingDevices.size());

        // Prepare the access descriptor array indicating where and how the backings should be visible.
        for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
            // Specify which device we are adding mappings for.
            accessDescriptors[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDescriptors[idx].location.id = mappingDevices[idx];

            // Specify both read and write access.
            accessDescriptors[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }

        // Apply the access descriptors to the whole VA range.
        status = cuMemSetAccess(*dptr, size, &accessDescriptors[0], accessDescriptors.size());
        if (status != CUDA_SUCCESS) {
            goto done;
        }
    }

done:
    if (status != CUDA_SUCCESS) {
        if (*dptr) {
            simpleFreeMultiDeviceMmap(*dptr, size, va_size);
        }
    }

    return status;
}



CUresult
simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size, size_t va_size)
{
    CUresult status = CUDA_SUCCESS;

    // Unmap the mapped virtual memory region
    // Since the handles to the mapped backing stores have already been released
    // by cuMemRelease, and these are the only/last mappings referencing them,
    // The backing stores will be freed.
    // Since the memory has been unmapped after this call, accessing the specified
    // va range will result in a fault (unitll it is remapped).
    std::cout << "unmap address range " << std::hex << (uint64_t)dptr << std::dec << ", size:" << size << std::endl;
    status = cuMemUnmap(dptr, size);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    // Free the virtual address region.  This allows the virtual address region
    // to be reused by future cuMemAddressReserve calls.  This also allows the
    // virtual address region to be used by other allocation made through
    // opperating system calls like malloc & mmap.
    std::cout << "freeing address range " << std::hex << dptr << ", size:" << std::dec << va_size << std::endl;
    status = cuMemAddressFree(dptr, va_size);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    return status;
}

using namespace std;

// Variables
CUdevice cuDevice;
CUcontext cuContext;
float *h_A;
float *h_B;
float *h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_A1;
CUdeviceptr d_B1;
CUdeviceptr d_C;
size_t va_size;
size_t allocationSizeA=0;
size_t allocationSizeB=0;
size_t allocationSizeC=0;

// Functions
int CleanupNoFailure();
void RandomInit(float *, int);

#ifdef __cplusplus
extern "C" {
extern void vectorAdd(const float *A, const float *B, float *C,
                          int numElements);
}
#endif

//collect all of the devices whose memory can be mapped from cuDevice.
vector<CUdevice> getBackingDevices(CUdevice cuDevice)
{
    int num_devices;

    checkCudaErrors(cuDeviceGetCount(&num_devices));

    vector<CUdevice> backingDevices;
    backingDevices.push_back(cuDevice);
    for (int dev = 0; dev < num_devices; dev++)
    {
        int capable = 0;
        int attributeVal = 0;

        // The mapping device is already in the backingDevices vector
        if (dev == cuDevice)
        {
            continue;
        }

        // Only peer capable devices can map each others memory
        checkCudaErrors(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
        if (!capable)
        {
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
    int fdA=-1, fdB=-1, ret;
    CUfileError_t status;
    CUdevice cuDevice;
    const char *TESTFILEA, *TESTFILEB;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle_A , cf_handle_B;

    printf("Vector Addition (Driver API)\n");
    int N = 28835840;

    size_t  size = 0;
    int attributeVal = 0;

    // Initialize
    checkCudaErrors(cuInit(0));

    cuDeviceGet(&cuDevice, 0);

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
    vector<CUdevice> backingDevices = getBackingDevices(cuDevice);
    //The vector addition happens on cuDevice, so the allocations need to be mapped there.
    vector<CUdevice> mappingDevices;
    mappingDevices = backingDevices;
    N = N * backingDevices.size();
    size = N * sizeof(float);

    printf("total number of elements in each vector :%d \n", N);

    // Create context
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));


    printf("size of each sysmem vector in bytes :%ld \n", size);
    // Allocate input vectors h_A and h_B in host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    printf("h_A(%lx) h_B(%lx) h_c(%lx) size: %ld\n", (uint64_t)h_A, (uint64_t)h_B, (uint64_t)h_C, size);

    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    // note that a call to cuCtxEnablePeerAccess is not needed even though
    // the backing devices and mapping device are not the same.
    // This is because the cuMemSetAccess call explicitly specifies
    // the cross device mapping.
    // cuMemSetAccess is still subject to the constraints of cuDeviceCanAccessPeer
    // for cross device mappings (hence why we checked cuDeviceCanAccessPeer earlier).

    va_size = size * 2;
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_C, va_size, &allocationSizeC, size/2, backingDevices, mappingDevices));
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_A, va_size, &allocationSizeA, size/2, backingDevices, mappingDevices));
    checkCudaErrors(simpleMallocMultiDeviceMmap(&d_B, va_size, &allocationSizeB, size/2, backingDevices, mappingDevices));

    printf("rounded size of each vector in bytes : d_A(%lx):%ld d_B(%lx):%ld d_C(%lx):%ld \n",
            (uint64_t) d_A, allocationSizeA, (uint64_t) d_B, allocationSizeB,
            (uint64_t)d_C, allocationSizeC);

    // Copy vectors from host memory to device memory
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, allocationSizeA));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, allocationSizeB));


    if(argc < 3) {
            std::cerr << argv[0] << " <filepathA>  <filepathB> "<< std::endl;
            exit(EXIT_FAILURE);
    }

    TESTFILEA = argv[1];
    TESTFILEB = argv[2];

    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cufile driver open error: "
                    << cuFileGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
    }

    std::cout << "opening file " << TESTFILEA << std::endl;
    std::cout << "opening file " << TESTFILEB << std::endl;

    // opens file A to write
    ret = open(TESTFILEA, O_CREAT | O_RDWR | O_DIRECT, 0664);
    if (ret < 0) {
            std::cerr << "file open error:"
                    << cuFileGetErrorString(errno) << std::endl;
            exit(EXIT_FAILURE);
    }
    fdA = ret;

    // opens file B to write
    ret = open(TESTFILEB, O_CREAT | O_RDWR | O_DIRECT, 0664);
    if (ret < 0) {
            std::cerr << "file open error:"
                    << cuFileGetErrorString(errno) << std::endl;
            exit(EXIT_FAILURE);
    }
    fdB = ret;


    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fdA;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle_A, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "file register error:"
                    << cuFileGetErrorString(status) << std::endl;
            close(fdA);
            exit(EXIT_FAILURE);
    }

    // opens file B to write
    ret = open(TESTFILEB, O_CREAT | O_RDWR | O_DIRECT, 0664);
    if (ret < 0) {
            std::cerr << "file open error:"
                    << cuFileGetErrorString(errno) << std::endl;
            close(fdA);
            exit(EXIT_FAILURE);
    }
    fdB = ret;

    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fdB;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle_B, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "file register error:"
                    << cuFileGetErrorString(status) << std::endl;
            close(fdA);
            close(fdB);
            exit(EXIT_FAILURE);
    }

    std::cout << "registering device memory of size :" << size << "allocated size " << allocationSizeA << std::endl;
    // registers device memory
    status = cuFileBufRegister((void*)d_A, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
            ret = -1;
            std::cerr << "buffer register A failed:"
                    << cuFileGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
    }

    std::cout << "registering device memory of size :" << size << "allocated size " << allocationSizeB << std::endl;
    // registers device memory
    status = cuFileBufRegister((void*)d_B, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
            ret = -1;
            std::cerr << "buffer register B failed:"
                    << cuFileGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
    }

    std::cout << "doubling device memory d_A d_B d_C to size:" << size << std::endl;
    d_A1 = ((CUdeviceptr)((char*)d_A + allocationSizeA));
    d_B1 = ((CUdeviceptr)((char*)d_B + allocationSizeB));


    checkCudaErrors(simpleMallocMultiDeviceMmapResize(d_C, va_size, &allocationSizeC, size, backingDevices, mappingDevices));

    size_t oldSize = allocationSizeA;
    size_t cur_size = allocationSizeA;
    size_t chunk_size = 32 * 1024 * 1024; // based on max 16 GPUs and granularity of 2MB

    // resize in chunks of 32 MB
    while(allocationSizeA < size) {
            if((size - allocationSizeA) <  chunk_size) {
                chunk_size = size - allocationSizeA;
            }
            // make the new allocation beyond the registered size
            cur_size = allocationSizeA + chunk_size;
            std::cout << "resize d_A memory to new current size:" <<  cur_size << std::endl;
            checkCudaErrors(simpleMallocMultiDeviceMmapResize(d_A, va_size, &allocationSizeA, cur_size, backingDevices, mappingDevices));
    }

    //copy the rest of the memory from host to newly resized memory
    checkCudaErrors(cuMemcpyHtoD(d_A1, (char*)h_A+oldSize, (size- oldSize)));
    oldSize = allocationSizeB;
    checkCudaErrors(simpleMallocMultiDeviceMmapResize(d_B, va_size, &allocationSizeB, size, backingDevices, mappingDevices));
    checkCudaErrors(cuMemcpyHtoD(d_B1, (char*)h_B+oldSize, (size - oldSize)));

    cuStreamSynchronize(0);
   
    printf("new size of each vector in bytes : d_A(%lx):%ld d_B(%lx):%ld d_C(%lx):%ld \n",
            (uint64_t) d_A, allocationSizeA, (uint64_t) d_B, allocationSizeB,
            (uint64_t)d_C, allocationSizeC);


    std::cout << "Auto register for newer allocation" << std::endl;

    std::cout << "iteratively writing from device memory d_A to file:" << TESTFILEA << std::endl;
    size_t cur_off = 0;
    size_t io_size = 2 * 1024 * 1024;
    size_t bytes_left = size;
    while (bytes_left) {
            if(bytes_left < io_size) {
                io_size = bytes_left;
            }
            ret = cuFileWrite(cf_handle_A, (void*)((char *)d_A), io_size, cur_off, cur_off);
            if (ret < 0) {
                    if (IS_CUFILE_ERR(ret))
                            std::cerr << "write failed : "
                                    << cuFileGetErrorString(ret) << std::endl;
                    else
                            std::cerr << "write failed : "
                                    << cuFileGetErrorString(errno) << std::endl;
                    exit(1);
            } else {
                    bytes_left -= ret;
                    std::cout << "current written bytes at offset:" << cur_off << " to d_A ret:" << ret << ", left:" << bytes_left << std::endl;
                    cur_off += ret;
            }
    }

    std::cout << "writing from device memory d_B to file:" << TESTFILEB << std::endl;
    // writes device memory contents B to a file B for size bytes
    ret = cuFileWrite(cf_handle_B, (void*)d_B, allocationSizeB, 0, 0);
    if (ret < 0) {
            if (IS_CUFILE_ERR(ret))
                    std::cerr << "write failed : "
                            << cuFileGetErrorString(ret) << std::endl;
            else
                    std::cerr << "write failed : "
                            << cuFileGetErrorString(errno) << std::endl;
            exit(1);
    } else {
            std::cout << "written bytes d_B:" << ret << std::endl;
            ret = 0;
    }

    std::cout << "clearing device memory" << std::endl;
    checkCudaErrors(cuMemsetD8(d_A, 0x0, allocationSizeA));
    checkCudaErrors(cuMemsetD8(d_B, 0x0, allocationSizeB));
    checkCudaErrors(cuMemsetD8(d_C, 0x0, allocationSizeC));
    cuStreamSynchronize(0);

    std::cout << "reading to device memory d_A from file:" << TESTFILEA << std::endl;

    // reads device memory contents A from file A for size bytes
    ret = cuFileRead(cf_handle_A, (void*)d_A, size, 0, 0);
    if (ret < 0) {
            if (IS_CUFILE_ERR(ret))
                    std::cerr << "read failed : "
                            << cuFileGetErrorString(ret) << std::endl;
            else
                    std::cerr << "read failed : "
                            << cuFileGetErrorString(errno) << std::endl;
            exit(1);
    } else {
            std::cout << "read bytes to d_A:" << ret << std::endl;
            ret = 0;
    }

    std::cout << "reading to device memory d_B from file:" << TESTFILEB << std::endl;
    // reads device memory contents B from file B for size bytes
    ret = cuFileRead(cf_handle_B, (void*)d_B, allocationSizeB, 0, 0);
    if (ret < 0) {
            if (IS_CUFILE_ERR(ret))
                    std::cerr << "read failed : "
                            << cuFileGetErrorString(ret) << std::endl;
            else
                    std::cerr << "read failed : "
                            << cuFileGetErrorString(errno) << std::endl;
            exit(1);
    } else {
            std::cout << "read bytes to d_B :" << ret << std::endl;
            ret = 0;
    }

    printf("GPU vector ADD for %d elements size:%ld \n", N, N*sizeof(float));
    vectorAdd((const float *)d_A, (const float *)d_B, (float *)d_C, N);
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, N * sizeof(float)));
    cuStreamSynchronize(0);

    // Verify result
    int i;

    for (i = 0; i < N; ++i)
    {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f)
        {
            printf("err element: %d h_C=%f, sum = %f \n", i, h_C[i], sum);
            break;
        }
    }

    CleanupNoFailure();
    printf("%s for elements=%d \n", (i==N) ? "Result = PASS" : "Result = FAIL", i);

    exit((i==N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure()
{
    CUfileError_t status;
    std::cout << "deregistering device memory" << std::endl;

    // deregister the device memory
    status = cuFileBufDeregister((void*)d_A);
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "buffer deregister failed:"
                    << cuFileGetErrorString(status) << std::endl;
    }

    // deregister the device memory
    status = cuFileBufDeregister((void*)d_B);
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "buffer deregister failed:"
                    << cuFileGetErrorString(status) << std::endl;
    }

    std::cout << "freeing d_A" << std::endl;
    // Free device memory
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_A, allocationSizeA, va_size));
    std::cout << "freeing d_B" << std::endl;
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_B, allocationSizeB, va_size));
    std::cout << "freeing d_C" << std::endl;
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_C, allocationSizeC, va_size));

    // Free host memory
    if (h_A)
    {
        free(h_A);
    }

    if (h_B)
    {
        free(h_B);
    }

    if (h_C)
    {
        free(h_C);
    }

    // deregister the device memory
    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cuFileDriverClose failed:"
                    << cuFileGetErrorString(status) << std::endl;
    }


    checkCudaErrors(cuCtxDestroy(cuContext));

    return EXIT_SUCCESS;
}
// Allocates an array with random float entries.
void RandomInit(float *data, int n)
{
    for (int i = 0; i < n; ++i)
    {
        data[i] = rand() / (float)RAND_MAX;
    }
}
