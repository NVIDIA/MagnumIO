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

#ifndef __CUFILE_SAMPLE_MEMMAP_HPP_
#define __CUFILE_SAMPLE_MEMMAP_HPP_

// Includes
#include <cstdlib>
#include <vector>

// Include CUDA headers
#include <builtin_types.h>
#include <cuda.h>

// Helper structs for easier cleanup
struct FileResources {
	int& fdA;
	int& fdB;
	CUfileHandle_t& cf_handle_A;
	CUfileHandle_t& cf_handle_B;
};

struct HostBuffers {
	float* h_A;
	float* h_B;
	float* h_C;
};

struct DeviceBuffers {
	CUdeviceptr& d_A;
	CUdeviceptr& d_B;
	CUdeviceptr& d_C;
	size_t& allocationSizeA;
	size_t& allocationSizeB;
	size_t& allocationSizeC;
	size_t& va_size;
};

////////////////////////////////////////////////////////////////////////////
//! Frees resources allocated by simpleMallocMultiDeviceMmap
//! @param[in] dptr  Virtual address reserved by simpleMallocMultiDeviceMmap
//! @param[in] size  allocationSize returned by simpleMallocMultiDeviceMmap
//! @param[in] va_size  Virtual address size returned by simpleMallocMultiDeviceMmap
//! @return CUresult
////////////////////////////////////////////////////////////////////////////
CUresult simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size, size_t va_size);

////////////////////////////////////////////////////////////////////////////
//! Cleans up resources allocated by simpleMallocMultiDeviceMmapResize
//! @param[in] dptr  Virtual address reserved by simpleMallocMultiDeviceMmapResize
//! @param[in] size  allocationSize returned by simpleMallocMultiDeviceMmapResize
//! @param[in] va_size  Virtual address size returned by simpleMallocMultiDeviceMmapResize
//! @param[in] allocationSize  allocationSize returned by simpleMallocMultiDeviceMmapResize
//! @return CUresult
////////////////////////////////////////////////////////////////////////////
CUresult simpleMallocMultiDeviceMmapResizeCleanup(CUdeviceptr dptr, size_t size, size_t va_size, size_t *allocationSize);

////////////////////////////////////////////////////////////////////////////
//! Resizes a multi-device mapped virtual address region
//! @param[in] dptr  Virtual address reserved by simpleMallocMultiDeviceMmap
//! @param[in] va_size  Virtual address size returned by simpleMallocMultiDeviceMmap
//! @param[in] allocationSize  allocationSize returned by simpleMallocMultiDeviceMmap
//! @param[in] size  size of the new allocation
//! @param[in] residentDevices  vector of resident devices
//! @param[in] mappingDevices  vector of mapping devices
//! @param[in] align  Optional alignment requirement if desired, default is 0
//! @return CUresult
////////////////////////////////////////////////////////////////////////////
CUresult simpleMallocMultiDeviceMmapResize(CUdeviceptr dptr, size_t va_size, size_t *allocationSize, size_t size,
		const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
		size_t align = 0);

////////////////////////////////////////////////////////////////////////////
//! Allocate virtually contiguous memory backed on separate devices
//! @param[out] dptr            Virtual address reserved for allocation
//! @param[out] allocationSize  Actual amount of virtual address space reserved.
//!                             Allocation size is needed in the free operation.
//! @param[in] size             The minimum size to allocate (will be rounded up to accommodate
//!                             required granularity).
//! @param[in] residentDevices  Specifies what devices the allocation should be striped across.
//! @param[in] mappingDevices   Specifies what devices need to read/write to the allocation.
//! @param[in] align  Optional alignment requirement if desired, default is 0
//! @return CUresult
//! @note       The VA mappings will look like the following:
//!
//!     v-stripeSize-v                v-rounding -v
//!     +-----------------------------------------+
//!     |      D1     |      D2     |      D3     |
//!     +-----------------------------------------+
//!     ^-- dptr                      ^-- dptr + size
//!
//! Each device in the residentDevices list will get an equal sized stripe.
//! Excess memory allocated will be that meets the minimum
//! granularity requirements of all the devices.
//!
//! @note uses cuMemGetAllocationGranularity cuMemCreate cuMemMap and cuMemSetAccess
//!   function calls to organize the va space
//!
//! @note uses cuMemRelease to release the allocationHandle.  The allocation handle
//!   is not needed after its mappings are set up.
////////////////////////////////////////////////////////////////////////////
CUresult simpleMallocMultiDeviceMmap(CUdeviceptr *dptr, size_t va_size, size_t *allocationSize, size_t size,
		const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
		size_t align = 0);

////////////////////////////////////////////////////////////////////////////
//! Collect all of the devices whose memory can be mapped from cuDevice.
//! @param[in] cuDevice  The device to collect backing devices for
//! @return std::vector<CUdevice>  A vector of devices that can be mapped from cuDevice
////////////////////////////////////////////////////////////////////////////
std::vector<CUdevice> getBackingDevices(CUdevice cuDevice);

////////////////////////////////////////////////////////////////////////////
//! Cleanup function for the cuMemMap samples
//! @param[in] fileResources  File resources
//! @param[in] hostBuffers  Host buffers
//! @param[in] deviceBuffers  Device buffers
//! @param[in] cuContext  CUDA context
////////////////////////////////////////////////////////////////////////////
void fullCleanup(FileResources& fileResources, HostBuffers& hostBuffers, DeviceBuffers& deviceBuffers, CUcontext& cuContext);

#endif