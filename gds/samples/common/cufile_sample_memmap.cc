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

#include <iostream>
#include <unistd.h>
#include <cstdlib>

// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_memmap.hpp"

CUresult simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size, size_t va_size) {
	CUresult status = CUDA_SUCCESS;

	// Unmap the mapped virtual memory region. Since the handles
	// to the mapped backing stores have already been released
	// by cuMemRelease, and these are the only/last mappings
	// referencing them, the backing stores will be freed.
	// Since the memory has been unmapped after this call,
	// accessing the specified va range will result in a fault
	// (until it is remapped).
	std::cout << "unmap address range: " << std::hex << static_cast<uint64_t>(dptr)
		<< std::dec << ", size: " << std::dec << size << std::endl;
	status = cuMemUnmap(dptr, size);
	if (status != CUDA_SUCCESS) {
		return status;
	}

	// Free the virtual address region. This allows the virtual
	// address region to be reused by future cuMemAddressReserve
	// calls. This also allows the virtual address region to be
	// used by other allocation made through operating system
	// calls like malloc & mmap.
	std::cout << "freeing address range: " << std::hex << static_cast<uint64_t>(dptr)
		<< ", size: " << std::dec << va_size << std::endl;
	status = cuMemAddressFree(dptr, va_size);
	if (status != CUDA_SUCCESS) {
		return status;
	}

	return status;
}

CUresult simpleMallocMultiDeviceMmapResizeCleanup(CUdeviceptr dptr, size_t size, size_t va_size, size_t *allocationSize) {

	CUresult status = CUDA_SUCCESS;
	if (dptr) {
		status = simpleFreeMultiDeviceMmap(dptr, size, va_size);
		*allocationSize = 0;
	}	

	return status;
}

CUresult simpleMallocMultiDeviceMmapResize(CUdeviceptr dptr, size_t va_size, size_t *allocationSize, size_t size,
		const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
		size_t align) {

	// Check that allocationSize is valid
	assert(allocationSize && "allocationSize must not be null");

	CUresult status = CUDA_SUCCESS;
	size_t min_granularity = 0;
	size_t oldStripeSize, stripeSize;
	size_t add_size;

	// Setup the properties common for all the chunks
	// The allocations will be device pinned memory.
	// This property structure describes the physical 
	// location where the memory will be allocated via
	// cuMemCreate along with additional properties.
	// In this case, the allocation will be pinned device
	// memory local to a given device.
	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.allocFlags.gpuDirectRDMACapable = 1;

	// Get the minimum granularity needed for the resident devices
	// (the max of the minimum granularity of each participating device)
	for (unsigned idx = 0; idx < residentDevices.size(); idx++) {
		size_t granularity = 0;

		// Get the minimum granularity for residentDevices[idx]
		prop.location.id = residentDevices[idx];
		status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		if (status != CUDA_SUCCESS) {
			simpleMallocMultiDeviceMmapResizeCleanup(dptr, size, va_size, allocationSize);
			return status;
		}

		if (min_granularity < granularity) {
			min_granularity = granularity;
		}
	}

	// Get the minimum granularity needed for the accessing devices
	// (the max of the minimum granularity of each participating device)
	for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
		size_t granularity = 0;

		// Get the minimum granularity for mappingDevices[idx]
		prop.location.id = mappingDevices[idx];
		status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		if (status != CUDA_SUCCESS) {
			simpleMallocMultiDeviceMmapResizeCleanup(dptr, size, va_size, allocationSize);
			return status;
		}

		if (min_granularity < granularity) {
			min_granularity = granularity;
		}
	}

	// Round up the size such that we can evenly split it into a 
	// stripe size that meets the granularity requirements. Essentially
	// size = N * residentDevices.size() * min_granularity is the requirement,
	// since each piece of the allocation will be stripeSize = N * min_granularity
	// and the min_granularity requirement applies to each stripeSize piece
	// of the allocation.
	size = round_up(size, residentDevices.size() * min_granularity);
	add_size = size - *allocationSize;
	oldStripeSize = *allocationSize / residentDevices.size();
	stripeSize = add_size / residentDevices.size();

	std::cout << "granularity: " << min_granularity << ", oldsize: " << *allocationSize 
		<< ", add_size: " << add_size << ", new_size: " << size << ", stripeSize: " << stripeSize 
		<< ", oldStripeSize: " << oldStripeSize << std::endl;

	// Create and map the backings on each gpu
	// NOTE: reusing CUmemAllocationProp prop from earlier 
	// with prop.type & prop.location.type already specified.
	for (size_t idx = 0; idx < residentDevices.size(); idx++) {
		CUresult status2 = CUDA_SUCCESS;
		// Set the location for this chunk to this device
		prop.location.id = residentDevices[idx];

		CUdevice device = 0;
		check_cudadrivercall(cuDeviceGet(&device, prop.location.id));	

		// On some systems it is possible that cuMemCreate might fail if
		// gpuDirectRDMACapable is set to true in the alloc_flags.
		// The following check ensures that we only set this flag in case
		// it is supported by system.
		int gpuDirectRDMACapable = 0;
		status = cuDeviceGetAttribute(
			&gpuDirectRDMACapable,
			CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
			device
		);

		if (status != CUDA_SUCCESS) {
			std::cerr << "Couldn't fetch CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED attribute, "
				<< "error " << status << ", disabling gpuDirectRDMACapable flag" << std::endl;
			prop.allocFlags.gpuDirectRDMACapable = 0;
		} else {
			std::cout << "Setting prop.allocFlags.gpuDirectRDMACapable flag" << gpuDirectRDMACapable << std::endl;
			prop.allocFlags.gpuDirectRDMACapable = gpuDirectRDMACapable;
		}

		// Create the allocation as a pinned allocation on this device
		std::cout << "adding physical mem of size " << stripeSize << " to gpu: " << idx << std::endl;
		CUmemGenericAllocationHandle allocationHandle = 0;
		status = cuMemCreate(&allocationHandle, stripeSize, &prop, 0);
		if (status != CUDA_SUCCESS) {
			simpleMallocMultiDeviceMmapResizeCleanup(dptr, size, va_size, allocationSize);
			return status;
		}

		// Assign the chunk to the appropriate VA range and release the handle.
		// After mapping the memory, it can be referenced by virtual address.
		// Since we do not need to make any other mappings of this memory or export it,
		// we no longer need and can release the allocationHandle.
		// The allocation will be kept live until it is unmapped.
		CUdeviceptr offset_dptr = dptr + *allocationSize + (stripeSize * idx);
		std::cout << "adding new mapping to VA space start at 0x"
				<< std::hex << static_cast<uint64_t>(offset_dptr)
				<< std::dec << " new size: " << stripeSize
				<< " gpu: " << idx << std::endl;
		status = cuMemMap(offset_dptr, stripeSize, 0, allocationHandle, 0);

		// The allocationHandle needs to be released even if the mapping failed.
		std::cout << "freeing allocation handle" << std::endl;
		status2 = cuMemRelease(allocationHandle);
		if (status == CUDA_SUCCESS) {
			status = status2; // cuMemRelease should not have failed here
		}

		// Cleanup in case of any mapping failures.
		if (status != CUDA_SUCCESS) {
			std::cerr << "failed mapping VA" << std::endl;
			simpleMallocMultiDeviceMmapResizeCleanup(dptr, size, va_size, allocationSize);
			return status;
		}
	}

	// Return the rounded up size to the caller for use in the free
	if (allocationSize) {
		*allocationSize = size;
	}
	
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
	std::cout << "setting access to entire VA space for dptr: " 
		<< std::hex << (uint64_t)dptr << std::dec << " size: " 
		<< size << " on " << mappingDevices.size() << " GPUs" 
		<< std::endl;
	status = cuMemSetAccess(dptr, size, &accessDescriptors[0], accessDescriptors.size());
	if (status != CUDA_SUCCESS) {
		std::cerr << "failed to set access to entire VA space" << std::endl;
		simpleMallocMultiDeviceMmapResizeCleanup(dptr, size, va_size, allocationSize);
		return status;
	}

	return status;
}

CUresult simpleMallocMultiDeviceMmap(CUdeviceptr *dptr, size_t va_size, size_t *allocationSize, size_t size,
		const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
		size_t align) {

	CUresult status = CUDA_SUCCESS;
	size_t min_granularity = 0;
	size_t stripeSize;

	// Setup the properties common for all the chunks, 
	// allocations will be device pinned memory.
	// This property structure describes the physical location
	// where the memory will be allocated via cuMemCreate
	// along with additional properties.
	// In this case, the allocation will be pinned device
	// memory local to a given device.
	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.allocFlags.gpuDirectRDMACapable = 1;

	// Get the minimum granularity needed for the resident devices
	// (the max of the minimum granularity of each participating device)
	for (unsigned idx = 0; idx < residentDevices.size(); idx++) {
		size_t granularity = 0;

		// Get the minimum granularity for residentDevices[idx]
		prop.location.id = residentDevices[idx];
		status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		if (status != CUDA_SUCCESS) {
			std::cerr << "Failed to get allocation granularity for resident device " << idx << std::endl;
			if (dptr) simpleFreeMultiDeviceMmap(*dptr, size, va_size);
			return status;
		}
		
		std::cout << "resident device idx: " << idx << ", gpuid: " 
			<< prop.location.id << ", granularity: " << granularity << std::endl;
		if (min_granularity < granularity) {
			min_granularity = granularity;
		}
	}

	// Get the minimum granularity needed for the accessing devices
	// (the max of the minimum granularity of each participating device)
	for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
		size_t granularity = 0;

		// Get the minimum granularity for mappingDevices[idx]
		prop.location.id = mappingDevices[idx];
		status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		if (status != CUDA_SUCCESS) {
			std::cerr << "Failed to get allocation granularity for mapping device " << idx << std::endl;
			if (dptr) simpleFreeMultiDeviceMmap(*dptr, size, va_size);
			return status;
		}

		std::cout << "mapping device idx: " << idx << ", gpuid: " 
			<< prop.location.id << ", granularity: " << granularity << std::endl;
		if (min_granularity < granularity) {
			min_granularity = granularity;
		}
	}

	// Round up the size such that we can evenly split it into a 
	// stripe size that meets the granularity requirements. Essentially
	// size = N * residentDevices.size() * min_granularity is the requirement,
	// since each piece of the allocation will be stripeSize = N * min_granularity
	// and the min_granularity requirement applies to each stripeSize piece
	// of the allocation.
	size = round_up(size, residentDevices.size() * min_granularity);
	stripeSize = size / residentDevices.size();
	std::cout << "actual allocation size: " << size  << ", min granularity: " 
		<< min_granularity << ", total devs: " << residentDevices.size() << std::endl;

	// Return the rounded up size to the caller for use in the free
	if (allocationSize) {
		*allocationSize = size;
	}

	// Reserve the required contiguous VA space for the allocations
	status = cuMemAddressReserve(dptr, va_size, align, 0, 0);
	if (status != CUDA_SUCCESS) {
		std::cerr << "Failed to reserve va addr: " << std::hex << static_cast<uint64_t>(*dptr) 
			<< std::dec << ", size: " << va_size << std::endl;
		if (dptr) simpleFreeMultiDeviceMmap(*dptr, size, va_size);
		return status;
	}

	// Create and map the backings on each gpu
	// Note: reusing CUmemAllocationProp prop from earlier
	// with prop.type & prop.location.type already specified.
	for (size_t idx = 0; idx < residentDevices.size(); idx++) {
		CUresult status2 = CUDA_SUCCESS;

		// Set the location for this chunk to this device
		prop.location.id = residentDevices[idx];

		CUdevice device = 0;
		check_cudadrivercall(cuDeviceGet(&device, prop.location.id));	

		// On some systems it is possible that cuMemCreate might fail
		// if gpuDirectRDMACapable is set to true in the alloc_flags.
		// The following check ensures that we only set this flag in
		// case it is supported by system.
		int gpuDirectRDMACapable = 0;
		status = cuDeviceGetAttribute(
			&gpuDirectRDMACapable,
			CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
			device
		);

		if (status != CUDA_SUCCESS) {
			std::cerr << "Couldn't fetch CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED attribute, error " 
				<< status << ", disabling gpuDirectRDMACapable flag" << std::endl;
			prop.allocFlags.gpuDirectRDMACapable = 0;
		} else {
			std::cout << "Setting prop.allocFlags.gpuDirectRDMACapable flag" << gpuDirectRDMACapable << std::endl;
			prop.allocFlags.gpuDirectRDMACapable = gpuDirectRDMACapable;
		}

		// Create the allocation as a pinned allocation on this device
		CUmemGenericAllocationHandle allocationHandle = 0;
		status = cuMemCreate(&allocationHandle, stripeSize, &prop, 0);
		if (status != CUDA_SUCCESS) {
			std::cerr << "Failed to create allocation on device " << idx << std::endl;
			if (dptr) simpleFreeMultiDeviceMmap(*dptr, size, va_size);
			return status;
		}

		// Assign the chunk to the appropriate VA range and release the handle.
		// After mapping the memory, it can be referenced by virtual address.
		// Since we do not need to make any other mappings of this memory or export it,
		// we no longer need and can release the allocationHandle.
		// The allocation will be kept live until it is unmapped.
		status = cuMemMap(*dptr + (stripeSize * idx), stripeSize, 0, allocationHandle, 0);

		// The allocationHandle needs to be released even if the mapping failed.
		std::cout << "freeing allocation handle" << std::endl;
		status2 = cuMemRelease(allocationHandle);
		if (status == CUDA_SUCCESS) {
			status = status2; // cuMemRelease should not have failed here
		}

		// Cleanup in case of any mapping failures.
		if (status != CUDA_SUCCESS) {
			std::cerr << "failed mapping VA" << std::endl;
			if (dptr) simpleFreeMultiDeviceMmap(*dptr, size, va_size);
			return status;
		}
	}

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
		std::cerr << "failed to set access to entire VA space" << std::endl;
		if (dptr) simpleFreeMultiDeviceMmap(*dptr, size, va_size);
		return status;
	}

	return status;
}

std::vector<CUdevice> getBackingDevices(CUdevice cuDevice) {
	int num_devices = 0;

	check_cudadrivercall(cuDeviceGetCount(&num_devices));

	std::vector<CUdevice> backingDevices;
	backingDevices.push_back(cuDevice);
	for (int dev = 0; dev < num_devices; dev++) {
		int capable = 0;
		int attributeVal = 0;

		// The mapping device is already in the backingDevices vector
		if (dev == cuDevice) {
			continue;
		}

		// Only peer capable devices can map each others memory
		check_cudadrivercall(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
		if (!capable) {
			continue;
		}

		// The device needs to support virtual address management for the required apis to work
		check_cudadrivercall(cuDeviceGetAttribute(&attributeVal,
							  CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
							  cuDevice));
		if (attributeVal == 0) {
			continue;
		}

		backingDevices.push_back(dev);
	}

	return backingDevices;
}

void fullCleanup(FileResources& fileResources, HostBuffers& hostBuffers, DeviceBuffers& deviceBuffers, CUcontext& cuContext) {
	CUfileError_t status;

	if (fileResources.fdA > 0)
		close(fileResources.fdA);

	if (fileResources.fdB > 0)
		close(fileResources.fdB);

	if (fileResources.cf_handle_A)
		cuFileHandleDeregister(fileResources.cf_handle_A);

	if (fileResources.cf_handle_B)
		cuFileHandleDeregister(fileResources.cf_handle_B);

	// Free host memory
	if (hostBuffers.h_A) delete[] hostBuffers.h_A;
	if (hostBuffers.h_B) delete[] hostBuffers.h_B;
	if (hostBuffers.h_C) delete[] hostBuffers.h_C;

	// Free device memory
	std::cout << "Freeing d_A" << std::endl;
	if (deviceBuffers.d_A)
		check_cudadrivercall(simpleFreeMultiDeviceMmap(deviceBuffers.d_A, deviceBuffers.allocationSizeA, deviceBuffers.va_size));
	std::cout << "Freeing d_B" << std::endl;
	if (deviceBuffers.d_B)
		check_cudadrivercall(simpleFreeMultiDeviceMmap(deviceBuffers.d_B, deviceBuffers.allocationSizeB, deviceBuffers.va_size));
	std::cout << "Freeing d_C" << std::endl;
	if (deviceBuffers.d_C)
		check_cudadrivercall(simpleFreeMultiDeviceMmap(deviceBuffers.d_C, deviceBuffers.allocationSizeC, deviceBuffers.va_size));

	status = cuFileDriverClose();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFileDriverClose failed: "
			<< cuFileGetErrorString(status) << std::endl;
	}

	check_cudadrivercall(cuCtxDestroy(cuContext));
}
