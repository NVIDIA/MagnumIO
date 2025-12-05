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

#ifndef __CUFILE_SAMPLE_THRUST_HPP_
#define __CUFILE_SAMPLE_THRUST_HPP_

#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>

// includes, Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "cufile_sample_utils.hpp"
#include "cufile_sample_memmap.hpp"

// Global variable for storing the index of the found value with thrust
__device__ long long int d_index = -1;

////////////////////////////////////////////////////////////////////////////
//! Thrust concurrent implementation of thrust::find() algorithm
//! @param[in] begin  Pointer to the beginning of the array
//! @param[in] value  Value to find
//! @param[in] numElements  Number of elements in the array
//! @param[in] d  Device index
////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void thrust_concurrent_find_kernel(T* begin, T value, size_t numElements, int d) {
	thrust::device_ptr<T> iter;
	iter = thrust::find(thrust::device, thrust::device_pointer_cast(begin), thrust::device_pointer_cast(begin+numElements-1), value);
	if (*iter == value) {
		d_index = thrust::distance(thrust::device_pointer_cast(begin), iter);
		d_index += d*numElements;
	}
}

////////////////////////////////////////////////////////////////////////////
//! Performs concurrent implementation of thrust::find() algorithm
//! @param[in] begin  Pointer to the beginning of the array
//! @param[in] end  Pointer to the end of the array
//! @param[in] value  Value to find
//! @param[in] num_devices  Number of devices
//! @return Index if value is found, else returns -1
////////////////////////////////////////////////////////////////////////////
template <typename T>
int thrust_concurrent_find(thrust::device_ptr<T> begin, thrust::device_ptr<T> end, T value, int num_devices) {
	size_t num_elements = (thrust::distance(begin, end))/num_devices;

	cudaStream_t streams[num_devices];
	for(int i=0; i<num_devices; i++) {
		check_cudaruntimecall(cudaSetDevice(i));
		check_cudaruntimecall(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
	}

	for (int i=0; i<num_devices; i++) {
		check_cudaruntimecall(cudaSetDevice(i));
		thrust_concurrent_find_kernel<T><<< 1, 1, 0, streams[i]>>>((T*) thrust::raw_pointer_cast(begin+i*num_elements), value, num_elements, i);
	}

	for (int i = 0; i < num_devices; i++) {
		check_cudaruntimecall(cudaSetDevice(i));
		check_cudaruntimecall(cudaStreamSynchronize(streams[i]));
		check_cudaruntimecall(cudaStreamDestroy(streams[i]));
	}

	int h_index[num_devices];
	for (int i = 0; i < num_devices; i++) {
		check_cudaruntimecall(cudaSetDevice(i));
		check_cudaruntimecall(cudaMemcpyFromSymbol(&h_index[i], d_index, sizeof(int)));
		if (h_index[i] != -1) return h_index[i];
	}
	return -1;
}

////////////////////////////////////////////////////////////////////////////
//! Class for managing Thrust vectors backed by device memory
//! @param[in] num_devices  Number of devices
////////////////////////////////////////////////////////////////////////////
template <typename T>
class CuFileThrustVector {
public:
	CuFileThrustVector(int num_devices) {
		// Initialize
		CUfileError_t status;
		CUdevice cuDevice;
		check_cudadrivercall(cuInit(0));
		check_cudadrivercall(cuDeviceGet(&cuDevice, 0));

		// Get mapping and backing devices
		for (int i=0; i<num_devices; i++) {
			check_cudadrivercall(cuDeviceGet(&cuDevice, i));
			mappingDevices.push_back(cuDevice);
			backingDevices.push_back(getBackingDevices(cuDevice));
		}

		// Initialize driver for GDS
		status = cuFileDriverOpen();
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "cuFile driver open error: "
				<< cuFileGetErrorString(status) << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	// Allocate virtually contiguous memory backed on separate devices
	// Returns thrust device pointer to the allocated space
	thrust::device_ptr<T> cufile_thrust_device_pointer(size_t N) {
		size = N * sizeof(T);
		check_cudadrivercall(simpleMallocMultiDeviceMmap(&devPtr, size, &allocationSize, size, backingDevices[0], mappingDevices));
		dev_ptr = thrust::device_pointer_cast((T*)devPtr);
		return dev_ptr;
	}


	// Returns raw pointer to the vector
	T* get_raw_pointer() {
		return thrust::raw_pointer_cast(dev_ptr);
	}

	// Write the vector to file using cuFile APIs
	int write_to_file(std::string file, off_t file_offset = 0, int index_offset = 0) {
		int fd = -1, ret = 0;
		CUfileError_t status;
		CUfileDescr_t cf_descr;
		CUfileHandle_t cf_handle = nullptr;
		off_t ptr_offset = index_offset * sizeof(T);

		fd = open(file.c_str(), O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0664);
		if (fd < 0) {
			std::cerr << "File open error: "
				<< cuFileGetErrorString(errno) << std::endl;
			return -1;
		}

		std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "File register error: "
				<< cuFileGetErrorString(status) << std::endl;
			close(fd);
			return -1;
		}

		// Registers device memory
		std::cout << "Registering device memory of size: " << std::dec << this->allocationSize << " bytes" << std::endl;
		status = cuFileBufRegister((void*)this->get_raw_pointer(), this->allocationSize, 0);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer register failed: "
				<< cuFileGetErrorString(status) << std::endl;
			cuFileHandleDeregister(cf_handle);
			close(fd);
			return -1;
		}

		check_cudadrivercall(cuStreamSynchronize(0));

		ret = cuFileWrite(cf_handle, (void*)this->get_raw_pointer(), this->allocationSize, file_offset, ptr_offset);
		if (ret < 0) {
			if (IS_CUFILE_ERR(ret))
				std::cerr << "Write failed: " << cuFileGetErrorString(ret) << std::endl;
			else
				std::cerr << "Write failed: " << cuFileGetErrorString(errno) << std::endl;
			ret = -1;
		} else {
			std::cout << "Written " << std::dec << ret << " bytes" << std::endl;
			ret = 0;
		}

		// Deregister the device memory
		std::cout << "Deregistering device memory" << std::endl;
		status = cuFileBufDeregister((void*)this->get_raw_pointer());
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer deregister failed: "
				<< cuFileGetErrorString(status) << std::endl;
			ret = -1;
		}

		cuFileHandleDeregister(cf_handle);
		close(fd);
		return ret;
	}

	// Copy host vector to the device vector
	void operator= (thrust::host_vector<T> h_vec) {
		check_cudadrivercall(cuMemcpyHtoD((CUdeviceptr)this->get_raw_pointer(), (const void *) thrust::raw_pointer_cast(&h_vec[0]), this->size));
	}

	// Assign value to the device vector
	void operator= (T value){
		check_cudadrivercall(cuMemsetD8((CUdeviceptr)this->get_raw_pointer(), value, this->allocationSize));
	}

	// Read from file to device vector using cuFile APIs
	int read_from_file(std::string file, off_t file_offset = 0, int index_offset = 0) {
		int fd = -1, ret = 0;
		CUfileError_t status;
		CUfileDescr_t cf_descr;
		CUfileHandle_t cf_handle = nullptr;
		off_t ptr_offset = index_offset * sizeof(T);

		fd = open(file.c_str(), O_RDONLY | O_DIRECT, 0664);
		if (fd < 0) {
			std::cerr << "File open error: "
				<< cuFileGetErrorString(errno) << std::endl;
			return -1;
		}

		std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "File register error: "
				<< cuFileGetErrorString(status) << std::endl;
			close(fd);
			return -1;
		}

		// Registers device memory
		std::cout << "Registering device memory of size: " << std::dec << this->allocationSize << " bytes" << std::endl;
		status = cuFileBufRegister((void*)this->get_raw_pointer(), this->allocationSize, 0);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer register failed: "
				<< cuFileGetErrorString(status) << std::endl;
			cuFileHandleDeregister(cf_handle);
			close(fd);
			return -1;
		}

		check_cudadrivercall(cuStreamSynchronize(0));

		ret = cuFileRead(cf_handle, (void*)this->get_raw_pointer(), this->allocationSize, file_offset, ptr_offset);
		if (ret < 0) {
			if (IS_CUFILE_ERR(ret))
				std::cerr << "Read failed: " << cuFileGetErrorString(ret) << std::endl;
			else
				std::cerr << "Read failed: " << cuFileGetErrorString(errno) << std::endl;
			ret = -1;
		} else {
			std::cout << "Read " << std::dec << ret << " bytes" << std::endl;
			ret = 0;
		}

		// Deregister the device memory
		std::cout << "Deregistering device memory" << std::endl;
		status = cuFileBufDeregister((void*)this->get_raw_pointer());
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer deregister failed: "
				<< cuFileGetErrorString(status) << std::endl;
			ret = -1;
		}

		cuFileHandleDeregister(cf_handle);
		close(fd);
		return ret;
	}

	~CuFileThrustVector() {
		std::cout << "Freeing device memory" << std::endl;
		check_cudadrivercall(simpleFreeMultiDeviceMmap(devPtr, allocationSize, size));
		CUfileError_t status = cuFileDriverClose();
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "cuFile driver close error: "
				<< cuFileGetErrorString(status) << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	CUdeviceptr devPtr;
	thrust::device_ptr<T> dev_ptr;
	size_t allocationSize, size;
	std::vector<CUdevice> mappingDevices;
	std::vector<std::vector<CUdevice>> backingDevices;
	int num_devices;
};

#endif
