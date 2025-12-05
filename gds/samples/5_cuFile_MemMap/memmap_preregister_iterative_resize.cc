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
 * This sample demonstrates use of cuFile with cuMemMap-ed allocations for
 * vector addition of C = A + B. This samples initializes the device memory
 * 0.5x the size of the vectors and then iteratively resizes the device memory
 * using cuMemMap-ed allocations. To prepare for this iterative memory region 
 * resize, cuFile will pre-emptively register the entire expected memory region.
 * To mimic the iterative region size, cuFile will showcase it's ability to
 * iteratively perform WRITEs on this region. It then proceeds to clear the
 * device memory and read the data back iteratively from the files. Vector 
 * addition is then performed on the GPU and the result is verified. Through 
 * the usage of cuMemMap-ed allocations, the user can specify the physical 
 * properties of their memory while retaining the contiguous nature of their 
 * access. Thus no change is required to the program structure.
 * 
 * This sample follows the same structure as the vectorAddDrv.cpp on 
 * CUDA samples repository.
 *
 * ./memmap_preregister_iterative_resize <testfileA> <testfileB> 
 * 
 * | Output |
 * Vector Addition cuMemMap with cuFile Iterative (Driver API)
 * Device 0 VIRTUAL ADDRESS MANAGEMENT SUPPORTED = true.
 * Total number of elements in each vector: 57671680
 * Size of each sysmem vector in bytes: 230686720
 * h_A(0xfb854fa00010), h_B(0xfb8541c00010), h_C(0xfb8512200010), elements: 57671680
 * ...
 * Result = PASS
 * Cleaning up
 * Freeing d_A
 * unmap address range: 460000000, size: 230686720
 * freeing address range: 460000000, size: 461373440
 * Freeing d_B
 * unmap address range: 480000000, size: 230686720
 * freeing address range: 480000000, size: 461373440
 * Freeing d_C
 * unmap address range: 440000000, size: 230686720
 * freeing address range: 440000000, size: 461373440
 */

#include <iostream>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <ctime>

// Include CUDA headers
#include <builtin_types.h>
#include <cuda.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_memmap.hpp"

static constexpr int NUM_ELEMENTS = 28835840;
static constexpr double EPSILON = 1e-7f;

#ifdef __cplusplus
extern "C" {
	extern void vectorAdd(const float *A, const float *B, float *C, int numElements);
}
#endif

// NOTE: Cleanup functionality has been moved to 
// common/cufile_sample_memmap.cpp to increase readability

// Host code
int main(int argc, char **argv) {
	std::srand(std::time(nullptr));
	size_t allocationSizeA = 0, allocationSizeB = 0, allocationSizeC = 0;
	float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
	std::string testfileA, testfileB;
	int fdA = -1, fdB = -1, ret = EXIT_SUCCESS;
	int N = NUM_ELEMENTS;
	size_t size = 0;
	int attributeVal = 0;
	size_t va_size = 0;
	size_t oldSize = 0;
	size_t cur_off = 0;
	size_t io_size = 0;
	size_t bytes_left = 0;
	size_t cur_size = 0;
	size_t chunk_size = 0;

	// cuFile Specific variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle_A = nullptr, cf_handle_B = nullptr;

	// CUDA Specific variables
	CUcontext cuContext;
	CUdevice cuDevice = 0;
	std::vector<CUdevice> mappingDevices;
	std::vector<CUdevice> backingDevices;
	CUdeviceptr d_A = 0, d_B = 0, d_C = 0;
	CUdeviceptr d_A1 = 0, d_B1 = 0;

	// Used later to help with cleanup
	FileResources fileResources {
		fdA, fdB,
		cf_handle_A,
		cf_handle_B
	};
	DeviceBuffers deviceBuffers {
		d_A, d_B, d_C,
		allocationSizeA,
		allocationSizeB,
		allocationSizeC,
		va_size
	};

	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <testfileA> <testfileB> " << std::endl;
		return EXIT_FAILURE;
	}

	// Initialize
	std::cout << "Vector Addition cuMemMap with cuFile Iterative (Driver API)" << std::endl;
	check_cudadrivercall(cuInit(0));
	check_cudadrivercall(cuDeviceGet(&cuDevice, 0));

	// Check that the selected device supports virtual address management
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
	backingDevices = getBackingDevices(cuDevice);
	// The vector addition happens on cuDevice, so the allocations need to be mapped there.
	mappingDevices = backingDevices;

	// Calculate the total number of elements and size of each vector
	N = N * backingDevices.size();
	size = N * sizeof(float);
	std::cout << "Total number of elements in each vector: " << N << std::endl;
	std::cout << "Size of each sysmem vector in bytes: " << size << std::endl;

	// Create context
#if CUDA_VERSION >= 13000
	check_cudadrivercall(cuCtxCreate(&cuContext, NULL, 0, cuDevice));
#else
	check_cudadrivercall(cuCtxCreate(&cuContext, 0, cuDevice));
#endif

	// Allocate input vectors h_A and h_B in host memory
	h_A = new float[N];
	h_B = new float[N];
	h_C = new float[N];
	// Initialize for resource cleanup
	HostBuffers hostBuffers {
		h_A, h_B, h_C
	};

	std::cout << "h_A(" << std::hex << h_A 
		<< "), h_B(" << std::hex << h_B 
		<< "), h_C(" << std::hex << h_C 
		<< "), elements: " << std::dec << N << std::endl;

	// Initialize input vectors
	std::generate(h_A, h_A + N, randFloat);
	std::generate(h_B, h_B + N, randFloat);

	// Allocate vectors in device memory
	// Note: A call to cuCtxEnablePeerAccess is not needed even though
	// the backing devices and mapping device are not the same.
	// This is because the cuMemSetAccess call explicitly specifies
	// the cross device mapping. cuMemSetAccess is still subject to the
	// constraints of cuDeviceCanAccessPeer for cross device mappings
	// (hence why we checked cuDeviceCanAccessPeer earlier).
	va_size = size * 2;
	check_cudadrivercall(simpleMallocMultiDeviceMmap(&d_C, va_size, &allocationSizeC, size/2, backingDevices, mappingDevices));
	check_cudadrivercall(simpleMallocMultiDeviceMmap(&d_A, va_size, &allocationSizeA, size/2, backingDevices, mappingDevices));
	check_cudadrivercall(simpleMallocMultiDeviceMmap(&d_B, va_size, &allocationSizeB, size/2, backingDevices, mappingDevices));
	std::cout << "Rounded size of each vector in bytes = d_A(" << std::hex << d_A << "):" 
		<< std::dec << allocationSizeA << ", d_B(" << std::hex << d_B << "):" << std::dec 
		<< allocationSizeB << ", d_C(" << std::hex << d_C << "):" << std::dec 
		<< allocationSizeC << std::endl;

	// Copy vectors from host memory to device memory
	check_cudadrivercall(cuMemcpyHtoD(d_A, h_A, allocationSizeA));
	check_cudadrivercall(cuMemcpyHtoD(d_B, h_B, allocationSizeB));

	// Get the file paths from the command line arguments
	testfileA = argv[1];
	testfileB = argv[2];

	status = cuFileDriverOpen();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFile driver open error: "
			<< cuFileGetErrorString(status) << std::endl;
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	}

	// Open file A to write/read
	std::cout << "Opening file " << testfileA << " for writing/reading" << std::endl;
	fdA = open(testfileA.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (fdA < 0) {
		std::cerr << "file open error:"
			<< cuFileGetErrorString(errno) << std::endl;
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	}

	// Register file A with cuFile
	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fdA;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle_A, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	}

	// Open file B to write/read
	std::cout << "Opening file " << testfileB << " for writing/reading" << std::endl;
	fdB = open(testfileB.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (fdB < 0) {
		std::cerr << "file open error: "
			<< cuFileGetErrorString(errno) << std::endl;
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	}

	// Register file B with cuFile
	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fdB;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle_B, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	}

	// Register device memory A
	std::cout << "Registering device memory for vector A of size: " << size << std::endl;
	status = cuFileBufRegister(reinterpret_cast<void*>(d_A), size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "buffer register A failed:"
			<< cuFileGetErrorString(status) << std::endl;
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	}

	// Register device memory B
	std::cout << "Registering device memory for vector B of size: " << size << std::endl;
	status = cuFileBufRegister(reinterpret_cast<void*>(d_B), size, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "buffer register B failed:"
			<< cuFileGetErrorString(status) << std::endl;
		cuFileBufDeregister(reinterpret_cast<void*>(d_A));
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	}

	// Doubling device memory to size
	std::cout << "Doubling device memory to size: " << size << std::endl;
	d_A1 = d_A + allocationSizeA;
	d_B1 = d_B + allocationSizeB;

	// Resize device memory C
	check_cudadrivercall(simpleMallocMultiDeviceMmapResize(d_C, va_size, &allocationSizeC, size, backingDevices, mappingDevices));

	// Resize device memory A (iterative)
	oldSize = allocationSizeA;
	cur_size = allocationSizeA;
	chunk_size = MiB(32); // Based on max 16 GPUs and granularity of 2MB
	// Resize in chunks of 32 MiB
	while (allocationSizeA < size) {
		if ((size - allocationSizeA) < chunk_size) {
			chunk_size = size - allocationSizeA;
		}
		// Make the new allocation beyond the registered size
		cur_size = allocationSizeA + chunk_size;
		std::cout << "Resize d_A memory to new current size: " << cur_size << std::endl;
		check_cudadrivercall(simpleMallocMultiDeviceMmapResize(d_A, va_size, &allocationSizeA, cur_size, backingDevices, mappingDevices));
	}

	// Copy the rest of the memory from host to newly resized memory for A
	check_cudadrivercall(cuMemcpyHtoD(d_A1, reinterpret_cast<char*>(h_A) + oldSize, size - oldSize));
	// Resize device memory B
	oldSize = allocationSizeB;
	check_cudadrivercall(simpleMallocMultiDeviceMmapResize(d_B, va_size, &allocationSizeB, size, backingDevices, mappingDevices));
	// Copy the rest of the memory from host to newly resized memory for B
	check_cudadrivercall(cuMemcpyHtoD(d_B1, reinterpret_cast<char*>(h_B) + oldSize, size - oldSize));
	check_cudadrivercall(cuStreamSynchronize(0));
   
	std::cout << "New size of each vector in bytes: d_A(" << std::hex << 
		d_A << "):" << std::dec << allocationSizeA << " d_B(" << std::hex
		<< d_B << "):" << std::dec << allocationSizeB << " d_C(" << std::hex
		<< d_C << "):" << std::dec << allocationSizeC << std::endl;
	std::cout << "No need to explicitly re-register device memory, pre-emptively registered for N instead of N / 2" << std::endl;

	std::cout << "Iteratively writing from device memory A to file: " << testfileA << std::endl;
	cur_off = 0;
	io_size = MiB(2);
	bytes_left = size;
	while (bytes_left) {
		if (bytes_left < io_size) {
			io_size = bytes_left;
		}

		ret = cuFileWrite(cf_handle_A, reinterpret_cast<void*>(d_A), io_size, cur_off, cur_off);
		if (ret < 0) {
			if (IS_CUFILE_ERR(ret))
				std::cerr << "Write failed: " << cuFileGetErrorString(ret) << std::endl;
			else
				std::cerr << "Write failed: " << cuFileGetErrorString(errno) << std::endl;
	
			cuFileBufDeregister(reinterpret_cast<void*>(d_A));
			cuFileBufDeregister(reinterpret_cast<void*>(d_B));
			fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
			return EXIT_FAILURE;
		} else {
			bytes_left -= ret;
			std::cout << "Current written bytes at offset: " << cur_off << " to d_A ret: " << ret << ", left: " << bytes_left << std::endl;
			cur_off += ret;
			ret = EXIT_SUCCESS;
		}
	}

	// Write device memory B to file B
	std::cout << "Writing from device memory B to file: " << testfileB << std::endl;
	ret = cuFileWrite(cf_handle_B, reinterpret_cast<void*>(d_B), allocationSizeB, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Write failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Write failed: " << cuFileGetErrorString(errno) << std::endl;

		cuFileBufDeregister(reinterpret_cast<void*>(d_A));
		cuFileBufDeregister(reinterpret_cast<void*>(d_B));
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	} else {
		std::cout << "Written " << ret << " bytes to " << testfileB << std::endl;
		ret = EXIT_SUCCESS;
	}

	// Clear device memory
	std::cout << "Clearing device memory" << std::endl;
	check_cudadrivercall(cuMemsetD8(d_A, 0x0, allocationSizeA));
	check_cudadrivercall(cuMemsetD8(d_B, 0x0, allocationSizeB));
	check_cudadrivercall(cuMemsetD8(d_C, 0x0, allocationSizeC));
	check_cudadrivercall(cuStreamSynchronize(0));

	std::cout << "Iteratively reading from file: " << testfileA << " to device memory A" << std::endl;
	cur_off = 0;
	io_size = MiB(2);
	bytes_left = size;
	while (bytes_left) {
		if (bytes_left < io_size) {
			io_size = bytes_left;
		}

		ret = cuFileRead(cf_handle_A, reinterpret_cast<void*>(d_A), io_size, cur_off, cur_off);
		if (ret < 0) {
			if (IS_CUFILE_ERR(ret))
				std::cerr << "Read failed: " << cuFileGetErrorString(ret) << std::endl;
			else
				std::cerr << "Read failed: " << cuFileGetErrorString(errno) << std::endl;
	
			cuFileBufDeregister(reinterpret_cast<void*>(d_A));
			cuFileBufDeregister(reinterpret_cast<void*>(d_B));
			fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
			return EXIT_FAILURE;
		} else {
			bytes_left -= ret;
			std::cout << "Current read bytes at offset: " << cur_off << " to d_A ret: " << ret << ", left: " << bytes_left << std::endl;
			cur_off += ret;
			ret = EXIT_SUCCESS;
		}
	}

	// Read device memory A from file A
	std::cout << "Reading from file: " << testfileA << " to device memory A" << std::endl;
	ret = cuFileRead(cf_handle_A, reinterpret_cast<void*>(d_A), allocationSizeA, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Read failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Read failed: " << cuFileGetErrorString(errno) << std::endl;
		
		cuFileBufDeregister(reinterpret_cast<void*>(d_A));
		cuFileBufDeregister(reinterpret_cast<void*>(d_B));
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	} else {
		std::cout << "Read " << ret << " bytes from " << testfileA << " to device memory A" << std::endl;
		ret = EXIT_SUCCESS;
	}

	// Read device memory B from file B
	std::cout << "Reading from file: " << testfileB << " to device memory B" << std::endl;
	ret = cuFileRead(cf_handle_B, reinterpret_cast<void*>(d_B), allocationSizeB, 0, 0);
	if (ret < 0) {
		if (IS_CUFILE_ERR(ret))
			std::cerr << "Read failed: " << cuFileGetErrorString(ret) << std::endl;
		else
			std::cerr << "Read failed: " << cuFileGetErrorString(errno) << std::endl;

		cuFileBufDeregister(reinterpret_cast<void*>(d_A));
		cuFileBufDeregister(reinterpret_cast<void*>(d_B));
		fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
		return EXIT_FAILURE;
	} else {
		std::cout << "Read " << ret << " bytes from " << testfileB << " to device memory B" << std::endl;
		ret = EXIT_SUCCESS;
	}

	// Vector addition on the GPU
	std::cout << "GPU vector ADD for " << N << " elements size: " << N * sizeof(float) << std::endl;
	vectorAdd(reinterpret_cast<const float*>(d_A), reinterpret_cast<const float*>(d_B), reinterpret_cast<float*>(d_C), N);

	// Copy result from device memory to host memory
	std::cout << "Copying result from device memory to host memory" << std::endl;
	check_cudadrivercall(cuMemcpyDtoH(h_C, d_C, N * sizeof(float)));
	check_cudadrivercall(cuStreamSynchronize(0));

	// Verify result
	std::cout << "Verifying result" << std::endl;
	int verified = 0;
	for (int i = 0; i < N; ++i) {
		double sum = h_A[i] + h_B[i];
		if (fabs(h_C[i] - sum) > EPSILON) {
			std::cerr << "Error element: " << i << " h_C=" << h_C[i] << ", sum = " << sum << std::endl;
			break;
		}
		verified++;
	}
	std::cout << (verified == N ? "Result = PASS" : "Result = FAIL") << std::endl;

	// Cleanup
	std::cout << "Cleaning up" << std::endl;
	status = cuFileBufDeregister(reinterpret_cast<void*>(d_A));
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "buffer deregister failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}
	status = cuFileBufDeregister(reinterpret_cast<void*>(d_B));
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "buffer deregister failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}
	fullCleanup(fileResources, hostBuffers, deviceBuffers, cuContext);
	return (verified == N ? EXIT_SUCCESS : EXIT_FAILURE);
}
