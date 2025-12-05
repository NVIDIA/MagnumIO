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
 * Sample cuFileBatchIOSubmit Cancel Test.
 * This sample program cancels I/O after submitting a read request through Batch API's.
 * 
 * ./batch_cancel <testfile> <gpuid> <numbatchentries(1-128)> <nondirectflag(0-all_direct, 1-half_direct)>
 * NOTE: nonDirFlag is optional, if not provided, it defaults to 0.
 *
 * | Output |
 * Opening file: <testfile>
 * Registering device memory of size: 4096 for <numbatchentries> batch entries
 * Setting Up Batch
 * Submitting Batch IO
 * Submitting IO: index 0, size: 4096, file_offset: 0, devPtr: 0xf3d7d9e00000
 * ...
 * Cancelling Batch IO
 * Batch IO Canceled
 * Got events: 0
 * cuFileBufDeregister done
 * cuFileHandleDeregister, close, cudaFree done
 * cuFileDriverClose done
 */

#include <iostream>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

static constexpr size_t MAX_BUFFER_SIZE = KiB(4);
static constexpr size_t MAX_BATCH_IOS = 128;

int main(int argc, char *argv[]) {
	std::array<int, MAX_BATCH_IOS> fds;
	std::array<void*, MAX_BATCH_IOS> devPtrs;
	int nonDirFlag = 0, gpuid = 0;
	ssize_t ret = EXIT_SUCCESS;
	std::string testfile;
	unsigned int flags = 0;
	unsigned int nr = 0;
	unsigned int batch_size = 0;
	size_t registered_buffers = 0;

	// cuFile specific variables
	CUfileError_t status;
	CUfileError_t errorBatch;
	CUfileBatchHandle_t batch_id = nullptr;
	std::array<CUfileDescr_t, MAX_BATCH_IOS> cf_descrs;
	std::array<CUfileHandle_t, MAX_BATCH_IOS> cf_handles;
	std::array<CUfileIOParams_t, MAX_BATCH_IOS> io_batch_params;
	std::array<CUfileIOEvents_t, MAX_BATCH_IOS> io_batch_events;

	// Initialize arrays
	fds.fill(-1);
	devPtrs.fill(nullptr);
	cf_handles.fill(nullptr);
	cf_descrs.fill(CUfileDescr_t{});
	io_batch_params.fill(CUfileIOParams_t{});
	io_batch_events.fill(CUfileIOEvents_t{});

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testfile> <gpuid> <numbatchentries(1-128)> "
			<< "<nondirectflag(0-all_direct, 1-half_direct)>" << std::endl;
		return EXIT_FAILURE;
	}

	// Parse command line arguments
	testfile = argv[1];
	gpuid = parseInt(argv[2]);
	batch_size = parseInt(argv[3]);

	if (batch_size > MAX_BATCH_IOS || batch_size < 1) {
		std::cerr << "Requested batch size is below 1 or exceeds maximum batch size limit: " << MAX_BATCH_IOS << std::endl;
		return EXIT_FAILURE;
	}

	// Parse optional command line arguments
	if (argc > 4) {
		nonDirFlag = parseInt(argv[4]);
		if (nonDirFlag != 0 && nonDirFlag != 1) {
			std::cerr << "Invalid nonDirFlag: " << nonDirFlag << std::endl;
			return EXIT_FAILURE;
		}
	}

	// Set CUDA device
	check_cudaruntimecall(cudaSetDevice(gpuid));

	// Open cuFile driver
	status = cuFileDriverOpen();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFileDriverOpen error: "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	// Opens a file to write
	std::cout << "Opening file: " << testfile << std::endl;
	for (size_t i = 0; i < batch_size; i++) {
		if (nonDirFlag == 0) {
			fds[i] = open(testfile.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
		} else {
			// Half of the files are opened with O_DIRECT, the other half are opened without it.
			if (i % 2 == 0) {
				fds[i] = open(testfile.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
			} else {
				fds[i] = open(testfile.c_str(), O_CREAT | O_RDWR, 0664);
			} 
		}

		if (fds[i] < 0) {
			std::cerr << "File open error: "
				<< cuFileGetErrorString(errno) << std::endl;
			ret = EXIT_FAILURE;
			goto close_files;
		}
	}

	// Register file handles
	for (size_t i = 0; i < batch_size; i++) {
		cf_descrs[i].handle.fd = fds[i];
		cf_descrs[i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&cf_handles[i], &cf_descrs[i]);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "File register error:"
				<< cuFileGetErrorString(status) << std::endl;
			ret = EXIT_FAILURE;
			goto deregister_filehandles;
		}
	}

	// Allocate device memory and fill with pattern
	for (size_t i = 0; i < batch_size; i++) {
		check_cudaruntimecall(cudaMalloc(&devPtrs[i], MAX_BUFFER_SIZE));

		// Fill device memory with pattern (specific per batch entry)
		check_cudaruntimecall(cudaMemset(static_cast<void*>(devPtrs[i]), 0xef + i, MAX_BUFFER_SIZE));
		check_cudaruntimecall(cudaStreamSynchronize(0));
	}

	// Registers device memory for each batch entry
	std::cout << "Registering device memory of size: " << MAX_BUFFER_SIZE << " for " << batch_size << " batch entries" << std::endl;
	for (size_t i = 0; i < batch_size; i++) {
		status = cuFileBufRegister(devPtrs[i], MAX_BUFFER_SIZE, 0);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer register failed: "
				<< cuFileGetErrorString(status) << std::endl;
			registered_buffers = i;
			ret = EXIT_FAILURE;
			goto deregister_bufferhandles;
		}
	}
	registered_buffers = batch_size;

	// Setup batch IO parameters
	for (size_t i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handles[i];
		io_batch_params[i].u.batch.devPtr_base = devPtrs[i];
		io_batch_params[i].u.batch.file_offset = i * MAX_BUFFER_SIZE;
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = MAX_BUFFER_SIZE;
		io_batch_params[i].opcode = CUFILE_READ;
	}
	
	std::cout << "Setting Up Batch" << std::endl;
	errorBatch = cuFileBatchIOSetUp(&batch_id, batch_size);
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr << "Error in setting Up Batch" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandles;
	}

	std::cout << "Submitting Batch IO" << std::endl;
	for (size_t i = 0; i < batch_size; i++) {
		std::cout << "Submitting IO: index " << i << ", size: " << MAX_BUFFER_SIZE 
			<< ", file_offset: " << i * MAX_BUFFER_SIZE << ", devPtr: " 
			<< devPtrs[i] << std::endl;
	}
	errorBatch = cuFileBatchIOSubmit(batch_id, batch_size, io_batch_params.data(), flags);	
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr << "Error in IO Batch Submit" << std::endl;
		cuFileBatchIODestroy(batch_id);
		ret = EXIT_FAILURE;
		goto deregister_bufferhandles;
	}

	std::cout << "Cancelling Batch IO" << std::endl;
	errorBatch = cuFileBatchIOCancel(batch_id);
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr << "Error in IO Batch Cancel" << std::endl;
		cuFileBatchIODestroy(batch_id);
		ret = EXIT_FAILURE;
		goto deregister_bufferhandles;
	}
	std::cout << "Batch IO Canceled" << std::endl;
	
	nr = batch_size;
	errorBatch = cuFileBatchIOGetStatus(batch_id, batch_size, &nr, io_batch_events.data(), NULL);	
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr << "Error in IO Batch Get Status" << std::endl;
		cuFileBatchIODestroy(batch_id);
		ret = EXIT_FAILURE;
		goto deregister_bufferhandles;
	}
	std::cout << "Got events: " << nr << std::endl;

	cuFileBatchIODestroy(batch_id);

// Cleanup labels
deregister_bufferhandles:
	for (size_t i = 0; i < registered_buffers; i++) {
		status = cuFileBufDeregister(devPtrs[i]);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer deregister failed: "
				<< cuFileGetErrorString(status) << std::endl;
			ret = EXIT_FAILURE;
		}
	}
deregister_filehandles:
	for (size_t i = 0; i < batch_size; i++)
		if (cf_handles[i]) cuFileHandleDeregister(cf_handles[i]);
close_files:
	for (size_t i = 0; i < batch_size; i++)
		if (fds[i] >= 0) close(fds[i]);

	// Free device memory
	for (size_t i = 0; i < batch_size; i++)
		if (devPtrs[i]) check_cudaruntimecall(cudaFree(devPtrs[i]));
	// Close cuFile driver
	status = cuFileDriverClose();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFileDriverClose failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
	}

	return ret;
}
