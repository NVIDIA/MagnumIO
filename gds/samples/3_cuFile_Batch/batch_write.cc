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
 * Sample cuFileBatchIOSubmit Write Test.
 * This sample program writes data from GPU memory to a file using
 * the Batch API's. For verification, input data has a pattern that
 * increments by 1 for each batch entry.
 *
 * User can verify the output if testfile after writing using
 * hexdump -C <testfile>
 * 0000000 efef efef efef efef efef efef efef efef
 * *
 * 0001000 f0f0 f0f0 f0f0 f0f0 f0f0 f0f0 f0f0 f0f0
 * ...
 * 
 * ./batch_write <testfile> <gpuid> <numbatchentries(1-128)> <nondirectflag(0-all_direct, 1-half_direct)>
 * NOTE: nonDirFlag is optional, if not provided, it defaults to 0.
 *
 * | Output |
 * Opening file: <testfile>
 * Registering device memory of size: 4096 for <numbatchentries> batch entries
 * Setting Up Batch
 * Submitting Batch IO
 * Submitting IO: index 0, size: 4096, file_offset: 0, devPtr: 0xf6f219e00000
 * ...
 * Batch IO Submitted
 * Got events: 8
 * Completed IO: index 0, size: 4096, file_offset: 0, devPtr: 0xf6f219e00000
 * ...
 * Batch IO Get status done, got completions for events: <numbatchentries>
 * Destroying batch IO, deregistering device memory, cleaning up, and closing cuFile driver
 * cuFileBufDeregister done
 * cuFileHandleDeregister, close, cudaFree done
 * cuFileDriverClose done
 */

#include <iostream>
#include <array>
#include <sys/types.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstdlib>
#include <string>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"

constexpr size_t MAX_BUFFER_SIZE = KiB(4);
constexpr size_t MAX_BATCH_IOS = 128;

int main(int argc, char *argv[]) {
	std::array<int, MAX_BATCH_IOS> fds;
	std::array<void*, MAX_BATCH_IOS> devPtrs;
	int nonDirFlag = 0, gpuid = 0;
	ssize_t ret = EXIT_SUCCESS;
	std::string testfile;
	unsigned int flags = 0;
	unsigned int nr = 0;
	unsigned int batch_size = 0;
	unsigned int num_completed = 0;
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
			std::cerr << "file register error:"
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
			goto deregister_buffershandles;
		}
	}
	registered_buffers = batch_size;

	// Set up batch IO parameters
	for (size_t i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handles[i];
		io_batch_params[i].u.batch.devPtr_base = devPtrs[i];
		io_batch_params[i].u.batch.file_offset = i * MAX_BUFFER_SIZE;
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = MAX_BUFFER_SIZE;
		io_batch_params[i].opcode = CUFILE_WRITE;
	}

	std::cout << "Setting Up Batch" << std::endl;
	errorBatch = cuFileBatchIOSetUp(&batch_id, batch_size);
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr << "Error in setting Up Batch" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_buffershandles;
	}

	std::cout << "Submitting Batch IO" << std::endl;
	for (size_t i = 0; i < batch_size; i++) {
		std::cout << "Submitting IO: index " << i << ", size: " << MAX_BUFFER_SIZE 
			<< ", file_offset: " << i * MAX_BUFFER_SIZE << ", devPtr: " 
			<< devPtrs[i] << std::endl;
	}
	errorBatch = cuFileBatchIOSubmit(batch_id, batch_size, io_batch_params.data(), flags);	
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr<< "Error in IO Batch Submit" << std::endl;
		cuFileBatchIODestroy(batch_id);
		ret = EXIT_FAILURE;
		goto deregister_buffershandles;
	}
	std::cout << "Batch IO Submitted" << std::endl;
	
	while (num_completed != batch_size) {
		io_batch_events.fill(CUfileIOEvents_t{});
		nr = batch_size;	
		errorBatch = cuFileBatchIOGetStatus(batch_id, batch_size, &nr, io_batch_events.data(), NULL);	
		if (errorBatch.err != CU_FILE_SUCCESS) {
			std::cerr << "Error in IO Batch Get Status" << std::endl;
			cuFileBatchIODestroy(batch_id);
			ret = EXIT_FAILURE;
			goto deregister_buffershandles;
		}

		std::cout << "Got events: " << nr << std::endl;
		num_completed += nr;
		for (unsigned j = 0; j < nr; j++) {
			std::cout << "Completed IO: index " << j << ", size: " << io_batch_params[j].u.batch.size
				<< ", file_offset: " << io_batch_params[j].u.batch.file_offset << ", devPtr: " 
				<< io_batch_params[j].u.batch.devPtr_base << std::endl;
		}
	}
	std::cout << "Batch IO Get status done, got completions for events: " << nr << std::endl;

	std::cout << "Destroying batch IO, deregistering device memory, cleaning up, and closing cuFile driver" << std::endl;
	cuFileBatchIODestroy(batch_id);

// Cleanup labels
deregister_buffershandles:
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
