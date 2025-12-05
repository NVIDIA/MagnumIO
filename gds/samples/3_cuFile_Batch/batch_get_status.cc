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
 * This samples submits reads in a batch, instead of reaping all the IOs.
 * It does a cuFileBatchIOGetStatus upto a max_nr value with a min_nr value as 0.
 * It keeps on calling cuFileBatchIOGetStatus with max_nr and min_nr until all
 * the IOs have finished. Verification is done by comparing the SHA256 of the
 * input data and the output data.
 *
 * ./batch_get_status <testfile> <gpuid> <nondirectflag(0-all_direct, 1-half_direct)>
 *
 * | Output |
 * Opening file: <testfile>
 * Registering device memory of size: 4096 for <numbatchentries> batch entries
 * Setting Up Batch
 * Submitting Batch IO
 * Submitting IO: index 0, size: 4096, file_offset: 0, devPtr: 0xebe2f9e00000
 * ...
 * Batch IO Submitted
 * Getting status of batch IO using range of min_nr and max_nr
 * Got events: <1-numbatchentries>
 * Got events: <1-numbatchentries>
 * ...
 * Batch IO Get status done got completions for <numbatchentries> events
 * <iDigest1>
 * <oDigest1>
 * SHA256 SUM Match
 * <iDigest2>
 * <oDigest2>
 * SHA256 SUM Match
 * ...
 * Destroying batch IO, deregistering device memory, cleaning up, and closing cuFile driver
 * cuFileBufDeregister done
 * cuFileHandleDeregister, close, cudaFree done
 * cuFileDriverClose done
 */

#include <fcntl.h>
#include <assert.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

// Include CUDA runtime header
#include <cuda_runtime.h>
// Include cuFile header
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t MAX_BUFFER_SIZE = KiB(4);
static constexpr size_t MAX_BATCH_IOS = 128;
static constexpr size_t MAX_NR = 16;

int main(int argc, char *argv[]) {
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;
	std::array<int, MAX_BATCH_IOS> fds;
	std::array<void*, MAX_BATCH_IOS> devPtrs;
	int nonDirFlag = 0, gpuid = 0;
	ssize_t ret = EXIT_SUCCESS;
	std::string testfile;
	unsigned int flags = 0;
	unsigned int batch_size = 0;
	unsigned int nr = MAX_NR;	// This is an in/out param to cuFileBatchIOGetStatus
	unsigned int min_nr = 0;
	unsigned int max_nr = 0;
	unsigned int entries_reaped = 0;
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

	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <testfile> <gpuid> "
			<< "<nondirectflag(0-all_direct, 1-half_direct)>" << std::endl;
		return EXIT_FAILURE;
	}

	// Parse command line arguments
	testfile = argv[1];
	gpuid = parseInt(argv[2]);

	// Parse optional command line arguments
	if (argc > 3) {
		nonDirFlag = parseInt(argv[3]);
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
		std::cerr << "cufile driver open error: "
			<< cuFileGetErrorString(status) << std::endl;
		return EXIT_FAILURE;
	}

	// Set batch size
	batch_size = MAX_BATCH_IOS;

	// Open file
	for (size_t i = 0; i < batch_size; i++) {
		if (nonDirFlag == 0) {
			fds[i] = open(testfile.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
		} else {
			if (i % 2 == 0) {
				fds[i] = open(testfile.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);
			} else {
				fds[i] = open(testfile.c_str(), O_CREAT | O_RDWR, 0664);
			}
		}
		if (fds[i] < 0) {
			std::cerr << "file open error:"
				<< cuFileGetErrorString(errno) << std::endl;
			ret = EXIT_FAILURE;
			goto close_files;
		}
	}

	// Register file handles for each file descriptor
	for (size_t i = 0; i < batch_size; i++) {
		cf_descrs[i].handle.fd = fds[i];
		cf_descrs[i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&cf_handles[i], &cf_descrs[i]);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "file register error: "
				<< cuFileGetErrorString(status) << std::endl;
			ret = EXIT_FAILURE;
			goto deregister_filehandles;
		}
	}

	// Allocate device memory for each file descriptor
	for (size_t i = 0; i < batch_size; i++) {
		devPtrs[i] = nullptr;
		check_cudaruntimecall(cudaMalloc(&devPtrs[i], MAX_BUFFER_SIZE));
		check_cudaruntimecall(cudaMemset(devPtrs[i], 0xab + i, MAX_BUFFER_SIZE));
		check_cudaruntimecall(cudaStreamSynchronize(0));	
	}

	// Register device memory for each file descriptor
	for (size_t i = 0; i < batch_size; i++) {
		status = cuFileBufRegister(devPtrs[i], MAX_BUFFER_SIZE, 0);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "buffer register failed:"
				<< cuFileGetErrorString(status) << std::endl;
			registered_buffers = i;
			ret = EXIT_FAILURE;
			goto deregister_bufferhandles;
		}
	}
	registered_buffers = batch_size;

	// Set up batch IO parameters for each file descriptor
	for (size_t i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handles[i];
		io_batch_params[i].u.batch.devPtr_base = devPtrs[i];
		io_batch_params[i].u.batch.file_offset = i * MAX_BUFFER_SIZE;
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = MAX_BUFFER_SIZE;
		io_batch_params[i].opcode = CUFILE_WRITE;
		io_batch_params[i].cookie = &io_batch_params[i];
	}
	
	// Set up batch IO with cuFile
	errorBatch = cuFileBatchIOSetUp(&batch_id, batch_size);
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr << "Error in setting Up Batch" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandles;
	}

	// Submit batch IO
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
	std::cout << "Batch IO Submitted" << std::endl;

	// Get status of batch IO using range of min_nr and max_nr
	min_nr = 0, max_nr = nr;
	std::cout << "Getting status of batch IO using range of min_nr and max_nr" << std::endl;
	while (entries_reaped != batch_size) {
		io_batch_events.fill(CUfileIOEvents_t{});
		// We are passing the timeout as NULL and hence we expect the number of entries returned to be
		// greater than min_nr and less than max_nr(nr).
		errorBatch = cuFileBatchIOGetStatus(batch_id, min_nr, &nr, io_batch_events.data(), NULL);
		if (errorBatch.err != CU_FILE_SUCCESS) {
			std::cerr << "Error in IO Batch Get Status" << std::endl;
			cuFileBatchIODestroy(batch_id);
			ret = EXIT_FAILURE;
			goto deregister_bufferhandles;
		}

		assert(nr <= max_nr);
		assert(nr >= min_nr);
		std::cout << "Got events: " << nr << std::endl;
		entries_reaped += nr;
		nr = max_nr;
	}
	std::cout << "Batch IO Get status done got completions for " << entries_reaped << " events" << std::endl;

	// Verify the data
	for (size_t i = 0; i < batch_size; i++) {
		iDigest.fill(0);
		if (SHASUM256(testfile, iDigest, MAX_BUFFER_SIZE, i * MAX_BUFFER_SIZE) < 0) {
			std::cerr << "SHASUM compute error" << std::endl;
			ret = EXIT_FAILURE;
			break;
		}
		DumpSHASUM(iDigest);

		oDigest.fill(0);
		if (SHASUM256_MEM(OSMemoryType::DEVICE, static_cast<char*>(devPtrs[i]), MAX_BUFFER_SIZE, oDigest, 0, MAX_BUFFER_SIZE) < 0) {
			std::cerr << "SHASUM Device mem compute error" << std::endl;
			ret = EXIT_FAILURE;
			break;
		}
		DumpSHASUM(oDigest);

		if (iDigest != oDigest) {
			std::cerr << "SHA SUM Mismatch" << std::endl;
			ret = EXIT_FAILURE;
			break;
		} else {
			std::cout << "SHA SUM Match" << std::endl;
			ret = EXIT_SUCCESS;
		}
	}

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
