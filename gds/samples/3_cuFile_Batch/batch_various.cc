/*
 * Copyright 2020-2025 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

/*
 * Sample cuFileBatchIOSubmit Read/Write Test for various combination 
 * of files opened in regular, O_DIRECT, unaligned I/O, unregistered buffers, 
 * registered buffers, GPU memory and system memory. This sample cycles batch
 * entries with different kinds of memory (cudaMalloc, cudaMallocHost, malloc,
 * mmap) to files in a single batch. For verification, SHASUM256 is used to
 * iteratively compute the digest of the data in the file and the device memory
 * (copied over to host memory).
 *
 * ./batch_various <testfile> <gpuid> <numbatchentries(1-128)> 
 * <Buf Register 0 - register all buffers, 1 - unregistered buffers>
 * <nondirectflag(0-all_direct, 1-half_direct)>
 *
 * | Output |
 * Submitting Write
 * Opening file /mnt/nvme/samplefiles/testfile6
 * Setting Up Batch
 * Submitting Batch IO
 * Submitting Write IO: index 0, size: 4096, file_offset: 0, devPtr: 0xef9ef9e00000
 * ...
 * Batch IO Submitted
 * Got events 8
 * Completed Write IO: index 0, size: 4096, file_offset: 0, devPtr: 0xef9ef9e00000
 * ...
 * Batch IO Get status done, got completions for events: 8
 * <iDigest1>
 * <oDigest1>
 * SHA SUM Match
 * ...
 * Submitting Read
 * Opening file /mnt/nvme/samplefiles/testfile6
 * Setting Up Batch
 * Submitting Batch IO
 * Submitting Read IO: index 0, size: 4096, file_offset: 0, devPtr: 0xef9ef9e00000
 * ...
 * Batch IO Submitted
 * Got events 8
 * Completed Read IO: index 0, size: 4096, file_offset: 0, devPtr: 0xef9ef9e00000
 * ...
 * Batch IO Get status done, got completions for events: 8
 * <iDigest1>
 * <oDigest1>
 * SHA SUM Match
 * ...
 * memory cleanup done
 * cuFileHandleDeregister and close done
 * cuFileDriverClose done
 */

#include <fcntl.h>
#include <assert.h>
#include <unistd.h>


#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/mman.h>

// Include CUDA runtime header file
#include <cuda_runtime.h>
// Include cuFile header file
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

enum MemType {
	CUDA_MALLOC = 0,
	CUDA_MALLOC_HOST = 1,
	MALLOC = 2,
	MMAP = 3,
	MAX_MEM_TYPES = 4
};

enum IOType {
	READ = 0,
	WRITE = 1
};

static constexpr size_t MAX_BUFFER_SIZE = KiB(4);
static constexpr size_t MAX_BATCH_IOS = 128;

static void memoryCleanup(std::array<void*, MAX_BATCH_IOS>& devPtrs) {
	for (size_t i = 0; i < MAX_BATCH_IOS; i++) {
		if (devPtrs[i] == nullptr)
			continue;

		if (i % MemType::MAX_MEM_TYPES == MemType::CUDA_MALLOC)
			check_cudaruntimecall(cudaFree(devPtrs[i]));
		else if (i % MemType::MAX_MEM_TYPES == MemType::CUDA_MALLOC_HOST)
			check_cudaruntimecall(cudaFreeHost(devPtrs[i]));
		else if (i % MemType::MAX_MEM_TYPES == MemType::MALLOC)
			free(devPtrs[i]);
		else if (i % MemType::MAX_MEM_TYPES == MemType::MMAP)
			munmap(devPtrs[i], MAX_BUFFER_SIZE);
	}

	std::cout << "Memory cleanup done" << std::endl;
}

int submit_batch(int argc, char *argv[], IOType io_type) {
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;
	std::array<int, MAX_BATCH_IOS> fds;
	std::array<void*, MAX_BATCH_IOS> devPtrs;
	std::array<uint8_t, MAX_BUFFER_SIZE> to_compare;
	std::array<uint8_t, MAX_BUFFER_SIZE> fill_pattern;
	int nonDirFlag = 0, gpuid = 0;
	ssize_t ret = EXIT_SUCCESS;
	std::string testfile;
	unsigned int flags = 0;
	unsigned int nr = 0;
	unsigned int batch_size = 0;
	unsigned int num_completed = 0;
	int registerBuf = 0;
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
	to_compare.fill(0);
	fill_pattern.fill(0xab);
	cf_handles.fill(nullptr);
	cf_descrs.fill(CUfileDescr_t{});
	io_batch_params.fill(CUfileIOParams_t{});
	io_batch_events.fill(CUfileIOEvents_t{});

	if (argc < 6) {
		std::cerr << "Usage: " << argv[0] << " <testfile> <gpuid> <numbatchentries(1-128)> "
			<< "<Buf Register 0 - register all buffers, 1 - unregistered buffers> "
			<< "<nondirectflag(0-all_direct, 1-half_direct)> " << std::endl;
		return EXIT_FAILURE;
	}
	
	testfile = argv[1];
	gpuid = parseInt(argv[2]);
	batch_size = parseInt(argv[3]);
	registerBuf = parseInt(argv[4]);
	nonDirFlag = parseInt(argv[5]);

	if (batch_size > MAX_BATCH_IOS || batch_size < 1) {
		std::cerr << "Requested batch size is below 1 or exceeds maximum batch size limit: " << MAX_BATCH_IOS << std::endl;
		return EXIT_FAILURE;
	}

	if (registerBuf != 0 && registerBuf != 1) {
		std::cerr << "Invalid registerBuf: " << registerBuf << std::endl;
		return EXIT_FAILURE;
	}

	if (nonDirFlag != 0 && nonDirFlag != 1) {
		std::cerr << "Invalid nonDirFlag: " << nonDirFlag << std::endl;
		return EXIT_FAILURE;
	}

	if (io_type == IOType::WRITE) {
		std::cout << "Submitting Write" << std::endl;
	} else {
		std::cout << "Submitting Read" << std::endl;
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

	// Open file in any mode
	std::cout << "Opening file " << testfile << std::endl;
	for (size_t i = 0; i < batch_size; i++) {
		int fileFlags = 0;
		if (io_type == IOType::WRITE) {
			fileFlags = O_CREAT | O_RDWR | O_TRUNC;
		} else {
			fileFlags = O_RDWR;
		}

		if (nonDirFlag == 0) {
			fds[i] = open(testfile.c_str(), fileFlags | O_DIRECT, 0664);
		} else {
			// Half of the files are opened in O_DIRECT mode, half in non O_DIRECT mode
			if (i % 2 == 0) {
				fds[i] = open(testfile.c_str(), fileFlags | O_DIRECT, 0664);
			} else {
				fds[i] = open(testfile.c_str(), fileFlags, 0664);
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

	// Allocate device memory
	for (size_t i = 0; i < batch_size; i++) {
		devPtrs[i] = nullptr;
		// Have the buffers be allocated in different memory types via it's (index mod MemType::MAX_MEM_TYPES)
		if (i % MemType::MAX_MEM_TYPES == MemType::CUDA_MALLOC) {
			check_cudaruntimecall(cudaMalloc(&devPtrs[i], MAX_BUFFER_SIZE));
		} else if (i % MemType::MAX_MEM_TYPES == MemType::CUDA_MALLOC_HOST) {
			check_cudaruntimecall(cudaMallocHost(&devPtrs[i], MAX_BUFFER_SIZE));
		} else if (i % MemType::MAX_MEM_TYPES == MemType::MALLOC) {
			devPtrs[i] = malloc(MAX_BUFFER_SIZE);
		} else if (i % MemType::MAX_MEM_TYPES == MemType::MMAP) {
			devPtrs[i] = mmap(NULL, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0 );
		}

		// Fill the buffers with a pattern different based on the memory type
		if (i % MemType::MAX_MEM_TYPES < MemType::MALLOC) {
			if (io_type == IOType::WRITE) {
				check_cudaruntimecall(cudaMemset(static_cast<void*>(devPtrs[i]), 0xab + i, MAX_BUFFER_SIZE));
				check_cudaruntimecall(cudaStreamSynchronize(0));	
			} else {
				check_cudaruntimecall(cudaMemset(static_cast<void*>(devPtrs[i]), 0, MAX_BUFFER_SIZE));
				check_cudaruntimecall(cudaStreamSynchronize(0));	
			}
		} else {
			if (io_type == IOType::WRITE) {
				std::memset(static_cast<void*>(devPtrs[i]), 0xab + i, MAX_BUFFER_SIZE);
			} else {
				std::memset(static_cast<void*>(devPtrs[i]), 0, MAX_BUFFER_SIZE);
			}
		}
	}

	// Register device memory if specified by the user
	if (registerBuf) {
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
	}

	// Set up batch IO parameters
	for (size_t i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handles[i];
		io_batch_params[i].u.batch.devPtr_base = devPtrs[i];
		io_batch_params[i].u.batch.file_offset = i * MAX_BUFFER_SIZE;
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = MAX_BUFFER_SIZE;
		if (io_type == IOType::WRITE) {
			io_batch_params[i].opcode = CUFILE_WRITE;
		} else {
			io_batch_params[i].opcode = CUFILE_READ;
		}
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
		std::cout << "Submitting " << (io_type == IOType::READ ? "Read" : "Write")
			<< " IO: index " << i << ", size: " << MAX_BUFFER_SIZE 
			<< ", file_offset: " << i * MAX_BUFFER_SIZE << ", devPtr: " 
			<< devPtrs[i] << std::endl;
	}
	errorBatch = cuFileBatchIOSubmit(batch_id, batch_size, io_batch_params.data(), flags);	
	if (errorBatch.err != CU_FILE_SUCCESS) {
		std::cerr<< "Error in IO Batch Submit" << std::endl;
		cuFileBatchIODestroy(batch_id);
		ret = EXIT_FAILURE;
		goto deregister_bufferhandles;
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
			goto deregister_bufferhandles;
		}

		std::cout << "Got events " << nr << std::endl;
		num_completed += nr;
		for (size_t i = 0; i < nr; i++) {
			std::cout << "Completed " << (io_type == IOType::READ ? "Read" : "Write")
				<< " IO: index " << i << ", size: " << io_batch_params[i].u.batch.size
				<< ", file_offset: " << io_batch_params[i].u.batch.file_offset << ", devPtr: " 
				<< io_batch_params[i].u.batch.devPtr_base << std::endl;
		}
	}
	std::cout << "Batch IO Get status done, got completions for events: " << nr << std::endl;

	// Verify the data in the file and the device memory between WRITE and READ
	for (size_t i = 0; i < batch_size; i++) {
		if (i % MemType::MAX_MEM_TYPES < MemType::MALLOC) {
			check_cudaruntimecall(cudaMemcpy(to_compare.data(), static_cast<void*>(devPtrs[i]), MAX_BUFFER_SIZE, cudaMemcpyDeviceToHost));
		} else {
			std::memcpy(to_compare.data(), static_cast<void*>(devPtrs[i]), MAX_BUFFER_SIZE);
		}

		iDigest.fill(0);
		if (SHASUM256(testfile, iDigest, MAX_BUFFER_SIZE, i * MAX_BUFFER_SIZE) < 0) {
			std::cerr << "SHASUM compute error" << std::endl;
			ret = EXIT_FAILURE;
			break;
		}
		DumpSHASUM(iDigest);

		oDigest.fill(0);
		if (SHASUM256_MEM(OSMemoryType::HOST, reinterpret_cast<char*>(to_compare.data()), MAX_BUFFER_SIZE, oDigest, 0, MAX_BUFFER_SIZE) < 0) {
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

// Cleanup labels
deregister_bufferhandles:
	if (registerBuf) {
		for (size_t i = 0; i < registered_buffers; i++) {
			status = cuFileBufDeregister(devPtrs[i]);
			if (status.err != CU_FILE_SUCCESS) {
				std::cerr << "Buffer deregister failed: "
					<< cuFileGetErrorString(status) << std::endl;
				ret = EXIT_FAILURE;
			}
		}
	}
deregister_filehandles:
	for (size_t i = 0; i < batch_size; i++)
		if (cf_handles[i]) cuFileHandleDeregister(cf_handles[i]);
close_files:
	for (size_t i = 0; i < batch_size; i++)
		if (fds[i] >= 0) close(fds[i]);

	// Free device memory
	memoryCleanup(devPtrs);
	// Close cuFile driver
	status = cuFileDriverClose();
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "cuFileDriverClose failed: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
	}

	return ret;
}

int main(int argc, char *argv[]) {
	return submit_batch(argc, argv, IOType::WRITE) || submit_batch(argc, argv, IOType::READ);
}
