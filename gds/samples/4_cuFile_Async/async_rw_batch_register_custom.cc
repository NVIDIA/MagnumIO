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
 * This is a data-integrity test for cuFileReadAsync/WriteAsync APIs where
 * the async APIs are used in a batch mode. In this case, we register custom streams
 * created with cudaStreamCreateWithFlags with cuFile.
 * The test does the following:
 * 1. Creates a Test file with a random pattern
 * 2. Test file is loaded to device memory using cuFileReadAsync
 * 3. From device memory, data is written to a new file using cuFileWriteAsync
 * 4. Test file and new file are compared for data integrity using SHA-256
 *
 * ./async_rw_batch_register_custom <testWriteReadFile> <testWriteFile> <gpuid>
 *
 * | Output |
 * Using cuFileAsync APIs in a batch mode with registered custom streams
 * Register stream 0xbd3d160f2b80 with cuFile
 * Register stream 0xbd3d161143d0 with cuFile
 * Register stream 0xbd3d160f3b40 with cuFile
 * Register stream 0xbd3d163694a0 with cuFile
 * Batch read to device memory from file: <testWriteReadFile> of size: 1048576 for 4 entries
 * Batch read submit done to file: <testWriteReadFile>
 * Batch write from device memory to file: <testWriteFile> of size: 1048576 for 4 entries
 * Batch write submit done to file: <testWriteFile>
 * <iDigest>
 * <oDigest>
 * SHA256 SUM Match
 */
 
#include <iostream>
#include <array>
#include <string>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

// Include CUDA runtime header file
#include <cuda_runtime.h>
// Include cuFile header file
#include <cufile.h>

#include "../include/cufile_sample_utils.hpp"
#include "../include/cufile_sample_shasum.hpp"

static constexpr size_t MAX_BUF_SIZE = MiB(1);
static constexpr size_t MAX_BATCH_SIZE = 4;

// Buffer pointer offset is set at submission time 
static constexpr size_t CUSTOM_CU_FILE_STREAM_FIXED_BUF_OFFSET = 1;
// File offset is set at submission time 
static constexpr size_t CUSTOM_CU_FILE_STREAM_FIXED_FILE_OFFSET = 2;
// File size is set at submission time 
static constexpr size_t CUSTOM_CU_FILE_STREAM_FIXED_FILE_SIZE = 4;
// Size, offset and buffer offset are 4 KiB aligned 
static constexpr size_t CUSTOM_CU_FILE_STREAM_PAGE_ALIGNED_INPUTS = 8;

// Composite flag for cuFileStreamRegister
// (all flags set and it's value is 15)
static constexpr size_t CUSTOM_CU_FILE_STREAM_FIXED_AND_ALIGNED =
CUSTOM_CU_FILE_STREAM_FIXED_BUF_OFFSET | CUSTOM_CU_FILE_STREAM_FIXED_FILE_OFFSET |
CUSTOM_CU_FILE_STREAM_FIXED_FILE_SIZE | CUSTOM_CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;

struct AsyncIOArgs {
   void *devPtr;
   size_t max_size;
   off_t offset;
   off_t buf_off;
   ssize_t read_bytes_done;
   ssize_t write_bytes_done;
};

int main(int argc, char *argv[]) {
	int wfd = -1, rfd = -1;
	ssize_t ret = EXIT_SUCCESS;
	Prng prng(255);
	std::string testWriteReadFile, testWriteFile;
	std::array<unsigned char, SHA256_DIGEST_LENGTH> iDigest, oDigest;
	std::array<AsyncIOArgs, MAX_BATCH_SIZE> args;
	size_t registered_buffers = 0;

	// cuFile Specific Variables
	CUfileError_t status;
	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_rhandle = nullptr;
	CUfileHandle_t cf_whandle = nullptr;

	size_t total_size = MAX_BUF_SIZE * MAX_BATCH_SIZE;
	// io stream associated with the I/O
	std::array<cudaStream_t, MAX_BATCH_SIZE> io_stream;

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <testWriteReadFile> <testWriteFile> <gpuid> " << std::endl;
		return EXIT_FAILURE;
	}

	testWriteReadFile = argv[1];
	testWriteFile = argv[2];
	check_cudaruntimecall(cudaSetDevice(parseInt(argv[3])));

	std::cout << "Using cuFileAsync APIs in a batch mode with registered custom streams" << std::endl;

	// Initialize all AsyncIOArgs
	for (auto& arg : args) {
		arg.devPtr = nullptr;
		arg.max_size = MAX_BUF_SIZE;
		arg.offset = 0;
		arg.buf_off = 0;
		arg.read_bytes_done = 0;
		arg.write_bytes_done = 0;
	}

	// Initialize streams
	io_stream.fill(0);

	if (testWriteReadFile == testWriteFile) {
		std::cerr << "Test file and write file cannot be the same for verification" << std::endl;
		return EXIT_FAILURE;
	}

	// Create testWriteReadFile file using standard Posix File IO calls and fill it with random data
	if (!createTestFile(testWriteReadFile, total_size)) {
		std::cerr << "Failed to create test file: " << testWriteReadFile << std::endl;
		return EXIT_FAILURE;
	}

	// Allocate device Memory and register with cuFile
	for (unsigned i = 0; i < MAX_BATCH_SIZE; i++) {
		check_cudaruntimecall(cudaMalloc(&args[i].devPtr, args[i].max_size));

		// Register buffers. For unregistered buffers, this call is not required.
		status = cuFileBufRegister(args[i].devPtr, args[i].max_size, 0);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Buffer register failed: "
				<< cuFileGetErrorString(status) << std::endl;
			registered_buffers = i;
			ret = EXIT_FAILURE;
			goto deregister_bufferhandles;
		}

		// Set offset for each batch entry
		if (i > 0)
			args[i].offset += args[i - 1].offset + args[i].max_size;
		else
			args[i].offset = 0;
		
		// Create a stream for each of the batch entries and zero out in the case of holes. 
		// NOTE: One could create a single stream as well for all I/Os
		check_cudaruntimecall(cudaStreamCreateWithFlags(&io_stream[i], cudaStreamNonBlocking));
		check_cudaruntimecall(cudaMemsetAsync(args[i].devPtr, 0, args[i].max_size, io_stream[i]));
		std::cout << "Register stream " << io_stream[i] << " with cuFile" << std::endl;
		status = cuFileStreamRegister(io_stream[i], CUSTOM_CU_FILE_STREAM_FIXED_AND_ALIGNED);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Stream register failed: "
				<< cuFileGetErrorString(status) << std::endl;
			registered_buffers = i + 1;
			ret = EXIT_FAILURE;
			goto deregister_bufferhandles;
		}
	}
	registered_buffers = MAX_BATCH_SIZE;

	// Open testWriteReadFile in O_DIRECT and read-only mode
	rfd = open(testWriteReadFile.c_str(), O_RDONLY | O_DIRECT);
	if (rfd < 0) {
		std::cerr << "Read file open error for " << testWriteReadFile << ": "
			<< cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_bufferhandles;
	}

	// Open testWriteFile in O_DIRECT and write-only mode
	wfd = open(testWriteFile.c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0664);
	if (wfd < 0) {
		std::cerr << "Write file open error for " << testWriteFile << ": "
			<< cuFileGetErrorString(errno) << std::endl;
		ret = EXIT_FAILURE;
		goto close_files;
	}

	// Register the filehandles
	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = rfd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_rhandle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error for " << testWriteReadFile << ": "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto close_files;
	}

	std::memset(static_cast<void*>(&cf_descr), 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = wfd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_whandle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "File register error for " << testWriteFile << ": "
			<< cuFileGetErrorString(status) << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}

	std::cout << "Batch read to device memory from file: " << testWriteReadFile
		<< " of size: " << MAX_BUF_SIZE << " for " << MAX_BATCH_SIZE << " entries" << std::endl;
	for (size_t i = 0; i < MAX_BATCH_SIZE; i++) {
		status = cuFileReadAsync(
			cf_rhandle, static_cast<unsigned char*>(args[i].devPtr),
			&args[i].max_size, &args[i].offset, &args[i].buf_off,
			&args[i].read_bytes_done, io_stream[i]
		);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Read failed for batch entry " << i << ": "
				<< cuFileGetErrorString(status) << std::endl;
			ret = EXIT_FAILURE;
			goto deregister_filehandles;
		}
	}
	std::cout << "Batch read submit done to file: " << testWriteReadFile << std::endl;

	// Write loaded data from GPU memory to a new file
	std::cout << "Batch write from device memory to file: " << testWriteFile
		<< " of size: " << MAX_BUF_SIZE << " for " << MAX_BATCH_SIZE << " entries" << std::endl;

	for (size_t i = 0; i < MAX_BATCH_SIZE; i++) {
		status = cuFileWriteAsync(
			cf_whandle, static_cast<unsigned char*>(args[i].devPtr),
			&args[i].max_size, &args[i].offset, &args[i].buf_off,
			&args[i].write_bytes_done, io_stream[i]
		);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "Write failed for batch entry " << i << ": "
				<< cuFileGetErrorString(status) << std::endl;
			ret = EXIT_FAILURE;
			goto deregister_filehandles;
		}
	}
	std::cout << "Batch write submit done to file: " << testWriteFile << std::endl;

	// Synchronize streams and check for result
	for (size_t i = 0; i < MAX_BATCH_SIZE; i++) {
		check_cudaruntimecall(cudaStreamSynchronize(io_stream[i]));

		if ((args[i].read_bytes_done < static_cast<ssize_t>(args[i].max_size)) ||
			(args[i].write_bytes_done < args[i].read_bytes_done)) {
			std::cerr << "IO error issued size: " << args[i].max_size <<
				", read: " << args[i].read_bytes_done <<
				", write: " <<  args[i].write_bytes_done << std::endl;
			ret = EXIT_FAILURE;
			goto deregister_filehandles;
		}
	}

	// Compare file signatures
	if (SHASUM256(testWriteReadFile, iDigest, total_size) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}
	DumpSHASUM(iDigest);

	if (SHASUM256(testWriteFile, oDigest, total_size) < 0) {
		std::cerr << "SHA256 compute error" << std::endl;
		ret = EXIT_FAILURE;
		goto deregister_filehandles;
	}
	DumpSHASUM(oDigest);

	if (iDigest != oDigest) {
		std::cerr << "SHA256 SUM Mismatch" << std::endl;
		ret = EXIT_FAILURE;
	} else {
		std::cout << "SHA256 SUM Match" << std::endl;
		ret = EXIT_SUCCESS;
	}

// Cleanup labels
deregister_filehandles:
	if (cf_rhandle != nullptr) cuFileHandleDeregister(cf_rhandle);
	if (cf_whandle != nullptr) cuFileHandleDeregister(cf_whandle);
close_files:
	if (wfd >= 0) close(wfd);
	if (rfd >= 0) close(rfd);
deregister_bufferhandles:
	for (size_t i = 0; i < registered_buffers; i++) {
		if (args[i].devPtr != nullptr) {
			status = cuFileBufDeregister(args[i].devPtr);
			if (status.err != CU_FILE_SUCCESS) {
				std::cerr << "Buffer deregister failed: "
					<< cuFileGetErrorString(status) << std::endl;
				ret = EXIT_FAILURE;
			}
			check_cudaruntimecall(cudaFree(args[i].devPtr));
		}
		if (io_stream[i] != nullptr) {
			status = cuFileStreamDeregister(io_stream[i]);
			if (status.err != CU_FILE_SUCCESS) {
				std::cerr << "Stream deregister failed: "
					<< cuFileGetErrorString(status) << std::endl;
				ret = EXIT_FAILURE;
			}
			check_cudaruntimecall(cudaStreamDestroy(io_stream[i]));
		}
	}

	return ret;
}
