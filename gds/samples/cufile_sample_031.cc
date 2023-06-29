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
/*
 *  This is a simple data-integrity test for cuFileReadAsync/WriteAsync APIs using default stream.
 *  The test does the following:
 *  1. Creates a Test file with pattern
 *  2. Test file is loaded to device memory (cuFileReadAsync)
 *  3. From device memory, data is written to a new file (cuFileWriteAsync)
 *  4. Test file and new file are compared for data integrity
 *
 * e9d2f73120b2f2b1d2782e8ef5a42a3259b3c2badc5edb6ee04d4bc7b7633a
 * e9d2f73120b2f2b1d2782e8ef5a42a3259b3c2badc5edb6ee04d4bc7b7633a
 * SHA SUM Match
 * API Version :
 * 440-442(us) : 1
 * 510-512(us) : 1
 *
 */
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <openssl/sha.h>

#include <cuda_runtime.h>

// include this header
#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace std;

// copy bytes
#define MAX_BUF_SIZE (1024 * 1024UL)
typedef struct io_args_s
{
   void *devPtr;
   size_t max_size;
   off_t offset;
   off_t buf_off;
   ssize_t read_bytes_done;
   ssize_t write_bytes_done;
} io_args_t;

int main(int argc, char *argv[]) {
	int rfd = -1, wfd = -1;
        void *hostPtr = nullptr;
        ssize_t ret = -1;
        io_args_t args;
	CUfileError_t status;
        CUfileDescr_t cf_descr;
        CUfileHandle_t cf_rhandle = nullptr;
        CUfileHandle_t cf_whandle = nullptr;
	unsigned char iDigest[SHA256_DIGEST_LENGTH], oDigest[SHA256_DIGEST_LENGTH];
	Prng prng(255);
	const char *TEST_READWRITEFILE, *TEST_WRITEFILE;

        if(argc < 4) {
                std::cerr << argv[0] << " <readfilepath> <writefilepath> <gpuid> "<< std::endl;
                exit(1);
        }

        TEST_READWRITEFILE = argv[1];
        TEST_WRITEFILE = argv[2];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[3])));

	// Create TEST_READWRITEFILE file using standard Posix File IO calls
	wfd = open(TEST_READWRITEFILE, O_RDWR | O_CREAT, 0644);
	if (wfd < 0) {
		std::cerr << "test file open error : " << TEST_READWRITEFILE << " "
			<< std::strerror(errno) << std::endl;
		return -1;
	}

        memset(&args, 0, sizeof(args));
        args.max_size = MAX_BUF_SIZE;
        // Allocate all the arguments in the heap.
	hostPtr = malloc(args.max_size);
	if (!hostPtr) {
		std::cerr << "buffer allocation failure : "
			<< std::strerror(errno) << std::endl;
		ret = -1;
		goto out;
	}
	memset(hostPtr, prng.next_random_offset(), args.max_size);
	ret = write(wfd, hostPtr, args.max_size);
	if (ret < 0) {
		std::cerr << "write failure : " << std::strerror(errno)
				<< std::endl;
		goto out;
	}
        close(wfd);
        wfd = -1;
	free(hostPtr);
        hostPtr = nullptr;

        // Allocate device Memory and register with cuFile
	check_cudaruntimecall(cudaMalloc(&args.devPtr, args.max_size));
	// Register buffers. For unregistered buffers, this call is not required.
	status = cuFileBufRegister(args.devPtr, args.max_size, 0);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "buf register failed: "
                        << cuFileGetErrorString(status) << std::endl;
		ret = -1;
		goto out;
        }

	// Open TEST_READWRITEFILE in O_DIRECT and readonly mode
	rfd = open(TEST_READWRITEFILE, O_RDONLY | O_DIRECT);
	if (rfd < 0) {
		std::cerr << "read file open error : " << TEST_READWRITEFILE << " "
			<< std::strerror(errno) << std::endl;
		ret = rfd;
		goto out;
	}

	// Open TEST_WRITEFILE in O_DIRECT and readonly mode
	wfd = open(TEST_WRITEFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (wfd < 0) {
		std::cerr << "write file open error : " << std::strerror(errno)
				<< std::endl;
		ret = wfd;
		goto out;
	}

        // Register the filehandles
        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = rfd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_rhandle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = -1;
                goto out;
        }

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = wfd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_whandle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = -1;
		goto out;
        }
	// special case for holes
	check_cudaruntimecall(cudaMemsetAsync(args.devPtr, 0, args.max_size, 0));
	std::cout << "submit read to device memory :" << TEST_READWRITEFILE
                  << "max_size" << args.max_size
		  << std::endl;
	status = cuFileReadAsync(cf_rhandle, (unsigned char *)args.devPtr,
                                 &args.max_size, &args.offset, &args.buf_off,
                                 &args.read_bytes_done, 0);
	if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "read failed : "
				<< cuFileGetErrorString(status) << std::endl;
		ret = -1;
		goto out;
	}
	std::cout << "reading submit done to file :" << TEST_READWRITEFILE << std::endl;
	// Write loaded data from GPU memory to a new file

	std::cout << "submit write from dev memory to file :" << TEST_WRITEFILE
                  << "max_size" << args.max_size
                  <<std::endl;
	status = cuFileWriteAsync(cf_whandle, (unsigned char *)args.devPtr,
                                  (size_t *)&args.max_size, &args.offset, &args.buf_off,
			          &args.write_bytes_done, 0);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "write failed : "
				<< cuFileGetErrorString(status) << std::endl;
		goto out;
	}
	std::cout << "writing submit done to file :" << TEST_WRITEFILE << std::endl;

	check_cudaruntimecall(cudaStreamSynchronize(0));

        if((args.read_bytes_done < (ssize_t)args.max_size) ||
           (args.write_bytes_done < args.read_bytes_done))
        {
                std::cerr << "io error issued size:" << args.max_size <<
                          " read:" << args.read_bytes_done <<
                          " write:" <<  args.write_bytes_done << std::endl;
                goto out;
        }
	// Compare file signatures
	ret = SHASUM256(TEST_READWRITEFILE, iDigest, args.max_size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto out;
        }

	DumpSHASUM(iDigest);

	ret = SHASUM256(TEST_WRITEFILE, oDigest, args.max_size);
	if(ret < 0) {
                std::cerr << "SHASUM compute error" << std::endl;
                goto out;
        }

	DumpSHASUM(oDigest);

	if (memcmp(iDigest, oDigest, SHA256_DIGEST_LENGTH) != 0) {
		std::cerr << "SHA SUM Mismatch" << std::endl;
		ret = -1;
	} else {
		std::cout << "SHA SUM Match" << std::endl;
		ret = 0;
	}

out:
        if(hostPtr)
		free(hostPtr);
        if(args.devPtr) {
                cuFileBufDeregister(args.devPtr);
                check_cudaruntimecall(cudaFree(args.devPtr));
        }
        if(cf_rhandle)
                cuFileHandleDeregister(cf_rhandle);
        if(cf_whandle)
                cuFileHandleDeregister(cf_whandle);
	close(rfd);
	close(wfd);
	return ret;
}
