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
 * Sample cuFileBatchIOSubmit Read Test.
 *
 * This sample program reads data from GPU memory to a file using the Batch API's.
 * For verification, input data has a pattern.
 * User can verify the output file-data after write using
 * hexdump -C <filepath>
 * 00000000  ab ab ab ab ab ab ab ab  ab ab ab ab ab ab ab ab  |................|
 */
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>


#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

//include this header file
#include "cufile.h"

#include "cufile_sample_utils.h"

using namespace std;

#define MAX_BUFFER_SIZE 4096
#define MAX_BATCH_IOS 128

int main(int argc, char *argv[]) {
	int fd[MAX_BATCH_IOS];
	int nonDirFlag = 0;
	ssize_t ret = -1;
	void *devPtr[MAX_BATCH_IOS];
	const size_t size = MAX_BUFFER_SIZE;
	CUfileError_t status;
	const char *TESTFILE;
	CUfileDescr_t cf_descr[MAX_BATCH_IOS];
        CUfileHandle_t cf_handle[MAX_BATCH_IOS];
	unsigned int i = 0;
	CUfileIOParams_t io_batch_params[MAX_BATCH_IOS];
	CUfileIOEvents_t io_batch_events[MAX_BATCH_IOS];
	unsigned int flags = 0;
       	CUstream stream;
	CUfileError_t errorBatch;
	CUfileBatchHandle_t batch_id;
	unsigned nr;
	unsigned batch_size;
	unsigned num_completed = 0;

	if(argc < 4) {
                std::cerr << argv[0] << " <filepath> <gpuid> <num batch entries> <nondirectflag>"<< std::endl;
                exit(1);
        }
        memset(&stream, 0, sizeof(CUstream));
        TESTFILE = argv[1];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[2])));

        status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "cufile driver open error: "
			<< cuFileGetErrorString(status) << std::endl;
                return -1;
        }
	std::cout << "opening file " << TESTFILE << std::endl;
	
	batch_size = atoi(argv[3]);
	
	if(batch_size > MAX_BATCH_IOS) {
		std::cerr << "Requested batch Size exceeds maximum Batch Size limit:" << MAX_BATCH_IOS << std::endl;
		return -1;
	}

	// opens a file to write
	if (argc > 4)
		nonDirFlag = atoi(argv[4]);
	for(i = 0; i < batch_size; i++) {
		if (nonDirFlag == 0) {
			fd[i] = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
		} else {
			if (i % 2 == 0) {
				fd[i] = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
			} else {
				fd[i] = open(TESTFILE, O_CREAT | O_RDWR, 0664);
			}
		}
		if (fd[i] < 0) {
			std::cerr << "file open error:"
			<< cuFileGetErrorString(errno) << std::endl;
			goto out1;
		}
	}

	memset((void *)cf_descr, 0, MAX_BATCH_IOS * sizeof(CUfileDescr_t));
	for(i = 0; i < batch_size; i++) {
		cf_descr[i].handle.fd = fd[i];
		cf_descr[i].type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		status = cuFileHandleRegister(&cf_handle[i], &cf_descr[i]);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "file register error:"
				<< cuFileGetErrorString(status) << std::endl;
			close(fd[i]);
			fd[i] = -1;
			goto out1;
		}
	}

	for(i = 0; i < batch_size; i++) {
		devPtr[i] = NULL;
		check_cudaruntimecall(cudaMalloc(&devPtr[i], size));
		check_cudaruntimecall(cudaMemset((void*)(devPtr[i]), 0xab, size));
		check_cudaruntimecall(cudaStreamSynchronize(0));	
	}
	// filler

	std::cout << "registering device memory of size :" << size << std::endl;
	// registers device memory
	for(i = 0; i < batch_size; i++) {
		status = cuFileBufRegister(devPtr[i], size, 0);
		if (status.err != CU_FILE_SUCCESS) {
			ret = -1;
			std::cerr << "buffer register failed:"
				<< cuFileGetErrorString(status) << std::endl;
			goto out2;
		}
	}

	std::cout << "writing from device memory" << std::endl;


	for(i = 0; i < batch_size; i++) {
		io_batch_params[i].mode = CUFILE_BATCH;
		io_batch_params[i].fh = cf_handle[i];
		io_batch_params[i].u.batch.devPtr_base = devPtr[i];
		io_batch_params[i].u.batch.file_offset = i * size;
		io_batch_params[i].u.batch.devPtr_offset = 0;
		io_batch_params[i].u.batch.size = size;
		io_batch_params[i].opcode = CUFILE_READ;
	}

	std::cout << "Setting Up Batch" << std::endl;
	
	errorBatch = cuFileBatchIOSetUp(&batch_id, batch_size);
	if(errorBatch.err != 0) {
		std::cerr << "Error in setting Up Batch" << std::endl;
		goto out3;
	}
	std::cout << "Submitting Batch IO" << std::endl;
	
	errorBatch = cuFileBatchIOSubmit(batch_id, batch_size, io_batch_params, flags);	
	if(errorBatch.err != 0) {
		std::cerr << "Error in IO Batch Submit" << std::endl;
		goto out3;
	}
	std::cout << "Batch IO Submitted" << std::endl;
	
	while(num_completed != batch_size) {
		memset(io_batch_events, 0, sizeof(*io_batch_events));
		nr = batch_size;
		errorBatch = cuFileBatchIOGetStatus(batch_id, batch_size, &nr, io_batch_events, NULL);	
		if(errorBatch.err != 0) {
			std::cerr << "Error in IO Batch Get Status" << std::endl;
			goto out4;
		}
		std::cout << "Got events " << nr << std::endl;
		num_completed += nr;
		for(i = 0; i < nr; i++) {
			uint64_t buf[MAX_BUFFER_SIZE];
			cudaMemcpy(buf, io_batch_params[i].u.batch.devPtr_base, io_batch_events[i].ret, cudaMemcpyDeviceToHost);
			std::cout << "Completed  IO, index" << i << "size: " << io_batch_events[i].ret << std::endl;
		}
	}
	std::cout << "Batch IO Get status done got completetions for " << nr << " events" << std::endl;
out4:
	cuFileBatchIODestroy(batch_id);

	//Submit Batch IO
	std::cout << "deregistering device memory" << std::endl;
out3:
	// deregister the device memory
	for(i = 0; i < batch_size; i++) {
		status = cuFileBufDeregister(devPtr[i]);
		if (status.err != CU_FILE_SUCCESS) {
			ret = -1;
			std::cerr << "buffer deregister failed:"
				<< cuFileGetErrorString(status) << std::endl;
		}
	}
	std::cout << "cuFile BufDeregsiter Done" << std::endl;
out2:
	for(i = 0; i < batch_size; i++) {
		check_cudaruntimecall(cudaFree(devPtr[i]));
	}
	std::cout << "cudaFree Done" << std::endl;
out1:
	// close file
	for(i = 0; i < batch_size; i++) {
		if (fd[i] > 0) {
			cuFileHandleDeregister(cf_handle[i]);
			close(fd[i]);
		}
	}
	std::cout << "cuFileHandleDeregister Done" << std::endl;

	status = cuFileDriverClose();
	std::cout << "cuFileDriverClose Done" << std::endl;
	if (status.err != CU_FILE_SUCCESS) {
		ret = -1;
		std::cerr << "cufile driver close failed:"
			<< cuFileGetErrorString(status) << std::endl;
	}
	ret = 0;
	return ret;

}
