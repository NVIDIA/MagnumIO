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
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

#include "cufile.h"

/*
 * In this sample program, main thread will allocate 100 MB of GPU memory
 * The entire GPU memory will be registered using cuFileBufRegister in the main
 * thread. Each thread will just read the data at different offsets.
 */

#define TOGB(x) ((x)/(1024*1024*1024L))
#define GB(x) ((x)*1024*1024*1024L)
#define MB(x) ((x)*1024*1024L)
#define KB(x) ((x)*1024L)

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
}

typedef struct thread_data
{
	void *devPtr;
	loff_t offset;
	loff_t devPtr_offset;
	CUfileHandle_t cfr_handle;
}thread_data_t;

static void *thread_fn(void *data)
{
	int ret;	
	thread_data_t *t = (thread_data_t *)data;

	cudaSetDevice(0);
	cudaCheckError();


	for (int i = 0; i < 100; i++) {
		/*
		 * Note the usage of devPtr_offset. Every thread has same devPtr handle
		 * which was registered using cuFileBufRegister; however all threads are
		 * working at different buffer offsets. This is optimal as GPU memory is
		 * registered once and no internal caching is used
		 */
		ret = cuFileRead(t->cfr_handle, t->devPtr, MB(10), t->offset, t->devPtr_offset);
		if (ret < 0) {
			fprintf(stderr, "cuFileRead failed with ret=%d\n", ret);
		}

	}

	fprintf(stdout, "Read Success file-offset %ld readSize %ld to GPU 0 buffer offset %ld size %ld\n",
			(unsigned long) t->offset, MB(10), (unsigned long) t->offset, MB(10));

	return NULL;
}

void help(void) {
	printf("\n./cufilesample_017 <file-path-1>\n");
}

int main(int argc, char **argv) {

	void *devPtr;
	size_t offset = 0;
	int fd;
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle;
	thread_data t[10];
	pthread_t thread[10];

	if (argc < 2) {
		fprintf(stderr, "Invalid input.\n");
		help();
		exit(1);
	}

	fd  = open(argv[1], O_RDWR | O_DIRECT);

	if (fd < 0) {
		fprintf(stderr, "Unable to open file %s fd %d errno %d\n",
				argv[1], fd, errno);
		exit(1);
	}


	memset((void *)&cfr_descr, 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);
	if (status.err != CU_FILE_SUCCESS) {
		printf("file register error: %s\n", CUFILE_ERRSTR(status.err));
		close(fd);
		exit(1);
	}

	cudaSetDevice(0);
	cudaCheckError();

	cudaMalloc(&devPtr, MB(100));
	cudaCheckError();

	/*
	 * Entire Memory is registered
	 */
	status = cuFileBufRegister(devPtr, MB(100), 0);
	if (status.err != CU_FILE_SUCCESS) {
		printf("Buffer register failed :%s\n", CUFILE_ERRSTR(status.err));
		cuFileHandleDeregister(cfr_handle);
		close(fd);
		exit(1);
	}

	for (int i = 0; i < 10; i++) {
		/*
		 * Every thread will get same devPtr address; additionally, every thread
		 * will share the same cuFileHandle.
		 */
		t[i].devPtr = devPtr;
		t[i].cfr_handle = cfr_handle;

		/*
		 * Every thread will work on different offset
		 */
		t[i].offset = offset;
		t[i].devPtr_offset = offset;
		offset += MB(10);
	}


	for (int i = 0; i < 10; i++) {
		pthread_create(&thread[i], NULL, &thread_fn, &t[i]);
	}


	for (int i = 0; i < 10; i++) {
		pthread_join(thread[i], NULL);
	}

	status = cuFileBufDeregister(devPtr);
	if (status.err != CU_FILE_SUCCESS) {
		fprintf(stderr, "cuFileBufDeregister failed :%s\n", CUFILE_ERRSTR(status.err));
	}

	cuFileHandleDeregister(cfr_handle);
	close(fd);
	cudaFree(devPtr);
	return 0;
}
