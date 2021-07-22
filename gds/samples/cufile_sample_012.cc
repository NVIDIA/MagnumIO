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
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

#include "cufile.h"

/*
 * This sample shows how two threads independently open the file and
 * have a sperate copy of CUfileHandle_t; also, it sets the cap on the max bar
 * size
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

static void *thread_fn(void *data)
{
	const int fd = *(int *)data;
	void *gpubuffer[64];
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle;
	int i, nr_ios;
	loff_t offset = 0;

	memset((void *)&cfr_descr, 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);
	if (status.err != CU_FILE_SUCCESS) {
		printf("file register error: %s\n", CUFILE_ERRSTR(status.err));
		close(fd);
		exit(1);
	}

	/* this should be set to same pci hierarchy of the peer device */
	cudaSetDevice(0);
	cudaCheckError();

	for (int j = 0; j < 64; j++) {
		cudaMalloc(&gpubuffer[j], MB(1));
		cudaCheckError();

		/*
		 * Each thread allocates GPU Memory
		 * and invoke cuFileBufRegister on the entire memory. This
		 * is done once; After registering the buffer, we read
		 * the data in chunks to the same buffer. We are using the GPU Buffer
		 * as a streaming buffer wherein we populate the data to the same
		 * buffer. This is optimal as data is directly DMA'ed to the registered
		 * buffer.
		 */
		status = cuFileBufRegister(gpubuffer[j], MB(1), 0);
		if (status.err != CU_FILE_SUCCESS) {
			printf("Buffer register failed :%s\n", CUFILE_ERRSTR(status.err));
			cudaFree(gpubuffer[j]);
			exit(1);
		}
	}

	nr_ios = GB(1)/MB(1);

	for (int j = 0; j < 64; j++) {
		offset = 0;
		for (i = 0; i < nr_ios; i++) {
			int ret;

			ret = cuFileRead(cfr_handle, gpubuffer[j], MB(1), offset, 0);
			if (ret < 0) {
				fprintf(stderr, "cuFile Read failed with ret=%d\n", ret);
				goto err;
			}

			offset += MB(1);
		}
	}

	fprintf(stdout, "Read Success from fd %d to GPU id 0\n", fd);
err:
	for (int j = 0; j < 64; j++) {
		status = cuFileBufDeregister(gpubuffer[j]);
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "Buffer Deregister failed :%s\n", CUFILE_ERRSTR(status.err));
		}

		cudaFree(gpubuffer[j]);
	}

	close(fd);
	return NULL;
}

void help(void) {
	printf("\n./cufilesample_012 <file-path-1> <file-path-2>\n");
}

int main(int argc, char **argv) {

	pthread_t thread1, thread2;
	CUfileError_t status;
	size_t size = 128 * 1024UL;

	if (argc < 3) {
		fprintf(stderr, "Invalid input.\n");
		help();
		exit(1);
	}

	status = cuFileDriverSetMaxPinnedMemSize(size);
    	if (status.err != CU_FILE_SUCCESS) {
		fprintf(stderr, "cuFileDriverSetMaxPinnedMemSize failed\n");
                exit(1);
        }

	int fd1  = open(argv[1], O_RDWR | O_DIRECT);

	if (fd1 < 0) {
		fprintf(stderr, "Unable to open file %s fd %d\n",
				argv[1], fd1);
		exit(1);
	}

	int fd2  = open(argv[2], O_RDWR | O_DIRECT);

	if (fd2 < 0) {
		fprintf(stderr, "Unable to open file %s fd %d\n",
				argv[2], fd2);
                close(fd1);
		exit(1);
	}


	pthread_create(&thread1, NULL, &thread_fn, (void *)&fd1);
	pthread_create(&thread2, NULL, &thread_fn, (void *)&fd2);

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);

	return 0;
}
