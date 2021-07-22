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
 * This sample shows how two threads can share the same CUFileHandle;
 * this program open the file in main thread and then the same file
 * descriptor is shared by both the threads.
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

typedef struct cfg {
	int gpu;
	const char *filename; //Debug only
        CUfileHandle_t cfr_handle;
} cfg_t;

static void *thread_fn(void *data)
{
	cfg_t *cfg = (cfg_t *)data;
	void *gpubuffer;
	int i, nr_ios;
	loff_t offset = 0;
	
	cudaSetDevice(cfg->gpu);
	cudaCheckError();

	/*
	 * Each thread allocates 1 GB of GPU Memory;
	 * However, there is no cuFileBufRegister now.
	 */
	cudaMalloc(&gpubuffer, GB(1));
	cudaCheckError();

	nr_ios = GB(1)/MB(100);

	for (i = 0; i < nr_ios; i++) {
		int ret;

		/*
		 * We are filling up the entire 1GB of GPU memory by reading file
		 * in chunks of 100 MB. When we have to do such incremental reads
		 * at different GPU Buffer offsets, we should avoid registering
		 * buffers using cuFileBufRegister in the loop. 
		 *
		 *  Ex:
		 *
		 *  for (...) {
		 *  	cuFileBufRegister((char *)gpubuffer + offset, MB(100);
		 *  	cuFileRead(cfg->cfr_handle, (char *)gpubuffer + offset, MB(100), offset, 0);
		 *  	offset += MB(100);
		 *  	cuFileBufDeregister((char *)gpubuffer + offset);
		 *  }
		 *
		 * The above code snippet Registers and Deregisters buffer in a loop at different
		 * GPU Buffer offsets. This is a very expensive operation and should be avoided.
		 * If this is the use case, we can skip the cuFileBufRegister altogether and 
		 * invoke cuFileRead directly to avoid the overhead. GDS Will use internal caching
		 * to transfer the data to the GPU Buffer.
		 */
		ret = cuFileRead(cfg->cfr_handle, (char *)gpubuffer + offset, MB(100), offset, 0);
		if (ret < 0) {
			fprintf(stderr, "cuFile Read failed with ret=%d\n", ret);
			goto err;
		}

		offset += MB(100);
	}

	fprintf(stdout, "Read Success from file %s to GPU %d\n", cfg->filename, cfg->gpu);
err:
	cudaFree(gpubuffer);
	return NULL;
}

void help(void) {
	printf("\n./cufilesample_011 <file-path-1> <gpu-id> <gpu-id>\n");
}

int main(int argc, char **argv) {

	pthread_t thread1, thread2;
	const char *file1;
	int fd;
	int gpu1, gpu2;
	cfg_t cfg1, cfg2;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle;
	CUfileError_t status;

	if (argc < 4) {
		fprintf(stderr, "Invalid input.\n");
		help();
		exit(1);
	}

	file1 = argv[1];
	gpu1 = atoi(argv[2]);
	gpu2 = atoi(argv[3]);

        status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
		fprintf(stderr, "GDS Driver open failed :%s\n", CUFILE_ERRSTR(status.err));
		exit(1);
        }

	/*
	 * Open the file and get the CUfileHanldle.
	 * Use the same handle for both the threads.
	 */
	fd  = open(file1, O_RDWR | O_DIRECT);

	if (fd < 0) {
		fprintf(stderr, "Unable to open file %s fd %d\n",
				file1, fd);
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


	cfg1.gpu = gpu1;
	cfg1.cfr_handle = cfr_handle;
	cfg1.filename = file1;

	cfg2.gpu = gpu2;
	cfg2.cfr_handle = cfr_handle;
	cfg2.filename = file1;

	pthread_create(&thread1, NULL, &thread_fn, &cfg1);
	pthread_create(&thread2, NULL, &thread_fn, &cfg2);

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);

	close(fd);
	cuFileDriverClose();

	return 0;
}
