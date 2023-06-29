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
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

#include "cufile.h"
#include "cufile_sample_utils.h"
/*
 This sample shows the usage of fcntl locks with GDS for unaligned writes to achieve atomic transactions.
*/

#define TOGB(x) ((x)/(1024*1024*1024L))
#define GB(x) ((x)*1024*1024*1024L)
#define MB(x) ((x)*1024*1024L)
#define KB(x) ((x)*1024L)
#define PAGE_SIZE 4096

#define ALIGN_UP(x, align_to)   (((x) + ((align_to)-1)) & ~((align_to)-1))
#define ALIGN_DOWN(x, a)        ((unsigned long)(x) & ~(((unsigned long)(a)) - 1))
#define MAX_RETRY 3

typedef struct thread_data
{
        void *devPtr; // device address
	int fd;
	CUfileHandle_t cfr_handle; //cuFile Handle
        loff_t offset; // File offset
	size_t size; // Read/Write size
} thread_data_t;

static void *read_thread_fn(void *data)
{
	int ret;
	thread_data_t *t = (thread_data_t *)data;
	int cnt;

	check_cudaruntimecall(cudaSetDevice(0));

			   /* l_type   l_whence  l_start  l_len    l_pid  */
	struct flock fl = { F_RDLCK, SEEK_SET,         0,       0,     0 };

	fl.l_pid = getpid();
	fl.l_type = F_RDLCK;
	fl.l_start = ALIGN_DOWN(t->offset, PAGE_SIZE);
	fl.l_len = ALIGN_UP(t->size, PAGE_SIZE);
	
	// Acquire lock at 4K boundary
        cnt = 0;
        while (1) {
                cnt++;
                if (fcntl(t->fd, F_SETLKW, &fl) == -1) {
                        printf("Failed to acquire read lock from offset %ld size %ld errno %d\n",
                                (unsigned long) fl.l_start, (unsigned long) fl.l_len, errno);
                        if (cnt == MAX_RETRY) {
                                exit(1);
                        } else {
                                printf("Retrying fcntl for read..\n");
                        }
                } else {
                        break;
                }
        }

        printf("Read lock acquired from offset %ld size %ld. Submit read at offset %ld size %ld\n",
                        (unsigned long) fl.l_start, (unsigned long) fl.l_len,
                        (unsigned long) t->offset, (unsigned long) t->size);

	ret = cuFileRead(t->cfr_handle, t->devPtr, t->size, t->offset, 0);
	if (ret < 0) {
		perror("cuFileRead Failed");
		printf("cuFileRead returned %d\n", ret);
		exit(1);
	}

	fl.l_type = F_UNLCK;  /* set to unlock same region */
	if (fcntl(t->fd, F_SETLKW, &fl) == -1) {
		perror("fcntl unlock failed");
		exit(1);
	}

        printf("Read success ret = %d at offset %ld size %ld\n", ret,
                       (unsigned long) t->offset, (unsigned long) t->size);

	return NULL;
}

static void *write_thread_fn(void *data)
{
	int ret;
	thread_data_t *t = (thread_data_t *)data;
	int cnt;

	/*
	 * We need to set the CUDA device; threads will not inherit main thread's
	 * CUDA context. In this case, since main thread allocated memory on GPU 0,
	 * we set it explicitly. However, threads have to ensure that they are in
	 * same cuda context as devPtr was allocated.
	 */
	check_cudaruntimecall(cudaSetDevice(0));

			   /* l_type   l_whence  l_start  l_len    l_pid  */
	struct flock fl = { F_WRLCK, SEEK_SET,         0,       0,     0 };

	fl.l_pid = getpid();
	fl.l_type = F_WRLCK;

	// Acquire lock at 4K boundary
	fl.l_start = ALIGN_DOWN(t->offset, PAGE_SIZE);
	fl.l_len = ALIGN_UP(t->size, PAGE_SIZE);
        cnt = 0;
        while (1) {
                cnt++;
                if (fcntl(t->fd, F_SETLKW, &fl) == -1) {
                        printf("Failed to acquire write lock from offset %ld size %ld errno %d\n",
                                (unsigned long) fl.l_start, (unsigned long) fl.l_len, errno);
                        if (cnt == MAX_RETRY) {
                                exit(1);
                        } else {
                                printf("Retrying fcntl for write..\n");
                        }
                } else {
                        break;
                }
        }
	
	printf("Write lock acquired from offset %ld size %ld. Submit write at offset %ld size %ld\n",
			(unsigned long) fl.l_start, (unsigned long) fl.l_len,
			(unsigned long) t->offset, (unsigned long) t->size);

	ret = cuFileWrite(t->cfr_handle, t->devPtr, t->size, t->offset, 0);
	if (ret < 0) {
		perror("cuFileWrite Failed");
		printf("cuFileWrite returned %d\n", ret);
		exit(1);
	}

	fl.l_type = F_UNLCK;  /* set to unlock same region */
	if (fcntl(t->fd, F_SETLKW, &fl) == -1) {
		perror("fcntl unlock failed");
		exit(1);
	}

	printf("Write success ret = %d at offset %ld size %ld\n", ret, 
		       (unsigned long) t->offset, (unsigned long) t->size);

	return NULL;
}

void help(void) {
	printf("\n./cufilesample_018 <file-path-1>\n");
}

int main(int argc, char **argv) {

	pthread_t write_thread1, write_thread2, read_thread3;
	CUfileError_t status;
       	void *devPtr;
    	int fd;
        CUfileDescr_t cfr_descr;
        CUfileHandle_t cfr_handle;
        thread_data t[3];

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

	check_cudaruntimecall(cudaSetDevice(0));
	check_cudaruntimecall(cudaMalloc(&devPtr, KB(4)));
	check_cudaruntimecall(cudaMemset(devPtr, 0xab, KB(4)));
	check_cudaruntimecall(cudaStreamSynchronize(0));
	
	// Thread 0 will write to file from offset 10 - write size 100 bytes
	// This is an unaligned write as offset is not 4K aligned. GDS will
	// convert this write to Read-Modify-Write
	t[0].fd = fd;
	t[0].devPtr = devPtr;
	t[0].cfr_handle = cfr_handle;
	t[0].offset = 10;
	t[0].size = 100;

	// Thread 1 will write to file from offset 50 - write size 200 bytes
	// This is an unaligned write as offset is not 4K aligned. GDS will
	// convert this write to Read-Modify-Write
	t[1].fd = fd;
	t[1].devPtr = devPtr;
	t[1].cfr_handle = cfr_handle;
	t[1].offset = 50;
	t[1].size = 200;

	// Thread 2 will read from file from offset 1000 - read size 100 bytes
	t[2].fd = fd;
	t[2].devPtr = devPtr;
	t[2].cfr_handle = cfr_handle;
	t[2].offset = 1000;
	t[2].size = 100;

	/*
	 * Thread 0 and Thread 1 are unaligned writes in a overlapping region.
	 * Thread 2 is a read but the range is not overlapping between writes.
	 *
	 * However, all three threads have a overlapping region between offset 0 and offset 4K.
	 * GDS does READ-MODIFY-WRITE on a 4K boundary. Hence, in the aforementioned case, 
	 * it is necessary for all three threads to acquire lock in 4k boundary even thorugh
	 * thread 2 doesn't have a direct overlap.
	 */
	pthread_create(&write_thread1, NULL, &write_thread_fn, &t[0]);
	pthread_create(&write_thread2, NULL, &write_thread_fn, &t[1]);
	pthread_create(&read_thread3, NULL, &read_thread_fn, &t[2]);

	pthread_join(write_thread1, NULL);
	pthread_join(write_thread2, NULL);
	pthread_join(read_thread3, NULL);

	cuFileHandleDeregister(cfr_handle);
	close(fd);
	check_cudaruntimecall(cudaFree(devPtr));

	return 0;
}
