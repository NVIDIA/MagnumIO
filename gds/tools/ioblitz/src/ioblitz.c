/* -*- mode: c; c-basic-offset: 2; indent-tabs-mode: nil; -*-
 * vim:expandtab:shiftwidth=2:tabstop=2:
 */
/******************************************************************************\
*                                                                              *
*        Copyright (c) 2020, John J. Ravi                                      *
*      See the file COPYRIGHT for a complete copyright notice and license.     *
*                                                                              *
********************************************************************************
*
* Definitions and prototypes of abstract I/O interface
*
\******************************************************************************/

#define _GNU_SOURCE

#include "ioblitz.h"

static void *copy_thread_fn(void *data) {
  thread_data_t *td = (thread_data_t *)data;
  // int ret =
    real_memcpy((char *)td->dst+td->offset, (char *)td->src+td->offset, td->size);
  return NULL;
}

void * memcpy(void *destination, const void *source, size_t num) {
  void *return_value;

  // printf("john memcpy\n");
  char *io_blitz_size_threshold;
  if( (io_blitz_size_threshold = getenv("IO_BLITZ_SIZE_THRESHOLD")) &&
      (num >= atoi(io_blitz_size_threshold))) {
    char *io_blitz_threads;

    uint8_t num_worker = 1;
    if(io_blitz_threads = getenv("IO_BLITZ_THREADS")) {
      num_worker = atoi(io_blitz_threads);
    }

    pthread_t threads[num_worker];
    thread_data_t td[num_worker];

    size_t io_chunk = num / num_worker;
    size_t io_chunk_rem = num % num_worker;

    for (uint8_t ii = 0; ii < num_worker; ii++) {
      td[ii].dst = destination;
      td[ii].src = source;
      td[ii].size = io_chunk;
      td[ii].offset = (size_t)ii*io_chunk;

      if(ii == num_worker-1) {
        td[ii].size = (size_t)(io_chunk + io_chunk_rem);
      }
    }

    for (int ii = 0; ii < num_worker; ii++) {
      // ret =
        pthread_create(&threads[ii], NULL, &copy_thread_fn, &td[ii]);
    }

    for (int ii = 0; ii < num_worker; ii++) {
      pthread_join(threads[ii], NULL);
    }
  }
  else {
    return_value = real_memcpy(destination, source, num);
  }

  return return_value;
}

static void *cufile_write_thread_fn(void *data) {
  cufile_thread_data_t *td = (cufile_thread_data_t *)data;

  ssize_t ret =
   real_cuFileWrite(
     td->cfr_handle, 
     td->wr_devPtr, 
     td->size, 
     td->offset, 
     td->devPtr_offset
   );

  if (ret != td->size) {
    fprintf(stderr, "thread write failed!\n");
  }

  // printf("ret code: %ld\n", ret);

  return NULL;
}

ssize_t cuFileWrite(CUfileHandle_t fh, const void *devPtr_base, size_t size, off_t file_offset, off_t devPtr_offset) {
  // TODO: return size only if completed successfully
  ssize_t return_value = size;

  // printf("john cuFileWrite\n");

  char *io_blitz_size_threshold;
  if( (io_blitz_size_threshold = getenv("IO_BLITZ_SIZE_THRESHOLD")) &&
      (size >= atoi(io_blitz_size_threshold))) {
    char *io_blitz_threads;

    uint8_t num_worker = 1;
    if(io_blitz_threads = getenv("IO_BLITZ_THREADS")) {
      num_worker = atoi(io_blitz_threads);
    }

    pthread_t threads[num_worker];
    cufile_thread_data_t td[num_worker];

    size_t io_chunk = size / num_worker;
    size_t io_chunk_rem = size % num_worker;

    for (uint8_t ii = 0; ii < num_worker; ii++) {
      td[ii].cfr_handle = fh;
      td[ii].wr_devPtr = devPtr_base;
      td[ii].size = io_chunk;
      td[ii].offset = (size_t)ii*file_offset;
      td[ii].devPtr_offset = (size_t)ii*devPtr_offset;

      td[ii].offset = (size_t)(file_offset + ii*io_chunk);
      td[ii].devPtr_offset = (size_t)(devPtr_offset + ii*io_chunk);

      if(ii == num_worker-1) {
        td[ii].size = (size_t)(io_chunk + io_chunk_rem);
      }
    }

    for (int ii = 0; ii < num_worker; ii++) {
      // ret =
        pthread_create(&threads[ii], NULL, &cufile_write_thread_fn, &td[ii]);
    }

    for (int ii = 0; ii < num_worker; ii++) {
      pthread_join(threads[ii], NULL);
    }
  }
  else {
   return_value = real_cuFileWrite(fh, devPtr_base, size, file_offset, devPtr_offset);
  }

  return return_value;

}

