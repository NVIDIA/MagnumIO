/* -*- mode: c; c-basic-offset: 2; indent-tabs-mode: nil; -*-
 * vim:expandtab:shiftwidth=2:tabstop=2:
 */
/******************************************************************************\
*                                                                              *
*        Copyright (c) 2020, John J. Ravi                                      *
*      See the file LICENSE for a complete copyright notice and license.       *
*                                                                              *
\******************************************************************************/

#ifndef _IOBLITZ_H
#define _IOBLITZ_H

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cufile.h>

typedef struct thread_data_t {
  void *dst;
  const void *src;
  size_t size;
  size_t offset;
} thread_data_t;

typedef struct cufile_thread_data_t {
  union {
    void *rd_devPtr;            /* read device address */
    const void *wr_devPtr;      /* write device address */
  };
  CUfileHandle_t cfr_handle; /* cuFile Handle */
  off_t offset;              /* File offset */
  off_t devPtr_offset;       /* device address offset */
  size_t size;               /* Read/Write size */
} cufile_thread_data_t;

typedef void * (*real_memcpy_t)(void *, const void *, size_t);

typedef ssize_t (*real_open_t)(const char *, int, mode_t);
typedef ssize_t (*real_close_t)(int);

typedef ssize_t (*real_pread_t)(int, void *, size_t, off_t);
typedef ssize_t (*real_pwrite_t)(int, const void *, size_t, off_t);

typedef ssize_t (*real_cuFileWrite_t)(CUfileHandle_t, const void *, size_t, off_t, off_t);

void * real_memcpy(void *destination, const void *source, size_t num) {
  return ((real_memcpy_t)dlsym(RTLD_NEXT, "memcpy"))(destination, source, num);
}

int real_open(const char *pathname, int flags, mode_t mode) {
  return ((real_open_t)dlsym(RTLD_NEXT, "open"))(pathname, flags, mode);
}

int real_close(int fd) {
  return ((real_close_t)dlsym(RTLD_NEXT, "close"))(fd);
}

ssize_t real_pread(int fd, void *data, size_t size, off_t offset) {
  return ((real_pread_t)dlsym(RTLD_NEXT, "pread"))(fd, data, size, offset);
}

ssize_t real_pwrite(int fd, const void *data, size_t size, off_t offset) {
  return ((real_pwrite_t)dlsym(RTLD_NEXT, "pwrite"))(fd, data, size, offset);
}

ssize_t real_cuFileWrite(CUfileHandle_t fh, const void *devPtr_base, size_t size, off_t file_offset, off_t devPtr_offset) {
  return ((real_cuFileWrite_t)dlsym(RTLD_NEXT, "cuFileWrite"))(fh, devPtr_base, size, file_offset, devPtr_offset);
}


#endif /* not _IOBLITZ_H */
