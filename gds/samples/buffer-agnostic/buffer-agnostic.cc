/* buffer-agnostic.cc
 *
 * This sample demonstrates how software can detect where a memory buffer is allocated by querying the
 * CUDA runtime.
 *
 * Author: John J. Ravi <jjravi@lbl.gov>
 */

#include <fcntl.h>
#include <assert.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <errno.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda.h>

#ifdef GDS_SUPPORT
#include "cufile.h"
#endif

#define check_cudaruntimecall(apiFuncCall)                                    \
  {                                                                           \
    cudaError_t _status = apiFuncCall;                                        \
    if (_status != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: error: rt function %s failed with error %s.\n", \
        __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));       \
      exit(-1);                                                               \
    }                                                                         \
  }

using namespace std;

#define MAX_BUFFER_SIZE (128 * 1024UL) // 128 KB

bool is_device_pointer (const void *ptr) {
  struct cudaPointerAttributes attributes;
  cudaPointerGetAttributes (&attributes, ptr);
  return (attributes.devicePointer != NULL);
}

typedef struct file_desc_t {
  int fd; /* the unix file descriptor */
#ifdef GDS_SUPPORT
  CUfileHandle_t  cf_handle; /* cufile handle */
#endif
} file_desc_t;

void buffer_alloc(void **ptr, size_t size) {
  check_cudaruntimecall(cudaMalloc(ptr, size));

#ifdef GDS_SUPPORT
  CUfileError_t status;
  int ret;

  std::cout << "registering device memory of size :" << size << std::endl;
  // registers device memory
  status = cuFileBufRegister(*ptr, size, 0);
  if (status.err != CU_FILE_SUCCESS) {
    ret = -1;
    printf("buffer register error: %d\n", status.err);
    exit(EXIT_FAILURE);
  }
#else
  
#endif
}

void buffer_free(void *ptr) {
#ifdef GDS_SUPPORT
  std::cout << "deregistering device memory" << std::endl;
  CUfileError_t status;
  int ret;

  // deregister the device memory
  status = cuFileBufDeregister(ptr);
  if (status.err != CU_FILE_SUCCESS) {
    ret = -1;
    printf("buffer deregister error: %d\n", status.err);
    exit(EXIT_FAILURE);
  }
#endif

  check_cudaruntimecall(cudaFree(ptr));
}

ssize_t buffer_write(file_desc_t file_handle, const void *buf, size_t count, off_t offset) {
  ssize_t ret;
  if( is_device_pointer(buf) ) {
    std::cout << "is a device pointer" << std::endl;
    // writes device memory contents to a file
#ifdef GDS_SUPPORT
    std::cout << "writing from device memory" << std::endl;
    ret = cuFileWrite(file_handle.cf_handle, buf, count, 0, 0);
    if (ret < 0) {
      printf("write failed error: %ld\n", ret);
    } else {
      std::cout << "written bytes :" << ret << std::endl;
      ret = 0;
    }
#else
    void *hostPtr = NULL;
    hostPtr = malloc(count);

    std::cout << "buffering on host memory" << std::endl;
    check_cudaruntimecall(cudaMemcpy(hostPtr, buf, count, cudaMemcpyDeviceToHost));

    ret = pwrite(file_handle.fd, hostPtr, count, 0);
    if (ret < 0) {
      std::cerr << "file write error:" << strerror(errno) << std::endl;
      exit(EXIT_FAILURE);
    }
    free(hostPtr);
#endif
  }
  else {
    std::cout << "is not a device pointer" << std::endl;
    std::cout << "writing from host memory" << std::endl;
    ret = pwrite(file_handle.fd, buf, count, offset);
  }

  return ret;
}

file_desc_t file_open(const char *filepath) {
  file_desc_t file_handle;
  ssize_t ret = -1;

  // opens a file to write
#ifdef GDS_SUPPORT
  ret = open(filepath, O_CREAT | O_RDWR | O_DIRECT, 0664);
#else
  ret = open(filepath, O_CREAT | O_RDWR, 0664);
#endif
  if (ret < 0) {
    std::cerr << "file open error:" << strerror(errno) << std::endl;
    exit(EXIT_FAILURE);
  }
  file_handle.fd = ret;

#ifdef GDS_SUPPORT
  CUfileError_t status;
  CUfileDescr_t cf_descr;

  memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = file_handle.fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&file_handle.cf_handle, &cf_descr);

  if (status.err != CU_FILE_SUCCESS) {
    printf("file register error: %d\n", status.err);
    close(file_handle.fd);
    file_handle.fd = -1;
    exit(EXIT_FAILURE);
  }
#endif

  return file_handle;
}

int main(int argc, char *argv[]) {
  file_desc_t file_handle;
  ssize_t ret = -1;
  void *bufPtr = NULL;
  const size_t size = MAX_BUFFER_SIZE;
  const char *TESTFILE;
  bool use_gpu_buffer = false;

#ifdef GDS_SUPPORT
  CUfileError_t status;
#endif

  if(argc < 3) {
    std::cerr << argv[0] << " <filepath> <-1|cpu, x|gpuid> "<< std::endl;
    exit(1);
  }

  TESTFILE = argv[1];

  if(atoi(argv[2]) >= 0) {
    check_cudaruntimecall(cudaSetDevice(atoi(argv[2])));
    use_gpu_buffer = true;
  }

#ifdef GDS_SUPPORT
  status = cuFileDriverOpen();
  if (status.err != CU_FILE_SUCCESS) {
    printf("cufile driver open error: %d\n", status.err);
    return EXIT_FAILURE;
  }
#endif

  std::cout << "opening file " << TESTFILE << std::endl;
  file_handle = file_open(TESTFILE);

  std::cout << "allocating memory buffer" << std::endl;
  if(use_gpu_buffer) {
    buffer_alloc(&bufPtr, size);
    // filler
    check_cudaruntimecall(cudaMemset(bufPtr, 0xab, size));
  }
  else {
    bufPtr = malloc(size);
    memset(bufPtr, 0xcd, size);
  }

  // writes device memory contents to a file
  buffer_write(file_handle, bufPtr, size, 0);

  std::cout << "freeing memory buffer" << std::endl;
  if(use_gpu_buffer) {
    buffer_free(bufPtr);
  }
  else {
    free(bufPtr);
  }

#ifdef GDS_SUPPORT
  // close file
  if (file_handle.fd > 0) {
    cuFileHandleDeregister(file_handle.cf_handle);
  }

  status = cuFileDriverClose();
  if (status.err != CU_FILE_SUCCESS) {
    printf("cufile driver close error: %d\n", status.err);
    exit(EXIT_FAILURE);
	}
#else
#endif

  std::cout << "closing file" << std::endl;
  if (file_handle.fd > 0) {
    close(file_handle.fd);
  }

	return ret;
}
