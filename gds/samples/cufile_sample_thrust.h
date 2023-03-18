#ifndef __CUFILE_SAMPLE_THRUST_H_
#define __CUFILE_SAMPLE_THRUST_H_

#include <cuda.h>
#include <vector>

// includes, Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static size_t round_up(size_t x, size_t y) {
    return ((x + y - 1) / y) * y;
}

template <typename T>
class cufile_thrust_vector 
{
    public:
        cufile_thrust_vector()
        {
            // Initialize
            CUfileError_t status;
            CUdevice cuDevice;
            checkCudaErrors(cuInit(0));
            cuDeviceGet(&cuDevice, 0);

            // Get number of devices
            int NUM_DEVICES;
            cudaGetDeviceCount(&NUM_DEVICES);

            // Get mapping and backing devices
            for (int i=0; i<NUM_DEVICES; i++) {
                cuDeviceGet(&cuDevice, i);
                mappingDevices.push_back(cuDevice);
                backingDevices.push_back(getBackingDevices(cuDevice));
            }

            // Initialize driver for GDS
            status = cuFileDriverOpen();
            if (status.err != CU_FILE_SUCCESS) {
                    std::cerr << "cufile driver open error: "
                            << cuFileGetErrorString(status) << std::endl;
                    exit(EXIT_FAILURE);
            }
        }
        
        // Allocate virtually contiguous memory backed on separate devices
        // Returns thrust device pointer to the allocated space
        thrust::device_ptr<T> cufile_thrust_device_pointer(size_t N)
        {
            size = N * sizeof(T);
            checkCudaErrors(simpleMallocMultiDeviceMmap(&devPtr, size, &allocationSize, size, backingDevices[0], mappingDevices));

            dev_ptr = thrust::device_pointer_cast((T*)devPtr);
            return dev_ptr;
        }

        CUresult
        simpleMallocMultiDeviceMmap(CUdeviceptr *dptr, size_t va_size, size_t *allocationSize, size_t size,
                const std::vector<CUdevice> &residentDevices, const std::vector<CUdevice> &mappingDevices,
                size_t align = 0)
        {
            CUresult status = CUDA_SUCCESS;
            size_t min_granularity = 0;
            size_t stripeSize;

            // Setup the properties common for all the chunks
            // The allocations will be device pinned memory.
            // This property structure describes the physical location where the memory will be allocated via cuMemCreate allong with additional properties
            // In this case, the allocation will be pinnded device memory local to a given device.
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.allocFlags.gpuDirectRDMACapable = 1;

            // Get the minimum granularity needed for the resident devices
            // (the max of the minimum granularity of each participating device)
            for (unsigned idx = 0; idx < residentDevices.size(); idx++) {
                size_t granularity = 0;

                //get the minnimum granularity for residentDevices[idx]
                prop.location.id = residentDevices[idx];
                status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
                if (status != CUDA_SUCCESS) {
                    goto done;
                }
            std::cout <<"resident device idx:" << idx << ",gpuid:" << prop.location.id << ",granularity " << granularity << std::endl;
                if (min_granularity < granularity) {
                    min_granularity = granularity;
                }
            }

            // Get the minimum granularity needed for the accessing devices
            // (the max of the minimum granularity of each participating device)
            for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
                size_t granularity = 0;

                //get the minnimum granularity for mappingDevices[idx]
                prop.location.id = mappingDevices[idx];
                status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
                if (status != CUDA_SUCCESS) {
                    goto done;
                }
                std::cout <<"mapping device idx:" << idx << ",gpuid:" << prop.location.id << ",granularity " << granularity << std::endl;
                if (min_granularity < granularity) {
                    min_granularity = granularity;
                }
            }

            // Round up the size such that we can evenly split it into a stripe size tha meets the granularity requirements
            // Essentially size = N * residentDevices.size() * min_granularity is the requirement,
            // since each piece of the allocation will be stripeSize = N * min_granularity
            // and the min_granularity requirement applies to each stripeSize piece of the allocation.
            size = round_up(size, residentDevices.size() * min_granularity);
            stripeSize = size / residentDevices.size();
            // stripesize = stripeSize;
            std::cout << "actual allocation size:" << size  << ",min granularity" << min_granularity << ",total devs:" << residentDevices.size() << std::endl;

            // Return the rounded up size to the caller for use in the free
            if (allocationSize) {
                *allocationSize = size;
            }

            // Reserve the required contiguous VA space for the allocations
            status = cuMemAddressReserve(dptr, va_size, align, 0, 0);
            if (status != CUDA_SUCCESS) {
            std::cout << "Failed to reserve va addr:" << std::hex << (uint64_t)dptr << std::dec << ", size:" << va_size << std::endl;
                goto done;
            }

            // Create and map the backings on each gpu
            // note: reusing CUmemAllocationProp prop from earlier with prop.type & prop.location.type already specified.
            for (size_t idx = 0; idx < residentDevices.size(); idx++) {
                CUresult status2 = CUDA_SUCCESS;

                // Set the location for this chunk to this device
                prop.location.id = residentDevices[idx];

                // Create the allocation as a pinned allocation on this device
                CUmemGenericAllocationHandle allocationHandle;
                status = cuMemCreate(&allocationHandle, stripeSize, &prop, 0);
                if (status != CUDA_SUCCESS) {
                    goto done;
                }

                // Assign the chunk to the appropriate VA range and release the handle.
                // After mapping the memory, it can be referenced by virtual address.
                // Since we do not need to make any other mappings of this memory or export it,
                // we no longer need and can release the allocationHandle.
                // The allocation will be kept live until it is unmapped.
                status = cuMemMap(*dptr + (stripeSize * idx), stripeSize, 0, allocationHandle, 0);

                // the handle needs to be released even if the mapping failed.
                status2 = cuMemRelease(allocationHandle);
                if (status == CUDA_SUCCESS) {
                    // cuMemRelease should not have failed here
                    // as the handle was just allocated successfully
                    // however return an error if it does.
                    status = status2;
                }

                // Cleanup in case of any mapping failures.
                if (status != CUDA_SUCCESS) {
                    goto done;
                }
            }

            {
                // Each accessDescriptor will describe the mapping requirement for a single device
                std::vector<CUmemAccessDesc> accessDescriptors;
                accessDescriptors.resize(mappingDevices.size());

                // Prepare the access descriptor array indicating where and how the backings should be visible.
                for (size_t idx = 0; idx < mappingDevices.size(); idx++) {
                    // Specify which device we are adding mappings for.
                    accessDescriptors[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    accessDescriptors[idx].location.id = mappingDevices[idx];

                    // Specify both read and write access.
                    accessDescriptors[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                }

                // Apply the access descriptors to the whole VA range.
                status = cuMemSetAccess(*dptr, size, &accessDescriptors[0], accessDescriptors.size());
                if (status != CUDA_SUCCESS) {
                    goto done;
                }
            }

        done:
            if (status != CUDA_SUCCESS) {
                if (*dptr) {
                    simpleFreeMultiDeviceMmap(*dptr, size, va_size);
                }
            }

            return status;
        }

        std::vector<CUdevice> getBackingDevices(CUdevice cuDevice)
        {
            int num_devices;

            checkCudaErrors(cuDeviceGetCount(&num_devices));

            std::vector<CUdevice> backingDevices;
            backingDevices.push_back(cuDevice);
            for (int dev = 0; dev < num_devices; dev++)
            {
                int capable = 0;
                int attributeVal = 0;

                // The mapping device is already in the backingDevices vector
                if (dev == cuDevice) {
                    continue;
                }

                // Only peer capable devices can map each others memory
                checkCudaErrors(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
                if (!capable) {
                    continue;
                }

                // The device needs to support virtual address management for the required apis to work
                checkCudaErrors(cuDeviceGetAttribute(&attributeVal,
                                    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                                    cuDevice));
                if (attributeVal == 0) {
                    continue;
                }

                backingDevices.push_back(dev);
            }
            return backingDevices;
        }

        CUresult
        simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size, size_t va_size)
        {
            CUresult status = CUDA_SUCCESS;

            // Unmap the mapped virtual memory region
            // Since the handles to the mapped backing stores have already been released
            // by cuMemRelease, and these are the only/last mappings referencing them,
            // The backing stores will be freed.
            // Since the memory has been unmapped after this call, accessing the specified
            // va range will result in a fault (unitll it is remapped).
            std::cout << "unmap address range " << std::hex << (uint64_t)dptr << std::dec << ", size:" << size << std::endl;
            status = cuMemUnmap(dptr, size);
            if (status != CUDA_SUCCESS) {
                return status;
            }
            // Free the virtual address region.  This allows the virtual address region
            // to be reused by future cuMemAddressReserve calls.  This also allows the
            // virtual address region to be used by other allocation made through
            // opperating system calls like malloc & mmap.
            std::cout << "freeing address range " << std::hex << dptr << ", size:" << std::dec << va_size << std::endl;
            status = cuMemAddressFree(dptr, va_size);
            if (status != CUDA_SUCCESS) {
                return status;
            }

            return status;
        }

        // Returns raw pointer to the vector
        T* get_raw_pointer()
        {
            return thrust::raw_pointer_cast(dev_ptr);
        }

        // Write the vector to file using cuFile APIs
        void write_to_file(const char* file, off_t file_offset = 0, int index_offset = 0)
        {
            int fd=-1, ret;
            CUfileDescr_t cf_descr;
            CUfileHandle_t cf_handle;
            CUfileError_t status;
            off_t ptr_offset = index_offset*sizeof(T);

            ret = open(file, O_CREAT | O_RDWR | O_DIRECT, 0664);
            if (ret < 0) {
                    std::cerr << "file open error:"
                            << cuFileGetErrorString(errno) << std::endl;
                    exit(EXIT_FAILURE);
            }
            fd = ret;

            memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
            cf_descr.handle.fd = fd;
            cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            status = cuFileHandleRegister(&cf_handle, &cf_descr);
            if (status.err != CU_FILE_SUCCESS) {
                    std::cerr << "file register error:"
                            << cuFileGetErrorString(status) << std::endl;
                    close(fd);
                    exit(EXIT_FAILURE);
            }

            std::cout << "registering device memory of size :" << this->allocationSize << std::endl;
            // registers device memory
            status = cuFileBufRegister((void*)this->get_raw_pointer(), this->allocationSize, 0);
            if (status.err != CU_FILE_SUCCESS) {
                    ret = -1;
                    std::cerr << "buffer register A failed:"
                            << cuFileGetErrorString(status) << std::endl;
                    exit(EXIT_FAILURE);
            }

            cuStreamSynchronize(0);

            ret = cuFileWrite(cf_handle, (void*)this->get_raw_pointer(), this->allocationSize, file_offset, ptr_offset);
            if (ret < 0) {
                    if (IS_CUFILE_ERR(ret))
                            std::cerr << "write failed : "
                                    << cuFileGetErrorString(ret) << std::endl;
                    else
                            std::cerr << "write failed : "
                                    << cuFileGetErrorString(errno) << std::endl;
            } else {
                    std::cout << "written bytes :" << ret << std::endl;
                    ret = 0;
            }

            std::cout << "deregistering device memory" << std::endl;

            // deregister the device memory
            status = cuFileBufDeregister((void*)this->get_raw_pointer());
            if (status.err != CU_FILE_SUCCESS) {
                    std::cerr << "buffer deregister failed:"
                            << cuFileGetErrorString(status) << std::endl;
            }
        }

        // Copy host vector to the device vector
        void operator= (thrust::host_vector<T> h_vec)
        {
            checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)this->get_raw_pointer(), (const void *) thrust::raw_pointer_cast(&h_vec[0]), this->allocationSize));
        }

        // Assign value to the device vector
        void operator= (T value)
        {
            checkCudaErrors(cuMemsetD8((CUdeviceptr)this->get_raw_pointer(), value, this->allocationSize));
        }

        // Read from file to device vector using cuFile APIs
        void read_from_file(const char* file, off_t file_offset = 0, int index_offset = 0)
        {
            int fd=-1, ret;
            CUfileDescr_t cf_descr;
            CUfileHandle_t cf_handle;
            CUfileError_t status;
            off_t ptr_offset = index_offset*sizeof(T);

            ret = open(file, O_CREAT | O_RDWR | O_DIRECT, 0664);
            if (ret < 0) {
                    std::cerr << "file open error:"
                            << cuFileGetErrorString(errno) << std::endl;
                    exit(EXIT_FAILURE);
            }
            fd = ret;

            memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
            cf_descr.handle.fd = fd;
            cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            status = cuFileHandleRegister(&cf_handle, &cf_descr);
            if (status.err != CU_FILE_SUCCESS) {
                    std::cerr << "file register error:"
                            << cuFileGetErrorString(status) << std::endl;
                    close(fd);
                    exit(EXIT_FAILURE);
            }

            std::cout << "registering device memory of size :" << this->allocationSize << std::endl;
            // registers device memory
            status = cuFileBufRegister((void*)this->get_raw_pointer(), this->allocationSize, 0);
            if (status.err != CU_FILE_SUCCESS) {
                    ret = -1;
                    std::cerr << "buffer register failed:"
                            << cuFileGetErrorString(status) << std::endl;
                    exit(EXIT_FAILURE);
            }

            cuStreamSynchronize(0);

            ret = cuFileRead(cf_handle, (void*)this->get_raw_pointer(), this->allocationSize, file_offset, ptr_offset);
            if (ret < 0) {
                    if (IS_CUFILE_ERR(ret))
                            std::cerr << "read failed : "
                                    << cuFileGetErrorString(ret) << std::endl;
                    else
                            std::cerr << "read failed : "
                                    << cuFileGetErrorString(errno) << std::endl;
            } else {
                    std::cout << "read bytes:" << ret << std::endl;
                    ret = 0;
            }

            std::cout << "deregistering device memory" << std::endl;

            // deregister the device memory
            status = cuFileBufDeregister((void*)this->get_raw_pointer());
            if (status.err != CU_FILE_SUCCESS) {
                    std::cerr << "buffer deregister failed:"
                            << cuFileGetErrorString(status) << std::endl;
            }

        }

        ~cufile_thrust_vector()
        {
            std::cout << "freeing d_vec" << std::endl;
            checkCudaErrors(simpleFreeMultiDeviceMmap(devPtr, allocationSize, size));
        }

        CUdeviceptr devPtr;
        thrust::device_ptr<T> dev_ptr;
        size_t allocationSize;
        size_t size;
        std::vector<CUdevice> mappingDevices;
        std::vector<std::vector<CUdevice>> backingDevices;
};

__device__ long long int d_index = -1;

// kernel for concurrent implementation of thrust::find() algorithm
template<typename T>
__global__ void thrust_concurrent_find_kernel(T* begin, T value, size_t numElements, int d) {
    thrust::device_ptr<T> iter;
    iter = thrust::find(thrust::device, thrust::device_pointer_cast(begin), thrust::device_pointer_cast(begin+numElements-1), value);
    if (*iter == value) {
        d_index = thrust::distance(thrust::device_pointer_cast(begin), iter);
        d_index += d*numElements;
    }
}

// Peforms concurrent implementation of thrust::find() algorithm
// Returns index if value is found, else returns -1
template <typename T>
int thrust_concurrent_find(thrust::device_ptr<T> begin, thrust::device_ptr<T> end, T value)
{
    int NUM_DEVICES;
    cudaGetDeviceCount(&NUM_DEVICES);

    size_t num_elements = (thrust::distance(begin, end))/NUM_DEVICES;

    cudaStream_t streams[NUM_DEVICES];
    for(int i=0; i<NUM_DEVICES; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    for (int i=0; i<NUM_DEVICES; i++) {
        checkCudaErrors(cudaSetDevice(i));
        thrust_concurrent_find_kernel<T><<< 1, 1, 0, streams[i]>>>((T*) thrust::raw_pointer_cast(begin+i*num_elements), value, num_elements, i);
    }

    for (int i = 0; i < NUM_DEVICES; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamSynchronize(streams[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    int h_index[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaMemcpyFromSymbol(&h_index[i], d_index, sizeof(int));
        if (h_index[i] != -1) return h_index[i];
    }
    return -1;
}

#endif
