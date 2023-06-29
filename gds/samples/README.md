## cuFile API Samples

Samples for CUDA Developers which demonstrates cuFile APIs in CUDA Toolkit. This version supports [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11.4 and above. Each sample is intended to show different aspect of application development using cuFile APIs.

**Note**: The sample tests expect the data files to be present and atleast 128MiB in size.
      The data files should have read/write permissions in GDS enabled mounts.

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [GDS Installation Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html).
**Note**: cuFile samples need a NVIDIA GPU with cuda compute capability 6 and above. 

## Getting the cuFile Samples

Using git clone the repository of CUDA Samples using the command below.
``` bash
git clone https://github.com/NVIDIA/MagnumIO.git
cd gds/samples
```

## Compilation

**Note**: Assuming the path to GDS package is /usr/local/cuda-11.4/gds
``` bash
export CUFILE_PATH=/usr/local/cuda/targets/x86_64-linux/lib/
make
```

## Usage

**cufile_sample_001**: Sample file write test with cuFileBufRegister and cuFileWrite
``` bash
./cufile_sample_001 <dir/file-path-1> <gpu-id>
```

**cufile_sample_002**: Sample file write test with cuFileWrite
``` bash
./cufile_sample_002 <file-path-1> <gpu-id>
```

**cufile_sample_003**: Sample file data integrity test with cuFileRead and cuFileWrite
``` bash
./cufile_sample_003 <file-path-1> <file-path-2> <gpu-id>
```

**cufile_sample_004**: Sample file data integrity test with cuFileRead and cuFileWrite using cuda driver APIs
``` bash
./cufile_sample_004 <file-path-1> <file-path-2> <gpu-id>
```

**cufile_sample_005**: Sample file write test by passing device memory offsets
``` bash
./cufile_sample_005 <file-path-1> <file-path-2> <gpu-id>
```

**cufile_sample_006**: Sample file read test iterating over a given size of the file.
``` bash
./cufile_sample_006 <file-path-1> <file-path-2> <gpu-id>
```

**cufile_sample_007**: Sample to show set/get properties
``` bash
./cufile_sample_007
```

**cufile_sample_008**: Sample to show types of error messages from the library
``` bash
./cufile_sample_008
```

**cufile_sample_009**: Sample multithreaded example with cuFileAPIs.
This sample shows how two threads work with per-thread CUfileHandle_t
``` bash
./cufilesample_009 <file-path-1> <file-path-2>
```

**cufile_sample_010**: Sample multithreaded example with cuFileAPIs.
This sample shows how two threads can share the same CUfileHandle_t.
**Note**: The gpu-id1 and gpu-id2 can be the same GPU.
``` bash
./cufilesample_010 <file-path-1> <gpu-id1> <gpu-id2>
```

**cufile_sample_011**: Sample multithreaded example with cuFileAPIs without using cuFileBufRegister.
**Note**: The gpu-id1 and gpu-id2 can be the same GPU.
``` bash
./cufilesample_011 <file-path-1> <gpu-id1> <gpu-id2>
```

**cufile_sample_012**: Sample multithreaded example with cuFileAPIs with usage of SetMaxBar1Size and SetMaxCacheSize APIs.
This sample uses cuFileBufRegister per thread.
``` bash
./cufilesample_012 <file-path-1> <file-path-2>
```

**cufile_sample_013**: Sample multithreaded example with cuFileAPIs with usage of SetMaxBar1Size and SetMaxCacheSize APIs
This sample uses cuFileBufRegister alternately per thread.
``` bash
./cufilesample_013 <file-path-1> <file-path-2>
```

**cufile_sample_014**: Sample to use a file using cuFileRead buffer offsets
``` bash
./cufilesample_014 <file-path-read> <file-path-write> <gpu-id>
```

**cufile_sample_015**: Sample file data integrity test with cuFileRead and cuFileWrite with Managed Memory
``` bash
./cufile_sample_015 <file-path-1> <file-path-2> <gpu-id> <mode>, where mode is the memory type
(DeviceMemory = 1, ManagedMemory = 2, HostMemory = 3)
``` 
**Note**: mode is the memory type (DeviceMemory = 1, ManagedMemory = 2, HostMemory = 3)

**cufile_sample_016**: Sample to test multiple threads reading data at different file offsets and
buffer offset of a memory allocated using single allocation but registered with cuFile at different
buffer offsets in each thread.
``` bash
./cufile_sample_016 <file-path>
```

**cufile_sample_017**: Sample to test multiple threads reading data at different file offsets and
buffer offsets of a memory allocated using single allocation and single buffer registered with cuFile in main thread. 
``` bash
./cufile_sample_017 <file-path>
```

**cufile_sample_018**: This sample shows the usage of fcntl locks with GDS for unaligned writes to achieve atomic transactions.
``` bash
./cufile_sample_018 <file-path>
```

**Note**: Following samples need cuFile library version 11.6 and above. 

**cufile_sample_019**: This sample shows the usage of cuFile Batch API for writes.
``` bash
./cufile_sample_019 <file-path> <gpuid> <num batch entries>
```

**cufile_sample_020**: This sample shows the usage of cuFile Batch API for reads.
``` bash
./cufile_sample_020 <file-path>  <gpuid> <num batch entries>
```

**cufile_sample_021**: This sample shows the usage of cuFile Batch API to cancel I/O after submitting a batch read.
``` bash
./cufile_sample_021 <file-path>  <gpuid> <num batch entries>
```

**cufile_sample_022**: This sample shows the usage of cuFile Batch API to perform cuFileBatchIOGetStatus after submitting a batch read. The non O_DIRECT mode works only with libcufile version 12.2 and above. In this sample, nondirectflag is not a mandatory option
``` bash
./cufile_sample_022 <file-path>  <gpuid> <nondirectflag>
```

**cufile_sample_023**: This sample shows the usage of cuFile API with simple cuMemMap allocations.
``` bash
./cufile_sample_023  <filepathA> <filepathB>
```

**cufile_sample_024**: This sample shows the usage of cuFile API with simple cuMemMap allocations and Thrust.
``` bash
./cufile_sample_024 <file-path> 
```

**cufile_sample_025**: This sample shows the usage of cuFile API with simple cuMemMap allocations with resize operation.
``` bash
./cufile_sample_025  <filepathA> <filepathB>
```

**cufile_sample_026**: This sample shows the usage of cuFile API with simple cuMemMap allocations with multiple resize operations.
``` bash
./cufile_sample_026  <filepathA> <filepathB>
```

**cufile_sample_027**: This sample shows cuFileBatchIOSubmit Write Test for unaligned I/O with a variation of files opened in O_DIRECT and non O_DIRECT mode. The non O_DIRECT mode works only with libcufile version 12.2 and above.
``` bash
./cufile_sample_027 <filepath> <gpuid> <num batch entries> <nondirectflag>
```

Note: Following samples work only with libcufile version 12.2 and above.

**cufile_sample_028**: This sample shows the simple usage of cuFileWrite API without O_DIRECT MODE. The non O_DIRECT mode works only with libcufile version 12.2 and above.
``` bash
./cufile_sample_028 <file-path> <gpuid>
```

**cufile_sample_029**: This sample shows usage of cuFileBatchIOSubmit API for writes with various combinations of files opened in regular mode, O_DIRECT mode,
                   unaligned I/O, half unregistered buffers and half registered buffers.
                   This sample has files opened with O_DIRECT and non O_DIRECT mode alternatively in the batch.
``` bash
./cufile_sample_029 <filepath> <gpuid> <num batch entries> <nondirectflag>
```

**cufile_sample_030**: This sample shows cuFileBatchIOSubmit Write Test for combination of unaligned I/O, unregistered buffers and registered buffers,
                   This sample has files opened with O_DIRECT and non O_DIRECT mode alternatively in the batch.
                   This sample cycles batch entries with different kinds of memory (cudaMalloc, cudaMallocHost, malloc, mmap) to files in a single batch.
``` bash
./cufile_sample_030 <filepath> <gpuid> <num batch entries> <Buf Register 0 - register all buffers, 1 - unregistered buffers> <nondirectflag>
```

**cufile_sample_031**: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs using default stream.
``` bash
./cufile_sample_031 <readfilepath> <writefilepath> <gpuid>
```

**cufile_sample_032**: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs.
``` bash
./cufile_sample_032 <readfilepath> <writefilepath> <gpuid>
```

**cufile_sample_033**: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs.
This shows how the async apis can be used in a batch mode.
``` bash
./cufile_sample_033 <readfilepath> <writefilepath> <gpuid>
```

**cufile_sample_034**: This is a data-integrity test using cuFileReadAsync/WriteAsync APIs with cufile stream registration.
This shows how the async apis can be used in a batch mode.
``` bash
./cufile_sample_034 <readfilepath> <writefilepath> <gpuid>
```
