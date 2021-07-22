## cuFile API Samples

Samples for CUDA Developers which demonstrates cuFile APIs in CUDA Toolkit. This version supports [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads). Each sample is intended to show different aspect of application development using cuFile APIs.

**Note**: The sample tests expect the data files to be present and atleast 128MiB in size.
      The data files should have read/write permissions in GDS enabled mounts.

## Prerequisites

Download and install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
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
export CUFILE_PATH=/usr/local/cuda-11.4/targets/x86_64-linux/lib/
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
./cufile_sample_015 <file-path-1> <file-path-2> <gpu-id> <mode>
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
