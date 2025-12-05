# cuFile API Samples

## Overview

In this directory, you will find sample programs which demonstrate usage of NVIDIA's cuFile APIs. Each sample is intended to showcase a different aspect of application development using cuFile APIs.

<strong>Note:</strong> The sample tests expect the data files to exist and be at least 128 MiB in size. The data files should have read/write permissions in GDS enabled mounts. Cleanup is handled using `goto` statements for simplicity. For idiomatic C++ resource management using RAII, refer to our C++ cuFile bindings.

## Layout

```
.
├── 1_cuFile_Basics
│   ├── bufregister_write.cc
│   ├── devmem_offset_write.cc
│   ├── driver_rw_integrity.cc
│   ├── error_and_properties.cc
│   ├── iterative_devmem_offset_rw.cc
│   ├── iterative_read.cc
│   ├── no_bufregister_write.cc
│   ├── no_odirect_write.cc
│   ├── runtime_rw_integrity.cc
│   └── various_mem_rw.cc
├── 2_cuFile_Multithreaded
│   ├── file_control_locks.cc
│   ├── highly_concurrent_halfreg.cc
│   ├── many_buffers_set_bar.cc
│   ├── no_bufregister.cc
│   ├── separate_handles.cc
│   ├── separate_handles_offset.cc
│   ├── shared_handles.cc
│   └── shared_handles_offset.cc
├── 3_cuFile_Batch
│   ├── batch_cancel.cc
│   ├── batch_get_status.cc
│   ├── batch_read.cc
│   ├── batch_various.cc
│   ├── batch_write.cc
│   └── batch_write_unaligned.cc
├── 4_cuFile_Async
│   ├── async_rw_batch_custom.cc
│   ├── async_rw_batch_register_custom.cc
│   ├── async_rw_custom.cc
│   └── async_rw_default.cc
├── 5_cuFile_MemMap
│   ├── memmap_preregister.cc
│   ├── memmap_preregister_iterative_resize.cc
│   ├── memmap_reregister.cc
│   └── memmap_thrust.cu
├── bindings
│   └── python
│       ├── sample_001.py
│       ├── sample_002.py
│       ├── sample_003.py
│       ├── sample_004.py
│       ├── sample_005.py
│       ├── sample_006.py
│       ├── sample_007.py
│       ├── sample_008.py
│       ├── sample_009.py
│       └── sample_010.py
├── common
│   ├── cufile_sample_memmap.cc
│   ├── cufile_sample_shasum.cc
│   └── cufile_sample_utils.cc
├── include
│   ├── cufile_sample_memmap.hpp
│   ├── cufile_sample_shasum.hpp
│   ├── cufile_sample_thrust.hpp
│   └── cufile_sample_utils.hpp
├── Makefile
├── Makefile.pip
├── README.md
├── requirements.txt
└── vectorAdd.cu
```

## Compilation

### Option 1: Using System-Installed CUDA

<strong>Note:</strong> Assuming the path to GDS package is `/usr/local/cuda-XX.X/gds` and from the top level directory of `samples/`:
```bash
export CUFILE_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib/
make
```

Individual files can also be compiled as follows:
```bash
make 3_cuFile_Batch/batch_various
```

### Option 2: Using pip-Installed CUDA Packages

If you have installed CUDA via pip, you can use the `Makefile.pip` instead. This is useful for containerized environments or when you don't have system-wide CUDA installation privileges.

**Prerequisites:**

1. **Install CUDA Driver** (if not already installed)

2. **Use the provided requirements.txt file** that contains all required CUDA 13.0 packages (included in this directory)

3. **Create and activate a Python virtual environment:**
```bash
python3 -m venv cuda-pip-env
source cuda-pip-env/bin/activate

# Handle pip installation (Python 3.7 requires special handling)
if [[ "$(python3 --version 2>&1)" =~ "Python 3.7" ]]; then
    GET_PYTHON_PIP="curl https://bootstrap.pypa.io/pip/3.7/get-pip.py --output get-pip.py"
else
    GET_PYTHON_PIP="curl https://bootstrap.pypa.io/get-pip.py --output get-pip.py"
fi

# Download and install/upgrade pip
$GET_PYTHON_PIP
python3 get-pip.py

# Install CUDA packages
pip install -r requirements.txt
```

4. **Create missing symlink for libcufile.so** (required for linking):
```bash
cd cuda-pip-env/lib/python$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/site-packages/nvidia/cu13/lib/
ln -s libcufile.so.0 libcufile.so
cd -  # Return to samples directory
```

**Build all samples:**
```bash
make -f Makefile.pip build
```

**Build individual samples:**
```bash
make -f Makefile.pip 3_cuFile_Batch/batch_various
```

**Verify pip CUDA paths:**
```bash
make -f Makefile.pip debug-paths
```

**Clean build artifacts:**
```bash
make -f Makefile.pip clean
```

<strong>Notes for pip-based builds:</strong>
- The `Makefile.pip` automatically detects the virtual environment and pip-installed CUDA packages
- Only dynamically-linked binaries are generated (no `_static` versions, as pip packages don't include static libraries)
- Ensure your Python virtual environment is activated before building
- The makefile looks for CUDA packages in `nvidia/cu13` (CUDA 13.x) within your Python site-packages
- The `libcufile.so` symlink is required because pip packages only provide versioned libraries (e.g., `libcufile.so.0`)

## Samples Usage

### Basics
The samples under `1_cuFile_Basics/` directory demonstrates cuFile's usage of synchronous I/O, buffer registration (and usage without it), data integrity checks while using either CUDA's Runtime or Driver APIs, handling of memory and file offsets, support for different memory types, and error handling along with property management in cuFile.

`bufregister_write.cc`: Sample file that incorporates cuFileBufRegister and cuFileWrite.
```
./bufregister_write <dir/testFile> <gpuid>
```

`no_bufregister_write.cc`: Sample file that uses cuFileWrite without explicit cuFileBufRegister.
```
./no_bufregister_write <dir/testFile> <gpuid>
```

`no_odirect_write.cc`: Sample file that performs cuFileWrite without the O_DIRECT flag.
```
./no_odirect_write <dir/testFile> <gpuid>
```

`runtime_rw_integrity.cc`: Sample file that performs data integrity test with cuFileRead and cuFileWrite using CUDA's runtime APIs.
```
./runtime_rw_integrity <dir/testWriteReadFile> <dir/testWriteFile> <gpuid>
```

`driver_rw_integrity.cc`: Sample file that performs data integrity test with cuFileRead and cuFileWrite using CUDA's driver APIs.
```
./driver_rw_integrity <dir/testWriteReadFile> <dir/testWriteFile> <gpuid>
```

`devmem_offset_write.cc`: Sample file that applies cuFileWrites using device memory offsets.
```
./devmem_offset_write <dir/testRandomFile> <dir/testFile> <gpuid>
```

`iterative_read.cc`: Sample file that iteratively performs cuFileReads.
```
./iterative_read <dir/testFile> <dir/testWriteFile> <gpuid>
```

`iterative_devmem_offset_rw.cc`: Sample file that uses both device memory and file offsets when applying cuFileReads.
```
./iterative_devmem_offset_rw <dir/testFile> <dir/testWriteFile> <gpuid>
```

`various_mem_rw.cc`: Sample file that performs data integrity test where cuFileRead and cuFileWrite use different memory formats.
```
./various_mem_rw <dir/testReadFile> <dir/testWriteFile> <gpuid> <mode(1:DeviceMemory, 2:ManagedMemory, 3:HostMemory)> 
```

`error_and_properties.cc`: Sample file that shows different types of errors and how to set/get properties from the cuFile library.
```
./error_and_properties
```

### Multithreaded
The samples under `2_cuFile_Multithreaded/` directory showcase the use of cuFile APIs in multithreaded environments. They include examples of threads operating with separate or shared file handles, with and without buffer registration, managing file and device memory offsets, and performing atomic file operations using file locks.

`separate_handles.cc`: Sample file demonstrating multithreaded usage of cuFile APIs, where each thread uses its own CUfileHandle_t.
```
./separate_handles <dir/testfile1> <dir/testfile2> <gpuid> 
```

`shared_handles.cc`: Sample file demonstrating multithreaded usage of cuFile APIs, where each thread shares the same CUfileHandle_t. NOTE: gpuid1 and gpuid2 can be the same GPU.
```
./shared_handles <dir/testfile> <gpuid1> <gpuid2>
```

`no_bufregister.cc`: Sample file showcasing multithreaded usage of cuFile APIs without cuFileBufRegister. NOTE: gpuid1 and gpuid2 can be the same GPU.
```
./no_bufregister <dir/testfile> <gpuid1> <gpuid2>
```

`many_buffers_set_bar.cc`: Sample multithreaded example that showcases cuFile APIs in an environment where each thread uses 64 buffers, and the maximum pinned memory mapped to GPU BAR space is set at initialization.
```
./many_buffers_set_bar <dir/testfile1> <dir/testfile2> <gpuid>
```

`highly_concurrent_halfreg.cc`: Sample multithreaded example showcasing cuFile APIs in a highly concurrent environment where half the threads use cuFileBufRegister. It also configures the maximum pinned memory mapped to GPU BAR space and sets the internal cache size.

```
./highly_concurrent_halfreg <dir/testfile1> <dir/testfile2> <gpuid> 
```

`separate_handles_offset.cc`: Sample multithreaded example where cuFileAPIs are applied to handle different file and device buffer offsets. In this case, each thread has it's own instance CUfileHandle_t.
```
./separate_handles_offset <dir/testfile> <gpuid>
```

`shared_handles_offset.cc`: Sample multithreaded example where cuFileAPIs are applied to handle different file and device buffer offsets. In this case, each thread is sharing a CUfileHandle_t.
```
./shared_handles_offset <dir/testfile> <gpuid> 
```

`file_control_locks.cc`: Sample multithreaded example where fcntl locks are utilized for unaligned writes to achieve atomic transactions. Note: This sample requires cuFile library version to be 11.6 and above.
```
./file_control_locks <dir/testfile> <gpuid>
```

### Batch
The samples under `3_cuFile_Batch/` directory demonstrate the usage of cuFile’s Batch I/O APIs. These examples cover batch I/Os with varying numbers of batch entries, support for unaligned I/O, cancellation of batch operations, querying different batch entries at a time, and handling mixed combinations of buffer types and file modes.

`batch_write.cc`: This sample shows the usage of cuFile Batch API for writes.
```
./batch_write <dir/testfile> <gpuid> <numbatchentries(1-128)> <nondirectflag(0-all_direct, 1-half_direct)
```

`batch_write_unaligned.cc`: This sample shows the usage of cuFile batch API for unaligned I/O with half registered buffers using cuFileBufRegister
```
./batch_write_unaligned <dir/testfile> <gpuid> <numbatchentries(1-128)> <nondirectflag(0-all_direct, 1-half_direct)>
```

`batch_read.cc`: This sample shows the usage of cuFile Batch API for reads.
```
./batch_read <dir/testfile> <gpuid> <numbatchentries(1-128)> <nondirectflag(0-all_direct, 1-half_direct)>
```

`batch_cancel.cc`: This sample shows the usage of cuFile Batch API to cancel I/O after submitting a batch read.
```
./batch_cancel <dir/testfile> <gpuid> <numbatchentries(1-128)> <nondirectflag(0-all_direct, 1-half_direct)>
```

`batch_get_status.cc`: This sample shows the usage of cuFile Batch API to perform cuFileBatchIOGetStatus where it doesn't wait for all batches to complete to begin reaping.
```
./batch_get_status <dir/testfile> <gpuid> <nondirectflag(0-all_direct, 1-half_direct)>
```

`batch_various.cc`: This samples shows the usage of cuFile Batch API with various combination of files opened in regular, O_DIRECT, unaligned I/O, unregistered buffers, registered buffers, GPU memory and system memory.
```
./batch_various <dir/testfile> <gpuid> <numbatchentries(1-128)> <Buf Register 0 - register all buffers, 1 - unregistered buffers> <nondirectflag(0-all_direct, 1-half_direct)> 
```

### Async
The samples in the `4_cuFile_Async/` directory showcase asynchronous I/O operations using the cuFile API. They include code with default and custom CUDA streams, as well as batch-mode asynchronous operations with and without buffer registration.

`async_rw_default.cc`: This sample showcases a data integrity test using cuFileReadAsync and cuFileWriteAsync with default streams.
```
./async_rw_default <dir/testWriteReadFile> <dir/testWriteFile> <gpuid> 
```

`async_rw_custom.cc`: This sample showcases a data integrity test using cuFileReadAsync and cuFileWriteAsync with custom streams.
```
./async_rw_custom <dir/testWriteReadFile> <dir/testWriteFile> <gpuid> 
```

`async_rw_batch_custom.cc`: This sample showcases a data integrity test where cuFileReadAsync and cuFileWriteAsync are used in a batch mode with custom streams.
```
./async_rw_batch_custom <dir/testWriteReadFile> <dir/testWriteFile> <gpuid>
```

`async_rw_batch_register_custom.cc`: This sample showcases a data integrity test where cuFileReadAsync and cuFileWriteAsync are used in a batch mode with custom streams that are registered with cuFile.
```
./async_rw_batch_register_custom <dir/testWriteReadFile> <dir/testWriteFile> <gpuid> 
```

### MemMap
The samples in the `5_cuFile_MemMap/` directory demonstrate cuFile usage integrated with cuMemMap allocations. They cover scenarios such as re-registering and pre-registering device memory before resizing, as well as usage alongside Thrust for GPU parallel algorithms.

`memmap_reregister.cc`: This sample shows the usage of cuFile API with simple cuMemMap allocations. In this example cuFile will re-register device memory after the resize operation.
```
./memmap_reregister <dir/testfileA> <dir/testfileB> 
```

`memmap_thrust.cu`: This sample shows the usage of cuFile API with simple cuMemMap allocations and Thrust.
```
./memmap_thrust <dir/testfile>
```

`memmap_preregister.cc`: This sample shows the usage of cuFile API with simple cuMemMap allocations. In this example cuFile will pre-emptively register expected device memory before the resize.
```
./memmap_preregister <dir/testfileA> <dir/testfileB>
```

`memmap_preregister_iterative_resize.cc`: This sample shows the usage of cuFile API with simple cuMemMap allocations. In this example cuFile will pre-emptively register expected device memory before the iterative resize of a region.
```
./memmap_preregister_iterative_resize <dir/testfileA> <dir/testfileB>
```

### Static Files
When using the standard `Makefile`, each sample has a corresponding `_static` version (e.g., `bufregister_write_static`). These binaries are functionally identical but are statically linked against the cuFile library.

<strong>Note:</strong> Static binaries are not available when building with `Makefile.pip` since pip-installed packages don't include static libraries.

## Include
Helper functions and common utilities across the samples are declared in the `include/` directory.

## Common
Helper functions and common utilities across the samples are defined in the `common/` directory.

## Bindings
Additional language bindings for these samples are provided under the `bindings/` directory.