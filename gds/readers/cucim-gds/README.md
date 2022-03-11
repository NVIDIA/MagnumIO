# cuCIM's GDS API

This repository provides a GDS implementation for reading files with cuCIM's GDS API.

With cuCIM, you can work with CuPy array or PyTorch's Tensor objects.

## GDS installation

[NVIDIA® GPUDirect® Storage (GDS)](https://developer.nvidia.com/gpudirect-storage) needs to be installed to use GDS feature (Since CUDA Toolkit 11.4, GDS client package has been available.)

File access APIs would still work without GDS but you won't see the speed up.
Please follow the [release note](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html) or the [installation guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#abstract) to install GDS in your host system.

- Note:: During the GDS prerequisite installation (step 3 of [the installation guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#install-prereqs>)), you would need MOFED (Mellanox OpenFabrics Enterprise Distribution) installed. MOFED is available at https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed.

## Prerequisites to execute example code

Make sure that you have CuPy and PyTorch installed in your Python environment.
(The below instruction assumes that you have CUDA Toolkit >= 11.4 installed)

### CuPy

See the instruction(https://cupy.dev/)

```bash
# For CUDA 11.4
python3 -m pip install cupy-cuda114

# For CUDA 11.5
python3 -m pip install cupy-cuda115
```

### PyTorch

See the [instruction](https://pytorch.org/)

```bash
python3 -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### cuCIM

```bash
python3 -m pip install --extra-index-url https://test.pypi.org/simple/ cucim==0.0.233
# Or, you can execute `python3 -m pip install cucim` since 2022 April.

# Install dependent libraries (You don't need to install below if you don't use cuCIM's scikit-image API)
python3 -m pip install scipy 'scikit-image<0.20.0'
```

## Usage

### Checking if GDS is enabled in the system

```bash
python3 -c 'import cucim; print(cucim.clara.filesystem.is_gds_available())'
# True
```

Even if the above command outputs `True`, make sure that GDS is not running in compatible mode.
If GDS is running in compatible mode, you may find `cufile.log` in the current folder with the following message

```bash
NOTICE  cufio-drv:693 running in compatible mode
```

Please also make sure that `NVMe` driver is `Supported` (if you are using NVMe SSD).

```bash
/usr/local/cuda/gds/tools/gdscheck -p

 GDS release version: 1.1.0.37
 nvidia_fs version:  2.6 libcufile version: 2.4
 ============
 ENVIRONMENT:
 ============
 =====================
 DRIVER CONFIGURATION:
 =====================
 NVMe               : Supported
 NVMeOF             : Unsupported
 SCSI               : Unsupported
 ScaleFlux CSD      : Unsupported
 NVMesh             : Unsupported
 DDN EXAScaler      : Unsupported
 IBM Spectrum Scale : Unsupported
 NFS                : Unsupported
 WekaFS             : Unsupported
 Userspace RDMA     : Unsupported
 --Mellanox PeerDirect : Enabled
 --rdma library        : Not Loaded (libcufile_rdma.so)
 --rdma devices        : Not configured
 --rdma_device_status  : Up: 0 Down: 0
 =====================
 CUFILE CONFIGURATION:
 =====================
 ...
 ==============
 PLATFORM INFO:
 ==============
 IOMMU: disabled
 Platform verification succeeded
```

### Getting Started

**test_gds.py**

```python
from cucim.clara.filesystem import CuFileDriver
import cucim.clara.filesystem as fs
import os
import cupy as cp
import torch

# Write a file with size 10 (in bytes)
with open("input.raw", "wb") as input_file:
    input_file.write(
        bytearray([101, 102, 103, 104, 105, 106, 107, 108, 109, 110]))

# Create a CuPy array with size 10 (in bytes)
cp_arr = cp.ones(10, dtype=cp.uint8)
# Create a PyTorch array with size 10 (in bytes)
cuda0 = torch.device('cuda:0')
torch_arr = torch.ones(10, dtype=torch.uint8, device=cuda0)

# Using CuFileDriver
# (Opening a file with O_DIRECT flag is required for GDS)
fno = os.open("input.raw", os.O_RDONLY | os.O_DIRECT)
with CuFileDriver(fno) as fd:
    # Read 8 bytes starting from file offset 0 into buffer offset 2
    read_count = fd.pread(cp_arr, 8, 0, 2)
    # Read 10 bytes starting from file offset 3
    read_count = fd.pread(torch_arr, 10, 3)
os.close(fno)

# Another way of opening file with cuFile
with fs.open("output.raw", "w") as fd:
    # Write 10 bytes from cp_array to file starting from offset 5
    write_count = fd.pwrite(cp_arr, 10, 5)

```

```bash
python3 test_gds.py
ls *.raw
# input.raw  output.raw
```

### Executing benchmark

The folder `cucim/benchmarks/gds` contains some benchmark code you can execute.

```bash
# Create test data at the current folder (creates data_X.X.blob files)
python3 cucim/benchmarks/gds/gen_testdata.py

# Benchmark sequential reads (creates gds_performance.csv)
python3 cucim/benchmarks/gds/gds_performance.py

# Benchmark random reads (creates gds_performance_random.csv)
python3 cucim/benchmarks/gds/gds_random_access_performance.py
```

### Jupyter Notebook

Please see [`cucim/notebooks/Accessing_File_with_GDS.ipynb`](https://nbviewer.org/github/rapidsai/cucim/blob/cucim_gds_reader/notebooks/Accessing_File_with_GDS.ipynb) file to see how to use cuFile (GDS) API in cuCIM.
