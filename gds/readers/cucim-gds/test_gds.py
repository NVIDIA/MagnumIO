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
