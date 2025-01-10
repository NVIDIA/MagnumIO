"""
 Copyright 2025 NVIDIA Corporation.  All rights reserved.
 
 Please refer to the NVIDIA end user license agreement (EULA) associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the EULA
 is strictly prohibited.
"""
"""
 Sample KvikIO read write data integrity test with write device offset.

 This test creates 128KB of random data and writes it to a file using standard IO
 After writing the data, it is read back using KvikIO into device memory. From 
 device memory, a portion is written back out to a separate file again using KvikIO
 and an offset in the write command. The corresponding portion of the data in the 
 original random file and the file written using KvikIO are then compared for
 data integrity. All file openings and closures are handled with context managers.
"""

import os
import sys
import cupy
import kvikio

# Constants
DATA_SIZE_BYTES = (128 * 1024) # 128 KB
DEVICE_OFFSET_BYTES = (64 * 1024) # 64 KB

def main(read_path, write_path):
    """
    Reads data from one file and writes a portion of it to another file using KvikIO.

    The write operation uses a device offset to write only the portion of the data
    following the offset in the buffer to the file.

    Args:
        read_path (str): The path to the file to read from.
        write_path (str): The path to the file to write to.
    """
    print("Creating random test data...")
    test_data = os.urandom(DATA_SIZE_BYTES)
    print("Writing random data using standard calls to file: " + read_path)
    with open(read_path, 'wb') as f:
        f.write(test_data)

    print("Create data vector on GPU to store data")
    buf = cupy.empty(DATA_SIZE_BYTES, dtype=cupy.uint8)

    print("Opening file for read: " + read_path)
    with kvikio.CuFile(read_path, "r") as fr:
        print("Read data to device memory from file: " + read_path)
        ret = fr.read(buf)
        print("Bytes read: " + str(ret))

    print("Opening file for write: " + write_path)
    with kvikio.CuFile(write_path, "w") as fw:
        print("Write data from device memory to separate file: " + write_path)
        ret = fw.raw_write(buf, DATA_SIZE_BYTES-DEVICE_OFFSET_BYTES, 0, DEVICE_OFFSET_BYTES)
        print("Bytes written: " + str(ret))

    print("Confirm written data in " + write_path + " matches corresponding data from " + read_path)
    with open(read_path, 'rb') as f1, open(write_path, 'rb') as f2:
        f1.seek(DEVICE_OFFSET_BYTES)
        f1_data = f1.read()
        f2_data = f2.read()
        if f1_data == f2_data:
            print("File contents match as expected")
        else:
            print("File contents do not match")
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: sample_005.py <file_path1> <file_path2>")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]
    main(path1, path2)