"""
 Copyright 2025 NVIDIA Corporation.  All rights reserved.
 
 Please refer to the NVIDIA end user license agreement (EULA) associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the EULA
 is strictly prohibited.
"""
"""
 Sample KvikIO read write data integrity test using context managers.

 This test creates 128KB of random data and writes it to a file using standard IO
 After writing the data, it is read back using KvikIO into device memory. From 
 device memory it is written back out to a separate file again using KvikIO. The
 original random file and the file written using KvikIO are then compared for
 data integrity. All file openings and closures are handled with context managers.
"""

import filecmp
import os
import sys
import cupy
import kvikio

# Constants
DATA_SIZE_BYTES = (128 * 1024) # 128 KB

def main(read_path: str, write_path: str) -> None:
    """
    Reads random data from one file and writes it to another file using KvikIO.
    The files are compared to confirm data integrity during the read and write.

    Args:
        read_path (str): The path to the file to read from.
        write_path (str): The path to the file to write to.
    """

    print("Creating random test data...")
    test_data = os.urandom(DATA_SIZE_BYTES)
    print(f"Writing random data using standard calls to file: {read_path}")
    with open(read_path, 'wb') as standard_file_writer:
        standard_file_writer.write(test_data)

    print("Create data vector on GPU to store data")
    buf = cupy.empty(DATA_SIZE_BYTES, dtype=cupy.uint8)

    print(f"Opening file for read: {read_path}")
    with kvikio.CuFile(read_path, "r") as file_reader:
        print(f"Read data to device memory from file: {read_path}")
        ret = file_reader.read(buf)
        if ret < 0:
            print(f"Error reading file: {ret}")
            return
        print(f"Bytes read: {ret}")

    print(f"Opening file for write: {write_path}")
    with kvikio.CuFile(write_path, "w") as file_writer:
        print(f"Write data from device memory to separate file: {write_path}")
        ret = file_writer.write(buf)
        if ret < 0:
            print(f"Error writing file: {ret}")
            return
        print(f"Bytes written: {ret}")

    print(f"Confirm written data in {write_path} matches data from {read_path}")
    if filecmp.cmp(read_path, write_path, shallow=False):
        print("File contents match")
    else:
        print("File contents do not match")
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: sample_002.py <file_path1> <file_path2>")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]
    main(path1, path2)