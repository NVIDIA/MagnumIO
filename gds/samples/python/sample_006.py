"""
 Copyright 2025 NVIDIA Corporation.  All rights reserved.
 
 Please refer to the NVIDIA end user license agreement (EULA) associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the EULA
 is strictly prohibited.
"""
"""
 Sample KvikIO iterative read test.

 This test creates 16MB of random data and writes it to a file using standard IO
 After writing the data, it is read back using KvikIO into device memory iteratively
 64kB at a time. From device memory it is written back out to a separate file again 
 using KvikIO. The original random file and the file written from device memory are 
 then compared for data integrity. All file openings and closures are handled with 
 context managers.
"""

import sys
import os
import kvikio
import cupy
import filecmp

DATA_SIZE = (16 * 1024 * 1024) # 16 MB
CHUNK_SIZE = (64 * 1024) # 64 KB

def main(read_path, write_path):
    print("Creating random test data...")
    test_data = os.urandom(DATA_SIZE)
    print("Writing random data using standard calls to file: " + read_path)
    with open(read_path, 'wb') as f:
        f = open(read_path, 'wb')
        f.write(test_data)

    print("Create data vector on GPU to store data")
    buf = cupy.empty(DATA_SIZE, dtype=cupy.uint8)

    print("Opening file for read: " + read_path)
    with kvikio.CuFile(read_path, "r") as fr:
        iterations = 0
        bytes_read = 0
        file_offset = 0
        device_offset = 0
        print("Read data to device memory from file in 64KB chunks: " + read_path)
        while bytes_read < DATA_SIZE:
            iterations += 1
            read_size = min(DATA_SIZE-bytes_read, CHUNK_SIZE)
            ret = fr.raw_read(buf, read_size, file_offset, device_offset)
            if ret < 0:
                print("Error during iteration, exiting")
                return
            bytes_read += ret
            file_offset += ret
            device_offset += ret
        print("Total bytes read: " + str(bytes_read))
        print("Iterations: " + str(iterations))

    print("Opening file for write: " + write_path)
    with kvikio.CuFile(write_path, "w") as fw:
        fw = kvikio.CuFile(write_path, "w")
        print("Write data from device memory to separate file: " + write_path)
        ret = fw.write(buf)
        print("Bytes written: " + str(ret))

    print("Confirm written data in " + write_path + " matches data from " + read_path)
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