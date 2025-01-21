"""
 Copyright 2025 NVIDIA Corporation.  All rights reserved.
 
 Please refer to the NVIDIA end user license agreement (EULA) associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the EULA
 is strictly prohibited.
"""
"""
 Sample KvikIO device offset test.

 This samples reads data from a file of random data into device memory at an
 offset. A portion of this data in the buffer from a second offset is then 
 written back out to a separate file. The appropriate latter portion of the 
 original file and the newly written file are then compared for data integrity.
"""

import os
import sys
import cupy
import kvikio

# Constants
FILE_SIZE_BYTES   = (128 * 1024) # 128 KB
DEV_READ_OFFSET_BYTES = (16 * 1024) # 16KB
DEV_WRITE_OFFSET_BYTES   = (32 * 1024) # 32 KB

def main(read_path: str, write_path: str) -> None:
    """
    Reads data from one file to a device buffer at an offset. Writes data to 
    another file from a different device buffer offset using KvikIO.

    The read operation uses a device offset to read the data into an offset
    location within the device buffer. The write operation writes the portion 
    of the data following a second offset out to a different file.

    Args:
        read_path (str): The path to the file to read from.
        write_path (str): The path to the file to write to.
    """

    print("Creating random test data...")
    test_data = os.urandom(FILE_SIZE_BYTES)
    print(f"Writing random data using standard calls to file: {read_path}")
    with open(read_path, 'wb') as standard_file_writer:
        standard_file_writer.write(test_data)

    print("Create data vector on GPU to store data")
    buf = cupy.empty(FILE_SIZE_BYTES+DEV_READ_OFFSET_BYTES, dtype=cupy.uint8)

    print(f"Opening file for read: {read_path}")
    with kvikio.CuFile(read_path, 'r') as file_reader:
        print(f"Read data to an offset in device memory from file: {read_path}")
        ret = file_reader.raw_read(buf, FILE_SIZE_BYTES, 0, DEV_READ_OFFSET_BYTES)
        print(f"Bytes read: {ret}")

    # Open file for write
    print(f"Opening file for write: {write_path}")
    with kvikio.CuFile(write_path, 'w') as file_writer:
        print(f"Write data from separate offset in device memory to separate file: {write_path}")
        ret = file_writer.raw_write(buf, FILE_SIZE_BYTES+DEV_READ_OFFSET_BYTES-DEV_WRITE_OFFSET_BYTES, 0, DEV_WRITE_OFFSET_BYTES)
        print(f"Bytes written: {ret}")

    print(f"Confirm written data in {write_path} matches corresponding data from {read_path}")
    with open(read_path, 'rb') as f1, open(write_path, 'rb') as f2:
        f1.seek(DEV_WRITE_OFFSET_BYTES-DEV_READ_OFFSET_BYTES)
        f1_data = f1.read()
        f2_data = f2.read()
        if f1_data == f2_data:
            print("File contents match as expected")
        else:
            print("File contents do not match")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: sample_008.py <file_path1> <file_path2>")
        sys.exit(1)

    read_path = sys.argv[1]
    write_path = sys.argv[2]
    main(read_path, write_path)