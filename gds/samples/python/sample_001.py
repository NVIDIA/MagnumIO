"""
 Copyright 2025 NVIDIA Corporation.  All rights reserved.
 
 Please refer to the NVIDIA end user license agreement (EULA) associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the EULA
 is strictly prohibited.
"""
"""
 Sample KvikIO write test.

 This test writes data from GPU memory to a file using write and raw_write.
 For verification, input data has a pattern.
 User can verify the output file-data after write using
 hexdump -C <filepath>
 00000000  ab ab ab ab ab ab ab ab  ab ab ab ab ab ab ab ab  |................|
 *
 00020000  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
 *
 00021000  ab ab ab ab ab ab ab ab  ab ab ab ab ab ab ab ab  |................|
 *
 00041000
"""

import sys
import cupy
import kvikio

# Constants
FILE_SIZE_BYTES   = (128 * 1024) # 128 KB
FILE_OFFSET_BYTES = (4 * 1024) # 4 KB

def main(path: str) -> None:
    """
    Writes data to a file using KvikIO.

    Args:
        path (str): The path to the file to write to.
    """
    
    print(f"Opening file: {path}")
    file_writer = kvikio.CuFile(path, "w")
    print(f"Creating data vector of size {FILE_SIZE_BYTES} bytes")
    buf = cupy.full((FILE_SIZE_BYTES,), int("ab", 16), dtype=cupy.uint8)
    print("Writing data vector to file")
    # write is a blocking call implemented by calling pwrite and waiting for 
    # the IO to complete before returning to the caller.
    # like pwrite, it uses an internal threadpool on top of the cufile library.
    # It supports host and device memory.
    ret = file_writer.write(buf)
    print(f"Bytes written: {ret}")

    print(f"Writing second data vector to file, offset by {FILE_OFFSET_BYTES} bytes")
    # raw_write is a blocking call like write. It is implemented at a lower 
    # level than write and does not include an internal threadpool on top of 
    # the cufile library. 
    ret = file_writer.raw_write(buf, FILE_SIZE_BYTES, FILE_SIZE_BYTES + FILE_OFFSET_BYTES)
    print(f"Bytes written: {ret}")

    print("Closing file")
    file_writer.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sample_001.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)