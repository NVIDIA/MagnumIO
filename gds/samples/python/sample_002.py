"""
 Copyright 2025 NVIDIA Corporation.  All rights reserved.
 
 Please refer to the NVIDIA end user license agreement (EULA) associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the EULA
 is strictly prohibited.
"""
"""
 Sample KvikIO write test using context manager.

 This writes data from GPU memory to a file using write and raw_write.
 The opening and closing of the file is handled by a context manager.
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
import kvikio
import cupy

WRITE_SIZE = (128 * 1024) # 128 KB
WRITE_OFFSET = (4 * 1024) # 4 KB

def main(path):
    print("Creating data vector of size " + str(WRITE_SIZE) + " bytes")
    a = cupy.full((WRITE_SIZE,), int("ab", 16), dtype=cupy.uint8)
    print("Opening file: " + path)
    with kvikio.CuFile(path, "w") as f:
        print("Writing data vector to file")
        # write is a blocking call implemented by calling pwrite and waiting for the IO to complete before returning to the caller.
        # like pwrite, it uses an internal threadpool on top of the cufile library. It supports host and device memory.
        ret = f.write(a)
        print("Bytes written: " + str(ret))


        print("Writing second data vector to file, offset by 4KB")
        # raw_write is a blocking call like write. It is implemented at a lower level than write and does not include an internal
        # threadpool on top of the cufile library. 
        ret = f.raw_write(a, WRITE_SIZE, WRITE_SIZE + WRITE_OFFSET)
        print("Bytes written: " + str(ret))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sample_002.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)