Memory Buffer Agnostic Sample

This sample demonstrates how software can detect where a memory buffer is allocated by querying the
CUDA runtime.

Compile using gnu make. 
$ make clean; make
$ make clean; make GDS_SUPPORT=TRUE # compile with cufile

Run instructions:
$ ./buffer-agnostic <filepath> <select memory buffer location> # 
$ ./buffer-agnostic test.bin -1 # write from a CPU memory buffer
$ ./buffer-agnostic test.bin 0 # write from GPU 0 memory buffer
$ ./buffer-agnostic test.bin 1 # write from GPU 1 memory buffer

