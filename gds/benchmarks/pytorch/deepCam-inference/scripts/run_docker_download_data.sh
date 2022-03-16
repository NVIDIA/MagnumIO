#!/bin/bash

# MIT License
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# set the paths
# this variable should point to the 
# directory where you want to download the data to
DATA_DIR=/my_data

# number of files to download
NUM_FILES=100

# number of parallel streams to use
NUM_STREAMS=4

#run
docker run --gpus=all \
              --rm \
	      --security-opt seccomp=unconfined \
	      --ipc host \
	      --net host \
	      --volume "${DATA_DIR}:/data:rw" \
	      --workdir "/opt/benchmarks/deepCam-inference" \
	      -it pytorch-gds-benchmarks:latest \
	      python utils/download_data.py \
	      --target-dir=/data \
	      --num-files=${NUM_FILES} \
	      --num-streams=${NUM_STREAMS}

