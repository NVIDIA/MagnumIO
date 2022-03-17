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
# directory where you downloaded the data to
DATA_DIR=/my_data

# this is the directory to which the result 
# files are written
OUTPUT_DIR=/my_runs

# this is a directory to which the TensorRT 
# compiled models will be written to
MODEL_DIR=/my_models

# switch for disabling (0) or enabling GDS (1)
ENABLE_GDS=1

#run
docker run --gpus=all \
           --rm \
           --security-opt seccomp=unconfined \
           --privileged \
           --ipc host \
           --net host \
           --volume /run/udev:/run/udev:ro \
           --device /dev/nvidia-fs0 \
           --device /dev/nvidia-fs1 \
           --device /dev/nvidia-fs2 \
           --device /dev/nvidia-fs3 \
           --device /dev/nvidia-fs4 \
           --device /dev/nvidia-fs5 \
           --device /dev/nvidia-fs6 \
           --device /dev/nvidia-fs7 \
           --device /dev/nvidia-fs8 \
           --device /dev/nvidia-fs9 \
           --device /dev/nvidia-fs10 \
           --device /dev/nvidia-fs11 \
           --device /dev/nvidia-fs12 \
           --device /dev/nvidia-fs13 \
           --device /dev/nvidia-fs14 \
           --device /dev/nvidia-fs15 \
           --volume "${DATA_DIR}:/data:ro" \
           --volume "${OUTPUT_DIR}:/runs:rw" \
           --volume "${MODEL_DIR}:/models:rw" \
           --workdir "/opt/benchmarks/deepCam-inference/scripts" \
           -it pytorch-gds-benchmarks:latest \
           ./run_benchmark.sh ${ENABLE_GDS}

