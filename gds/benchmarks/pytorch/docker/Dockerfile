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

FROM nvcr.io/nvidia/pytorch:21.11-py3 as builder

ENV DEBIAN_FRONTEND noninteractive

# install GDS
RUN apt update -y && apt install -y liburcu-dev linux-headers-generic #linux-headers-$(uname -r)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03\
-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub && \
    apt-get update -y && apt install -y libcufile-dev-11-6

# cleanup
RUN rm -rf /usr/local/cuda /usr/local/cuda-11 && \
    ln -s /usr/local/cuda-11.5/ /usr/local/cuda && \
    ln -s /usr/local/cuda-11.5/ /usr/local/cuda-11 && \
    cp -r /usr/local/cuda-11.6/targets/x86_64-linux/include/* /usr/local/cuda/include/ && \
    cp -r /usr/local/cuda-11.6/targets/x86_64-linux/lib/* /usr/local/cuda/lib64/ && \
    cp -r /usr/local/cuda-11.6/gds /usr/local/cuda

# torch_tensorrt prereqs
# bazel
RUN apt-get update && apt-get install -y curl gnupg  && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
    mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install -y bazel-4.0.0 && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/bazel-4.0.0 /usr/bin/bazel
# Workaround for bazel expecting both static and shared versions, we only use shared libraries inside container
RUN cp /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer_static.a

# notebook
RUN pip install notebook

# install mpi4py
RUN pip install h5py mpi4py

# install other prereqs
RUN pip install matplotlib PILLOW

FROM builder

# torch_tensorrt
RUN cd /opt && git clone https://github.com/NVIDIA/Torch-TensorRT.git torch_tensorrt && \
    cd /opt/torch_tensorrt && \
    git checkout -b stable-21.12 --no-track origin/release/ngc/21.12 && \
    git fetch origin pull/719/head:fix_branch && \
    git config user.name "Container" && \
    git config user.email "" && \
    git merge fix_branch

RUN cd /opt/torch_tensorrt && cp ./docker/WORKSPACE.docker WORKSPACE
RUN cd /opt/torch_tensorrt && bazel build //:libtorchtrt --compilation_mode opt
RUN cd /opt/torch_tensorrt/py && \
    pip install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org && \
    jupyter nbextension enable --py widgetsnbextension

# Locale is not set by default
RUN apt-get update && apt-get install -y locales ninja-build && \
    rm -rf /var/lib/apt/lists/* && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN cd /opt/torch_tensorrt/py && python3 setup.py bdist_wheel --use-cxx11-abi
RUN cd /opt/torch_tensorrt/py/dist && pip install *.whl && rm -rf *.whl

RUN conda init bash

ENV LD_LIBRARY_PATH /opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/lib:$LD_LIBRARY_PATH
ENV PATH /opt/conda/lib/python3.8/site-packages/torch_tensorrt/bin:${PATH}

# upgrade DALI to nightly
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110

# copy benchmarks
#deepcam
COPY ./deepCam-inference /opt/benchmarks/deepCam-inference
RUN mkdir /share && cat /opt/benchmarks/deepCam-inference/share/model_part* > /share/model.pth

# create additional folders for mapping data in
RUN mkdir -p /data && mkdir -p /data && mkdir -p /data/output

# env variables
ENV UCX_MEMTYPE_CACHE n
ENV UCX_TLS cma,mm
