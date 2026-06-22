################################################################################
#
# Copyright 1993-2025 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Linux Platforms)
#
################################################################################

# Common includes and paths for CUDA
ARCH ?= $(shell uname -m)
CUDA_PATH   := /usr/local/cuda
ifeq ($(ARCH),aarch64)
	CUFILE_PATH ?= /usr/local/cuda/targets/sbsa-linux/lib/
	CUFILE_INCLUDE_PATH ?= /usr/local/cuda/targets/sbsa-linux/include/
else
	CUFILE_PATH ?= /usr/local/cuda/targets/x86_64-linux/lib/
	CUFILE_INCLUDE_PATH ?= /usr/local/cuda/targets/x86_64-linux/include/
endif
CXXFLAGS    := -Wall
CXXFLAGS    += -I $(CUFILE_INCLUDE_PATH)
CXXFLAGS    += -I $(CUDA_PATH)/include/ 
CXXFLAGS    += -I ./include/
###########################
# Enable the following line for code coverage
ifneq ($(CONFIG_CODE_COVERAGE),)
CXXFLAGS    += -ftest-coverage -fprofile-arcs
endif
CXXFLAGS 	+= -std=c++17
###########################
CUDART_STATIC := -Bstatic -L $(CUDA_PATH)/lib64/ -lcudart_static -lrt -lpthread -ldl
LDFLAGS       :=  $(CUFILE_LIB) $(CUDART_STATIC) -lcrypto -lssl -g
CUFILE_LIB    := -L $(CUFILE_PATH) -lcufile
CUFILE_LIB_STATIC  := -L $(CUFILE_PATH) -lcufile_static
LDFLAGS       :=  $(CUFILE_LIB) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt -ldl
LDFLAGS_STATIC     :=  $(CUFILE_LIB_STATIC) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt -ldl
INSTALL_GDSSAMPLES_PREFIX = /usr/local/gds/samples
NVCC          := $(CUDA_PATH)//bin/nvcc
# CUDA 13.x CCCL headers (Thrust, CUB, libcu++) require C++17 for host code in .cu files.
NVCCFLAGS     += -std=c++17

################################################################################
CC:=/usr/bin/g++
DOCKER_BUILD_REPO=gds_build:manylinux_2_28-cuda-12-2
DOCKERFILE_BUILD_PATH=../../docker/manylinux_2_28-docker/Dockerfile-manylinux_2_28-cuda-12-2

# Target rules
all: build
release_samples:
	docker build -t $(DOCKER_BUILD_REPO) -f $(DOCKERFILE_BUILD_PATH) ./
	docker run -v `pwd`/../../:/nvme-direct:rw,z --rm -it $(DOCKER_BUILD_REPO) sh -c "cd /nvme-direct/tools/samples;$(MAKE) build"
	@echo "Release samples Built"

# Get all .cc files inside common/
common_cc := $(wildcard common/*.cc)
commonobjs := $(patsubst common/%.cc,common/%.o,$(common_cc))
samples = $(filter-out common/%.cc, $(wildcard */*.cc))
cusamples = $(wildcard */*.cu)
objs = $(samples:.cc=) $(samples:.cc=_static)
cuobjs = $(cusamples:.cu=) vectorAdd.o
build: $(commonobjs) $(objs) $(cuobjs)

CUDA_ARCH = -gencode=arch=compute_75,code=sm_75 \
			-gencode=arch=compute_80,code=sm_80 \
			-gencode=arch=compute_86,code=sm_86 \
			-gencode=arch=compute_87,code=sm_87 \
			-gencode=arch=compute_89,code=sm_89 \
			-gencode=arch=compute_90,code=sm_90 \
			-gencode=arch=compute_90,code=compute_90						

vectorAdd.o : vectorAdd.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) --cudadevrt static --cudart static \
		$(CUDA_ARCH)  --ptxas-options=-v --compiler-options \
		--compile --lib $< -o $@

%: %.cc $(CUFILE_INCLUDE_PATH)/cufile.h vectorAdd.o $(commonobjs)
	$(CC) $(INCLUDES) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
%_static: %.cc $(CUFILE_INCLUDE_PATH)/cufile.h vectorAdd.o $(commonobjs)
	$(CC)  $(INCLUDES) $(CXXFLAGS) $^ -o $@ $(LDFLAGS_STATIC)
	
5_cuFile_MemMap/memmap_thrust: 5_cuFile_MemMap/memmap_thrust.cu $(commonobjs)
	$(NVCC) $(NVCCFLAGS) -I $(CUFILE_INCLUDE_PATH) $(INCLUDES) --cudadevrt static --cudart static $(CUDA_ARCH) $^ -o $@ $(CUFILE_LIB) -lcuda
	$(NVCC) $(NVCCFLAGS) -I $(CUFILE_INCLUDE_PATH) $(INCLUDES) --cudart static $(CUDA_ARCH) $^ -o $@_static $(CUFILE_LIB_STATIC) -lcuda

install:
	cp -r vectorAdd.cu Makefile README.md 1_cuFile_Basics/ 2_cuFile_Multithreaded/ 3_cuFile_Batch/ 4_cuFile_Async/ 5_cuFile_MemMap/ include/ common/ $(INSTALL_GDSSAMPLES_PREFIX)/

clean:
	rm -f $(objs) $(cuobjs) $(cusamples:.cu=_static)
	rm -f *.o common/*.o cufile.log

.PHONY : build install clean
