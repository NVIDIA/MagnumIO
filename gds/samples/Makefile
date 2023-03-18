################################################################################
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
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
CUDA_PATH   := /usr/local/cuda
CUFILE_PATH ?= /usr/local/cuda/targets/x86_64-linux/lib/
CUFILE_INCLUDE_PATH ?= /usr/local/cuda/targets/x86_64-linux/include/
CXXFLAGS    := -Wall
CXXFLAGS    += -I $(CUDA_PATH)/include/ 
CXXFLAGS    += -I $(CUFILE_INCLUDE_PATH)
###########################
# Enable the following line for code coverage
ifneq ($(CONFIG_CODE_COVERAGE),)
CXXFLAGS    += -ftest-coverage -fprofile-arcs
endif
CXXFLAGS += -std=c++17
###########################
CUDART_STATIC := -Bstatic -L $(CUDA_PATH)/lib64/ -lcudart_static -lrt -lpthread -ldl
LDFLAGS     :=  $(CUFILE_LIB) $(CUDART_STATIC) -lcrypto -lssl
CUFILE_LIB  := -L $(CUFILE_PATH) -lcufile
CUFILE_LIB_STATIC  := -L $(CUFILE_PATH) -lcufile_static
#CUFILE_LIB := -Bstatic -L $(CUFILE_PATH) -lcufile_static
LDFLAGS     :=  $(CUFILE_LIB) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt -ldl
LDFLAGS_STATIC     :=  $(CUFILE_LIB_STATIC) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt -ldl
INSTALL_GDSSAMPLES_PREFIX = /usr/local/gds/samples
NVCC          :=$(CUDA_PATH)//bin/nvcc

################################################################################
CC:=g++
DOCKER_BUILD_REPO=gds_build:manylinux2014-mofed-5-1-cuda-11-1
DOCKERFILE_BUILD_PATH=../../docker/manylinux2014-docker/Dockerfile-manylinux2014-cuda-11-1-MOFED-5-1
# Target rules
all: build
release_samples:
	docker build -t $(DOCKER_BUILD_REPO) -f $(DOCKERFILE_BUILD_PATH) ./
	docker run -v `pwd`/../../:/nvme-direct:rw --rm -it $(DOCKER_BUILD_REPO) sh -c "cd /nvme-direct/tools/samples;$(MAKE) build"
	@echo "Release samples Built"

vectorAdd.o : vectorAdd.cu
	$(NVCC) $(INCLUDES) --cudart static -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 --ptxas-options=-v --compiler-options --compile --lib -o vectorAdd.o vectorAdd.cu

build: cufile_sample_001 cufile_sample_002 cufile_sample_003 cufile_sample_004 cufile_sample_005 cufile_sample_006 cufile_sample_007 cufile_sample_008 cufile_sample_009 cufile_sample_010 cufile_sample_011 cufile_sample_012 cufile_sample_013 cufile_sample_014 cufile_sample_015 cufile_sample_016 cufile_sample_017 cufile_sample_018 cufile_sample_019 cufile_sample_020 cufile_sample_021 cufile_sample_022 cufile_sample_023 cufile_sample_024 cufile_sample_025 cufile_sample_026 cufile_sample_027 cufile_sample_001_static cufile_sample_002_static cufile_sample_003_static cufile_sample_004_static cufile_sample_005_static cufile_sample_006_static cufile_sample_007_static cufile_sample_008_static cufile_sample_009_static cufile_sample_010_static cufile_sample_011_static cufile_sample_012_static cufile_sample_013_static cufile_sample_014_static cufile_sample_015_static cufile_sample_016_static cufile_sample_017_static cufile_sample_018_static cufile_sample_019_static cufile_sample_020_static cufile_sample_021_static cufile_sample_022_static cufile_sample_023_static cufile_sample_024_static cufile_sample_025_static cufile_sample_026_static cufile_sample_027_static

cufile_sample_023: cufile_sample_023.cc vectorAdd.o
	$(CC) $(CXXFLAGS) $< vectorAdd.o -o $@ $(LDFLAGS)

cufile_sample_025: cufile_sample_025.cc vectorAdd.o
	$(CC) $(CXXFLAGS) $< vectorAdd.o -o $@ $(LDFLAGS)

cufile_sample_026: cufile_sample_026.cc vectorAdd.o
	$(CC) $(CXXFLAGS) $< vectorAdd.o -o $@ $(LDFLAGS)

cufile_sample_023_static: cufile_sample_023.cc vectorAdd.o
	$(CC) $(CXXFLAGS) $< vectorAdd.o -o $@ $(LDFLAGS_STATIC)

cufile_sample_025_static: cufile_sample_025.cc vectorAdd.o
	$(CC) $(CXXFLAGS) $< vectorAdd.o -o $@ $(LDFLAGS_STATIC)

cufile_sample_026_static: cufile_sample_026.cc vectorAdd.o
	$(CC) $(CXXFLAGS) $< vectorAdd.o -o $@ $(LDFLAGS_STATIC)

cufile_sample_024: cufile_sample_024.cu
	$(NVCC) -I $(CUFILE_INCLUDE_PATH) $(INCLUDES) $< -o $@ $(CUFILE_LIB) -lcuda
cufile_sample_024_static: cufile_sample_024.cu
	$(NVCC) -I $(CUFILE_INCLUDE_PATH) $(INCLUDES) --cudart static $< -o $@ $(CUFILE_LIB_STATIC) -lcuda

%: %.cc $(CUFILE_INCLUDE_PATH)/cufile.h
	$(CC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
%_static: %.cc $(CUFILE_INCLUDE_PATH)/cufile.h
	$(CC) $(CXXFLAGS) $< -o $@ $(LDFLAGS_STATIC)

install:
	cp vectorAdd.cu cufile_sample_024.cu cufile_sample_*.cc cufile_sample_*.h Makefile README $(INSTALL_GDSSAMPLES_PREFIX)/

clean:
	find . -type f -executable -delete
	rm -f *.o

.PHONY : build install clean
