# Common includes and paths for CUDA
CUDA_PATH   := /usr/local/cuda
CUFILE_PATH := /usr/local/gds/lib
CXXFLAGS    += -I $(CUDA_PATH)/include/ 

GDS_SUPPORT = FALSE

CXX:=nvcc

###########################
# Enable the following line for code coverage
#CXXFLAGS    += -ftest-coverage -fprofile-arcs
###########################
CUDA_LIB    := -L $(CUDA_PATH)/lib64/ -lcuda -L $(CUDA_PATH)/lib64/ -lcudart
################################################################################

ifeq ($(GDS_SUPPORT),TRUE)
CXXFLAGS    += -I $(CUFILE_PATH) -DGDS_SUPPORT
CUFILE_LIB  := -L $(CUFILE_PATH) -lcufile
LDFLAGS     :=  $(CUFILE_LIB) $(CUDA_LIB) -lcrypto -lssl -lpthread -ldl
else
LDFLAGS     :=  $(CUDA_LIB) -lcrypto -lssl -lpthread -ldl
endif

# Target rules
all: build

build: buffer-agnostic

%: %.cc $(CUFILE_PATH)/cufile.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

.PHONY : clean
clean:
	rm -f buffer-agnostic
