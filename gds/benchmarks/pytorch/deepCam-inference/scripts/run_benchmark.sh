#!/bin/bash

# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#env
export OMPI_MCA_btl=^openib

#mpi options
mpioptions="--allow-run-as-root"

# paths
data_dirs="/data"
output_prefix="my_experiment"
model_dir="/models"
overwrite_output=1

#global parameters
num_gpu=8
num_warmup_runs=3
num_runs=5
max_threads=32
padded=0
loadertag="numpy-dali"
mode="inference"
memmap=0

# switch to enable or disable GDS support
gds=${1:-0}

# do not modify beyond this point
if [ "${gds}" == "1" ]; then
    inter_threads="32 16 8 4 2 1"
    intra_threads="32 16 8 4 2 1"
else
    inter_threads="32 16 8 4 2 1"
    intra_threads="1"
fi
loadersuffix=${loadertag/\-/_}

if [ "${memmap}" == "1" ]; then
    memmaptag="-memmap"
    memmaparg=""
else
    memmaptag=""
    memmaparg="--disable_mmap"
fi

if [ "${gds}" == "1" ]; then
    devarg="--enable_gds"
    devtag="-gpu"
else
    devarg=""
    devtag="-cpu"
fi

# construct arguments
common_args="${devarg} ${memmaparg} --enable_fp16 --enable_trt --enable_graphs"
common_tags="fp16-trt-graph"
if [ "${mode}" == "inference" ]; then
    args="${common_args} --drop_fs_cache --preprocess --mode=inference"
    tags="${common_tags}-nocache-preprocess-inference${padtag}${memmaptag}"
elif [ "${mode}" == "training" ]; then
    args="${common_args} --preprocess --mode=train"
    tags="${common_tags}-preprocess-train${padtag}${memmaptag}"
elif [ "${mode}" == "io" ]; then
    args="${devarg} ${memmaparg} --drop_fs_cache"
    tags="nocache${padtag}${memmaptag}"
elif [ "${mode}" == "io-cache" ]; then
    args="${devarg} ${memmaparg}"
    tags="${padtag}${memmaptag}"
fi


# loop over thread/batchsize configs
for totalranks in ${num_gpu}; do
    for local_batch_size in 8 4 2 1; do
	batch_size=$(( ${totalranks} * ${local_batch_size} ))
	for max_inter_threads in ${inter_threads}; do
	    for max_intra_threads in ${intra_threads}; do
	    
		#check if we have too many threads
		nthreads=$(( ${max_inter_threads} * ${max_intra_threads} ))
		if [[ ${nthreads} -gt ${max_threads} ]]; then
		    continue
		fi
	    
		#check if output already exists
		outputpath="/runs/${output_prefix}"
		outputfilename="iostats_${loadertag}${devtag}_features${tags}_bs${batch_size}_ngpu${totalranks}_ninterthreads${max_inter_threads}_nintrathreads${max_intra_threads}.out"
	    	
		# create output directory
		if [ ! -d "${outputpath}" ]; then
		    mkdir -p ${outputpath}
		fi
	
	        # overwrite if requested
		if [ "${overwrite_output}" == "1" ]; then
		    rm -f ${outputpath}/${outputfilename}
		fi

		# skip file if exists
		if [ -f "${outputpath}/${outputfilename}" ]; then
		    echo "${outputpath}/${outputfilename} already exists, skipping"
		    continue
		fi
	    	
		#run the benchmark
		mpirun -np ${totalranks} ${mpioptions} bind.sh --cpu=exclusive \
		       $(which python) ../driver/test_${loadersuffix}.py \
		       --data_dirs ${data_dirs} \
		       --trt_model_dir ${model_dir} \
		       --outputfile "${outputpath}/${outputfilename}" \
		       --num_warmup_runs ${num_warmup_runs} \
		       --num_runs ${num_runs} \
		       --batch_size ${batch_size} \
		       --max_inter_threads ${max_inter_threads} \
		       --max_intra_threads ${max_intra_threads} \
		       ${args}
		
		# check for execution errors
		if [ $? -ne 0 ]; then
                    exit
		fi
		
	    done
	done
    done
done

