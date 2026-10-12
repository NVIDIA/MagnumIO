# DeepCAM
DeepCAM is a scientific image segmentation application from climate research
and part of MLPerf HPC. Please see the corresponding [MLCommons website](https://github.com/mlcommons/hpc/tree/main/deepcam)
for more information.

## Introduction
The focus here is on inference, not training and thus this benchmark is
shipped	with a pre-trained DeepCAM model under the `share` subdirectory.
To make the repository more github-friendly, we have split the model into multiple parts but it can be re-assembled via

```
cat share/model_part* > share/model.pth
```

This step is required in order to run the inference workflow. 

## Running DeepCAM Using Docker
For convenience we provide combined scripts for launching a docker container built using the instructions from the [main page](../../README.md) and running the benchmark inside or downloading the dataset. Those scripts can be found under the `scripts` directory and are prefixed with `run_docker`. You will have to adjust some variables inside those scripts, for example specify the paths to the input data and turn GDS on (or off). Please see the comments in the scripts for details.

The sections below describe how the code can be run natively (or from inside a docker container).

## Obtaining the Data
The data is hosted at [NERSC](https://nerscg.gov), which is a facility of the US Department of energy in Berkeley, California. It can be downloaded via the [web interface](https://portal.nersc.gov/project/dasrepo/deepcam/climate-data/All-Hist/) or by using the download script provided with this benchmark. We recommended following the latter approach. The usage is as follows:

```
python utils/download_data.py --target-dir=<target directory> --num-files=<number of files total> --num-streams=<number of parallel streams>
```
The `--target-dir` specifies the directory the data will be downloaded to. 

The argument `--num-files` can be used to specify the number of files which are downloaded. In total, there are about 120K files which occupy around 10 TB of disk space. For this benchmark, we recommend to download O(100) files per GPU involved in the test.

The argument `--num-streams` determines how many threads are spawned to accelerate the download. Please do not specify a too large number here as the NERSC web server may regard this as a DOS attack and might block your IP. A value of up to 4 is recommended.
This scripts downloads the original data files in HDF5 format and converts them into numpy as required for this benchmark.


## Run the Benchmark
The benchmark uses mpi4py to coordinate work between different GPU in a distributed environment. The forward passes are embarassingly parallel
but after benchmark step a barrier is issued and performance metrics are computed.

In order to run the benchmark on multiple gpu, you can use something like

```
mpirun -np <num-total-GPUs> <other-MPI-options> \
       python driver/test_numpy_dali.py \
              --data_dirs <list with directories where the data is stored> \
              --trt_model_dir <when TensorRT is used, where to store or load TensorRT compiled to/from> \
              --outputfile <file which contains result performance metrics> \
              --num_warmup_runs <number of epochs which are excluded from measurements> \
	      --num_runs <number of measurement epochs> \
	      --batch_size <global batch size, will be distributed evenly across all GPUs> \
              --max_inter_threads <number of threads used for inter-sample-parallelism> \
              --max_intra_threads <number of threads used for intra-sample-parallelism (only applies to GDS enabled loader)> \
              <additional args>
```

The `<additional args>` steer how the benchmark is run. For example, for IO only measurements (no forward pass), specify `--mode=io`.
Accordingly, for inference specify `--mode=inference`. Furthermore, the benchmark offers several flags to enable a variety of performance
relevant features. For example, GDS can be enabled by specifying `--enable_gds`.
For more information about the individual options, use `python driver/test_numpy_dali.py --help`.

We also offer example run scripts under the `scripts` subdirectory. The `run_benchmark.sh` script performs a grid search over inter- and intra-threads
in order to optimize IO performance.

## TensorRT
The DeepCAM inference benchmark supports TensorRT support via [Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT/tree/v1.0.0) by specifying the `--enable_trt` flag. Additionally, a `--trt_model_dir` has to be specified, pointing to a writeable path which will host the TensorRT converted models. Please note that a model converted with Torch-TensorRT is tied to the GPU architecture and device ID it was converted on. Therefore, a separate model file will be stored for each GPU in a node.

Note: if you encounter error messages related to Tensor RT and device incompatibilities, remove the `trt` model files from the model directory to trigger a re-compilation. Errors like these occur if the system driver version has changed between compiling the models for TRT and running the code.

## GDS Support
The benchmark uses the [GDS enabled DALI numpy reader](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/numpy_reader.html#GPUDirect-Storage-Support). Numpy (npy) is the only file format in DALI which currently supports GDS. For more information about the pipeline used in this benchmark, please see `data/cam_numpy_dali_dataset.py`.

## Software Versions
The benchmark was tested with the following software versions:

PyTorch Container: nvcr.io/nvidia/pytorch:21.11-py3
CUDA SDK: 11.6
CUDA Driver: 510.47.03
GDS: 1.2.1
pytorch: 1.11.0a0+b6df043
cuDNN: 8.3.0
DALI: 1.12.0dev.20220304
Torch-TensorRT: 1.1.0a0+00da3d1f

The Dockerfile from the `pytorch/docker` already uses these versions. Concerning DALI, any version newer than the one stated is also supposed to work.
