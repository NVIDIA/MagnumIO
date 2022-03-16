# Pytorch GDS Benchmarks

## Introduction
This repository hosts GDS enabled benchmarks written in PyTorch.
At this point in time, only DeepCAM is available, but we hope to add
more benchmarks soon.

Please check the individual README files under the corresponding benchmark folders for more information.

## Docker Container
We recommend using docker to run these benchmarks. To this end, we are providing a dockerfile along with a docker build script under the `docker` subdirectory. In order to build the container perform the following steps:

```
cd docker
bash build_docker.sh
```

Note that since this benchmark has many dependencies, the build process can take a few minutes.

## Running Benchmarks
The benchmark source codes are copied into the container and can be found under `/opt/benchmarks`. This is
done to simplify the deployment process, but the benchmark folders can also be mounted into the container instead.
For instructions on running a specific benchmark please read the respective instructions in the sub folder. 
If you are using docker, make sure that you step inside the container first before you execute any of the commands listed there. For convenience, most benchmarks provide a script for launching a container and executing the code inside. Those scripts can be found under the `scripts` subdirectory and are prefixed with `run_docker_`.
