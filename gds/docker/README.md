## Overview

GPUDirect Storage needs access to character devices to communicate with `nvidia-fs.ko` and MLNX_OFED stack. To enable GPUDirect Storage support in Docker environments, the script `gds-run-container` is a convenient wrapper around docker command to enable GPUDirect Storage support for deploying in development environments and testing GDS in containers.

## Security Considerations
To enable GPUDirect Storage, the container will need to add following flags

<table>
<thead>
<tr>
<th>docker option</th>
<th>purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td> --gpus=all</td>
<td> Enable nvidia GPUS</td>
</tr>
<tr>
<td> --ipc host  </td>
<td> For GDS stats using shared memory</td>
</tr>

<tr>
<td> --volume /run/udev:/run/udev:ro </td>
<td> To allow detection of block devices, GPUs </td>
</tr>

<tr>
<td> /dev/nvidia*, /dev/nvidia-caps* </td>
<td> Access to GPU character devices are provided by specifying --gpus option </td>
</tr>
<tr>
<td> /proc/driver/ </td>
<td> Paths to access nvidia-fs stats are already exposed by default using procfs paths /proc/driver/ </td>
</tr>
<tr>
<td>/sys/class/</td>
<td> sysfs paths /sys/class/ for udev devices are exposedby default for docker containers </td>
</tr>
<tr>
<td> --net host --cap-add=IPC_LOCK --device=/dev/infiniband/rdma_cm </td>
<td> For MOFED stack using RDMA connection manager </td>
</tr>
<tr>
<td> --device=/dev/infiniband/uverbs* </td>
<td> Access to uverbs API from userspace for WekaFS and GPFS </td>
</tr>
<tr>
<td>--device=/dev/nvidia-fs*</td>
<td> Access to nvidia-fs.ko character devices </td>
</tr>
</tbody>
</table>

## Usage

Assuming the script is installed in /usr/local/cuda/gds/tools `PATH=$PATH:/usr/local/cuda/gds/tools/`
``` bash
$ gds-run-container help
Note: gds-run-container is a wrapper script around docker to provide GDS specific flags and device(s)

GPUDirect Storage Options:
 	--enable-gds         pass nvidia_fs character device(s) and flags
 	--enable-gds-compat  allow for GDS to work in compat mode only
 	--enable-mofed       pass  mellanox character device(s) and flags


Usage:	docker [OPTIONS] COMMAND

A self-sufficient runtime for containers

Options:
   ...
   ...

  unpause     Unpause all processes within one or more containers
  update      Update configuration of one or more containers
  version     Show the Docker version information
  wait        Block until one or more containers stop, then print their exit codes

Run 'docker COMMAND --help' for more information on a command.
```

**Note** The script is a pass through for all other docker commands

``` bash
$ gds-run-container image ls

REPOSITORY            TAG                                 IMAGE ID       CREATED        SIZE

gds-partner           1.0.0-cuda-11.2                     e8319f254587   39 hours ago   5.36GB
nvcr.io/nvidia/cuda   11.2.1-devel-ubuntu20.04            6f43d7321de0   8 weeks ago    4.2GB
hello-world           latest                              d1165f221234   3 months ago   13.3kB
nvcr.io/nvidia/cuda   11.1-devel-ubuntu20.04              0185bd6d8ed9   6 months ago   4.76GB
```

## Running a GDS test docker container with MLNX_OFED and GDS support

``` bash
WORK_DIR=/opt/gds
GDS_VOLUME=/GDS_MOUNT
IMAGE="gds-partner:1.0.0-cuda-11.2-mlnx-5.1-2.5.8.0"

$ gds-run-container run --rm --gpus=all --enable-mofed --enable-gds\
                      --volume ${GDS_VOLUME}:/data:rw \
                      --workdir ${WORK_DIR} \
                      -it ${IMAGE} /usr/local/gds/tools/gdscheck -p
```

## Running a GDS test docker container without MLNX_OFED support
``` bash
WORK_DIR=/opt/gds
GDS_VOLUME=/GDS_MOUNT
IMAGE="gds-partner:1.0.0-cuda-11.2-mlnx-5.1-2.5.8.0"

$ gds-run-container run --rm --gpus=all --enable-gds\
                        --volume ${GDS_VOLUME}:/data:rw \
                        --workdir ${WORK_DIR} \
                        -it ${IMAGE} /usr/local/gds/tools/gdscheck -p
```
## Running a GDS test docker container in compat mode only
``` bash
WORK_DIR=/opt/gds
GDS_VOLUME=/GDS_MOUNT
IMAGE="gds-partner:1.0.0-cuda-11.2-mlnx-5.1-2.5.8.0"

$ gds-run-container run --rm --gpus=all --enable-gds-compat \
                        --volume ${GDS_VOLUME}:/data:rw \
                        --workdir ${WORK_DIR} \
                        -it ${IMAGE} /usr/local/gds/tools/gdscheck -p
```

## Running a pytorch GDS container with GDS support
``` bash
WORK_DIR=/opt/gds
GDS_VOLUME=/GDS_MOUNT
IMAGE="pytorch-gds"

$ gds-run-container run --rm --gpus=all --enable-mofed --enable-gds\
            --volume ${GDS_VOLUME}:/data:rw \
            --workdir ${WORK_DIR} \
            -it ${IMAGE} bash
```

## Using cufile.json inside the container
Each container will need to be able to provide a ``cufile.json`` file. By default ``libcufile.so`` will be checking the ``/etc/cufile.json path``. The path to the configuration file can be changed using the environment variable **CUFILE_ENV_PATH_JSON**.


Example, assuming /opt/gds/cufile.json is present in the image or copied to the container.

``` bash
WORK_DIR=/opt/gds
GDS_VOLUME=/GDS_MOUNT
IMAGE="pytorch-gds"

$ gds-run-container run --rm --gpus=all --enable-mofed --enable-gds\
                        CUFILE_ENV_PATH_JSON=/opt/gds/cufile.json \
                        --volume ${GDS_VOLUME}:/data:rw \
                        --workdir ${WORK_DIR} \
                        -it ${IMAGE} bash
```
For further detail on the configuration file contents please refer to [GPUDirect Storage documentation](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html#gds-parameters).
