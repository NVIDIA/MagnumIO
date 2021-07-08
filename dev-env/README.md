# NVIDIA Magnum IO Developer Environment

NVIDIA Magnum IO is the collection of I/O technologies from NVIDIA and
Mellanox that make up the I/O subsystem of the modern data center, and
enable applications at scale. Making use of GPUS, or scaling an application
up to multiple GPUs, or scaling it out across multiple nodes, will probably
make use of the libraries in Magnum IO.

The Magnum IO Developer Environment container contains a comprehensive set
of tools to scale I/O. It serves two primary purposes 1) Allow developers to
begin scaling applications on a laptop, desktop, workstation, or in the cloud.
2) Serve as the basis for a build container locally or in a CI/CD system.

Quick Links:

* [Magnum IO Container on NGC](https://ngc.nvidia.com/catalog/containers/nvidia:magnum-io:magnum-io)
* [Magnum IO Code repo](https://github.com/NVIDIA/MagnumIO)
* [Report issues](https://github.com/NVIDIA/MagnumIO/issues)

## Contents

* [Quick Start](#quick-start)
* [Minimum Hardware and Software](#minimum-hardware-and-software)
* [Magnum IO Components](#magnum-io-components)
* [Operating System Setup](#operating-system-setup)
* [Installing the NVIDIA GPU Driver](#installing-the-nvidia-gpu-driver)
* [More Information](#more-information)

## Quick Start

There are two ways to get the container and run it. With either option
the system must be setup to run GPU enabled containers, which the
`installer.sh` script can do.

Recommended is to pull the NVIDIA NGC Catalog (option 1). If you plan to
customize the container further, it can be used as the FROM to build a new
container. Also supported is building the container locally (option 2).

_For usage and command documentation:_ `./installer.sh help`

### Option 1: Pull container from NVIDIA NGC Catalog:

1. Setup system

```bash
# Clone repo
git clone https://github.com/NVIDIA/MagnumIO.git
cd magnumio/dev-env

# Setup system (driver, CUDA, Docker)
./installer.sh setup-system
```
2. Visit https://ngc.nvidia.com/catalog/containers/nvidia:magnum-io:magnum-io
and find the latest version.
3. Pull

```bash
docker pull nvcr.io/nvidia/magnum-io/magnum-io:TAG
```

4. Run

```bash
docker run --gpus all --rm -it \
  --user "$(id -u):$(id -g)" \
  --volume $HOME:$HOME \
  --volume /run/udev:/run/udev:ro \
  --workdir $HOME \
  magnum-io:TAG
```

### Option 2: Build the container locally:

```bash
# Clone repo
git clone https://github.com/NVIDIA/MagnumIO.git
cd magnumio/dev-env

# Setup system (driver, CUDA, Docker)
./installer.sh setup-system

# Build the container
./installer.sh build-container

#Run the container with HOME directory mounted
./installer.sh run-container
```

### Upgrading

The `installer.sh` script is designed to detect old versions of dependencies
and upgrade them as needed. Simply `git pull` a new version, and run the
system-setup again.

```bash
git pull
./installer.sh setup-system
```

## Minimum Hardware and Software

The container should run and build code on any system with a GPU and minimal
drivers enforced by the `installer.sh` script. This allows developers to
integrate Magnum IO APIs easily, and get started on almost any system.

However some components of Magnum IO require specific hardware or
configurations to run applications with the APIs fully enabled.

For example GDS requires the latest Tesla, Volta or Ampere GPUs, ext4
mounting options, and GDS enabled storage systems (an NVMe drive at minimum).
More on GDS setup in the
[NVIDIA GPUDirect Storage Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)

In practice, this means almost all development can be done on a local system,
many tests can be run locally, but tests at scale are done in a cluster or
cloud environment.

## Magnum IO Components

### NCCL

The
[NVIDIA Collective Communications Library](https://developer.nvidia.com/nccl)
(NCCL, pronounced “nickel”)
is a library providing inter-GPU communication primitives that are
topology-aware and can be easily integrated into applications.

NCCL is smart about I/O on systems with complex topology: systems with
multiple CPUs, GPUs, PCI busses, and network interfaces. It can selectively
use NVLink, Ethernet, and InfiniBand, using multiple links when possible.
Consider using NCCL APIs whenever you plan your application or library to
run on a mix of multi-GPU multi-node systems in a data center, cloud, or
hybrid system. At runtime, NCCL determines the topology and optimizes
layout and communication methods.

### NVSHMEM

[NVSHMEM](https://developer.nvidia.com/nvshmem) creates a global address
space for data that spans the memory of multiple GPUs and can be accessed
with fine-grained GPU-initiated operations, CPU-initiated operations, and
operations on CUDA streams.

In many HPC workflows, models and simulations are run that far exceed the
size of a single GPU or node. NVSHMEM allows for a simpler asynchronous
communication model in a shared address space that spans GPUs within or
across nodes, with lower overheads, possibly resulting in stronger scaling
compared to a traditional Message Passing Interface (MPI).

### UCX

[Unified Communication X](https://www.openucx.org/documentation/) (UCX) uses
high-speed networks, including InfiniBand, for inter-node communication and
shared memory mechanisms for efficient intra-node communication.  If you
need a standard CPU-driven MPI, PGAS OpenSHMEM libraries, and RPC, GPU-aware
communication is layered on top of UCX.

UCX is appropriate when driving I/O from the CPU, or when system memory is
being shared. UCX enables offloading the I/O operations to both host adapter
(HCA) and switch, which reduces CPU load. UCX simplifies the portability of
many peer-to-peer operations in MPI systems.

### GDS

[NVIDIA GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage)
(GDS) enables a direct data path for Remote Direct Memory Access (RDMA)
transfers between GPU memory and storage, which avoids a bounce buffer and
management by the CPU. This direct path increases system bandwidth and
decreases the latency and utilization load on the CPU.

GDS and the cuFile APIs should be used whenever data needs to move directly
between storage and the GPU. With storage systems that support GDS,
significant increases in performance on clients are observed when I/O is a
bottleneck. In cases where the storage system does not support GDS, I/O
transparently falls back to normal file reads and writes.

Moving the I/O decode/encode from the CPU to GPU creates new opportunities for
direct data transfers between storage and GPU memory which can benefit from
GDS performance. An increasing number of data formats are supported in CUDA.

### NSight

[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) lets you
see what’s happening in the system and
[NVIDIA Cumulus NetQ](https://cumulusnetworks.com/products/netq/) allows you
to analyze what’s happening on the NICs and switches.
Both are critical to finding some causes of bottlenecks in multi-node
applications.

Nsight Systems is a low-overhead performance analysis tool designed to
provide insights that you need to optimize your software. It provides
everything that you would expect from a profiler for a GPU. Nsight Systems
has a tight integration with many core CUDA libraries, giving you detailed
information on what is happening.

Nsight Systems allows you to see exactly what’s happening on the system,
what code is taking a long time, and when algorithms are waiting on GPU/CPU
compute, or device I/O. Nsight Systems is relevant to Magnum IO and included
in the Magnum IO container for convenience, but its scope spans well outside
of Magnum IO to monitoring compute that’s unrelated to I/O.

### NetQ (not in container)

NetQ is a highly scalable, modern, network operations tool set that provides
visibility, troubleshooting and lifecycle management of your open networks
in real time. It enables network profiling functionality that can be used
along with Nsight Systems or application logs to observe the network’s
behavior while the application is running.

NetQ is part of Magnum IO given its integral involvement in managing IO in
addition to profiling it, but is not inside the container as it runs on the
nodes and switches of the network.

## Operating System Setup

Disable "Secure Boot" in the system BIOS/UEFI before installing Linux.

### Ubuntu

The Data Science stacks are supported on Ubuntu LTS 18.04.1+ or 20.04
with the 4.15+ kernel. Ubuntu can be downloaded from
[https://www.ubuntu.com/download/desktop](https://www.ubuntu.com/download/desktop)

Support for Red Hat is planned in later releases.

## Installing the NVIDIA GPU Driver

It is important that updated NVIDIA drivers are installed on the system.
The minimum version of the NVIDIA driver supported is 460.39.
More recent drivers may be available, and should work correctly.

### Ubuntu or RHEL v8.x Driver Install

Driver install for Ubuntu is handled by `./installer.sh setup-system`
so no manual install should be required.

If the driver if too old or the script is having problems, the driver can
be removed (this may have side effects, read the warnings) and reinstalled:

```bash
./installer.sh purge-driver
# reboot
./installer.sh setup-system
# reboot
```

## More Information

* [NVIDIA Magnum IO](https://developer.nvidia.com/magnum-io)
* [Blog: Optimizing Data Movement in GPU Applications with the NVIDIA Magnum IO Developer Environment](https://developer.nvidia.com/blog/optimizing-data-movement-in-gpu-apps-with-magnum-io-developer-environment/)
