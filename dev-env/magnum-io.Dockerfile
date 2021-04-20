# GENERATED FILE, DO NOT EDIT

FROM nvcr.io/nvidia/cuda:11.2.2-devel-ubuntu20.04

# NVIDIA Nsight Systems 2021.1.1
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/hpccm.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nsight-systems-cli-2021.1.1 && \
    rm -rf /var/lib/apt/lists/*

# Mellanox OFED version 5.2-2.2.0.0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - && \
    mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/5.2-2.2.0.0/ubuntu20.04/mellanox_mlnx_ofed.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ibverbs-providers \
        ibverbs-utils \
        libibmad-dev \
        libibmad5 \
        libibumad-dev \
        libibumad3 \
        libibverbs-dev \
        libibverbs1 \
        librdmacm-dev \
        librdmacm1 && \
    rm -rf /var/lib/apt/lists/*

# GDRCOPY version 2.2
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/NVIDIA/gdrcopy/archive/v2.2.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/v2.2.tar.gz -C /var/tmp -z && \
    cd /var/tmp/gdrcopy-2.2 && \
    mkdir -p /usr/local/gdrcopy/include /usr/local/gdrcopy/lib && \
    make prefix=/usr/local/gdrcopy lib lib_install && \
    echo "/usr/local/gdrcopy/lib" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig && \
    rm -rf /var/tmp/gdrcopy-2.2 /var/tmp/v2.2.tar.gz
ENV CPATH=/usr/local/gdrcopy/include:$CPATH \
    LIBRARY_PATH=/usr/local/gdrcopy/lib:$LIBRARY_PATH

# UCX version 1.10.0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils-dev \
        file \
        libnuma-dev \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/openucx/ucx/releases/download/v1.10.0/ucx-1.10.0.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/ucx-1.10.0.tar.gz -C /var/tmp -z && \
    cd /var/tmp/ucx-1.10.0 &&   ./configure --prefix=/usr/local/ucx --disable-assertions --disable-debug --disable-doxygen-doc --disable-logging --disable-params-check --disable-static --enable-mt --enable-optimizations --with-cuda=/usr/local/cuda --with-gdrcopy=/usr/local/gdrcopy && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    echo "/usr/local/ucx/lib" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig && \
    rm -rf /var/tmp/ucx-1.10.0 /var/tmp/ucx-1.10.0.tar.gz
ENV CPATH=/usr/local/ucx/include:$CPATH \
    LIBRARY_PATH=/usr/local/ucx/lib:$LIBRARY_PATH \
    PATH=/usr/local/ucx/bin:$PATH

# NVSHMEM 2.0.2-0
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://developer.nvidia.com/nvshmem-src-202-0 && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/nvshmem-src-202-0 -C /var/tmp && \
    cd /var/tmp/nvshmem_src_2.0.2-0 && \
    CUDA_HOME=/usr/local/cuda NVSHMEM_MPI_SUPPORT=0 NVSHMEM_PREFIX=/usr/local/nvshmem make -j$(nproc) install && \
    rm -rf /var/tmp/nvshmem_src_2.0.2-0 /var/tmp/nvshmem-src-202-0
ENV CPATH=/usr/local/nvshmem/include:$CPATH \
    LIBRARY_PATH=/usr/local/nvshmem/lib:$LIBRARY_PATH \
    PATH=/usr/local/nvshmem/bin:$PATH

# GDS 0.95

RUN wget -qO - https://repo.download.nvidia.com/baseos/GPG-KEY-dgx-cosmos-support | apt-key add - && \
    echo "deb https://repo.download.nvidia.com/baseos/ubuntu/focal/x86_64/ focal-updates preview" >> /etc/apt/sources.list.d/hpccm.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gds-tools-11-2=0.95.0.94-1 \
        libcufile-11-2=0.95.0.94-1 \
        libcufile-dev-11-2=0.95.0.94-1 && \
    rm -rf /var/lib/apt/lists/*

COPY magnum-io.Dockerfile \
    third_party.txt \
    README.md \
    /

ENV MAGNUM_IO_VERSION=21.04

SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash" ]


