#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Global Paramaters
MAGNUM_IO_VERSION=21.07
MIN_DRIVER=470.42
MIN_CUDA=11.4.0
MIN_DOCKER=20.10.3

SCRIPT_NAME=$(basename $0)
RUNFROM=$(dirname $(readlink -f $0))
LOGFILE=install.log
REBOOT=0
LOGOUT=0


if [ -f /etc/redhat-release ]; then
  if grep -q -i "release 7" /etc/redhat-release ; then
    OS_FLAVOR="redhat"
    OS_RELEASE=7
  elif grep -q -i "release 8" /etc/redhat-release ; then
    OS_FLAVOR="redhat"
    OS_RELEASE=8
  else
    echo "Only Red Hat 7.x and 8.x flavors supported."
    exit
  fi
else
  OS_FLAVOR="ubuntu"
  OS_RELEASE=$(lsb_release -r | awk '{print $2}')
  if [ $OS_RELEASE != "18.04" ] && [ $OS_RELEASE != "20.04" ]; then
    echo "Only Ubuntu 18.04 and 20.04 supported."
    exit
  fi
fi


nvlog () {
  echo "##NV## `date` ## $1"
}


require_user () {
  if [ $(id -u) = 0 ]; then
    nvlog "ERROR: Cannot run this step as root, run script as user or without 'sudo'"
    exit 1
  fi
}


semver_gte () {
  # $1 >= $2 ?
  [ "$2" != "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}


detect_driver () {
  if [ -f /usr/bin/nvidia-smi ]; then
    DRIVER_VER=$(/usr/bin/nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d " " -f1 2> /dev/null)
    if [ $? -ne 0 ]; then
      DRIVER_VER=0
    fi
    if [ $DRIVER_VER = "Failed" ]; then
      DRIVER_VER=0
    fi
  else
    DRIVER_VER=0
  fi
}


install_driver () {
  nvlog "START Installing Driver"

  if [ $OS_FLAVOR = "ubuntu" ]; then
    if [ -f /usr/bin/nvidia-uninstall ]; then
      cat << EOF

  Found /usr/bin/nvidia-uninstall which means a driver .run file was used
  on this machine. Driver install/update cannot proceed. The solution is to
  purge the driver and reinstall it with the correct apt repositories.

  Make sure you are connected to the internet and run:

    ${SCRIPT_NAME} purge-driver
    ${SCRIPT_NAME} install-driver

  Then rerun the command you just ran to proceed.

EOF
      exit
    fi
  fi

  semver_gte $DRIVER_VER $MIN_DRIVER
  if [ $? -eq 1 ]; then
    nvlog "Driver is new enough - skip install"
    nvlog "END Installing Driver"
    return
  fi

  set -e
  if [ $OS_FLAVOR = "ubuntu" ]; then
    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt-get -y update
    sudo apt-get -y upgrade
    sudo apt-get -y install nvidia-driver-470
    sudo apt-get -y autoremove
    REBOOT=1
  else
    nvlog "Automated NVIDIA driver install on Red Hat is not supported."
    nvlog "Please install NVIDIA driver $MIN_DRIVER or newer and run again."
    exit
  fi
  set +e

  nvlog "END Installing Driver"
}


purge_driver () {
  nvlog "START Purge Driver"

  if [ $OS_FLAVOR != "ubuntu" ]; then
    nvlog "ERROR: Automated NVIDIA driver purge for Red Hat not supported."
    nvlog "Please run /usr/bin/nvidia-uninstall and reboot to remove driver."
    exit
  fi

  cat << EOF

WARNING:
Removing the NVIDIA Driver will also remove CUDA and other libraries like
nvidia-docker2 that depend on the driver.

Helpful once the system is rebooted:

    ${SCRIPT_NAME} setup-system

EOF

  read -p "DANGER: Are you SURE [y/N]?" -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    nvlog "Starting removal..."
    if [ -f /usr/bin/nvidia-uninstall ]; then
      nvlog "Running /usr/bin/nvidia-uninstall first."
      sudo /usr/bin/nvidia-uninstall
    fi
    sudo apt-get -y purge nvidia-*
    sudo apt -y autoremove
    sudo rm -f /etc/modprobe.d/blacklist-nouveau.conf
    sudo rm -f /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
    sudo update-initramfs -k all -u
    REBOOT=1
  else
    nvlog "Aborting - doing nothing"
  fi

  nvlog "END Purge Driver"
}


detect_cuda () {
  if [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VER=$(cat /usr/local/cuda/version.txt | awk '{ print $3 }' 2> /dev/null)
    if [ $? -ne 0 ]; then
      CUDA_VER=0
    fi
  else
    CUDA_VER=0
  fi
}


install_cuda () {
  nvlog "START Installing CUDA"
  semver_gte $CUDA_VER $MIN_CUDA
  if [ $? -eq 1 ]; then
    nvlog "CUDA is new enough - skip install"
    nvlog "END Installing CUDA"
    return
  fi

  set -e
  if [ $OS_FLAVOR = "ubuntu" ]; then
    if [ $OS_RELEASE = "18.04" ]; then
      curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin \
        -o cuda.pin
      sudo mv cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600
      sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
      sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
      sudo apt-get update
      sudo apt-get -y install cuda-toolkit-11-4

      echo "export PATH=/usr/local/cuda/bin/:\$PATH # MAGNUMIO-DEV-ENV-ADDED" >> ${HOME}/.bashrc
      echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/lib:\$LD_LIBRARY_PATH # MAGNUMIO-DEV-ENV-ADDED" >> ${HOME}/.bashrc
      source ${HOME}/.bashrc
    else
      curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
        -o cuda.pin
      sudo mv cuda.pin /etc/apt/preferences.d/cuda-repository-pin-600
      sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
      sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
      sudo apt-get update
      sudo apt-get -y install cuda-toolkit-11-4

      echo "export PATH=/usr/local/cuda/bin/:\$PATH # MAGNUMIO-DEV-ENV-ADDED" >> ${HOME}/.bashrc
      echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/lib:\$LD_LIBRARY_PATH # MAGNUMIO-DEV-ENV-ADDED" >> ${HOME}/.bashrc
      source ${HOME}/.bashrc
    fi
  else
    if [ $OS_FLAVOR = "redhat7" ]; then
      sudo yum-config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
      sudo yum clean all
      sudo yum install -y cuda-toolkit-11-4
    else
      sudo dnf config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
      sudo dnf clean all
      sudo dnf -y install cuda-toolkit-11-4
    fi

    echo "export PATH=/usr/local/cuda/bin/:\$PATH # MAGNUMIO-DEV-ENV-ADDED" >> ${HOME}/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/lib:\$LD_LIBRARY_PATH # MAGNUMIO-DEV-ENV-ADDED" >> ${HOME}/.bashrc
    source ${HOME}/.bashrc
  fi
  set +e

  nvlog "END Installing CUDA"
}


detect_docker () {
  DOCKER_VER=$(docker version --format '{{.Client.Version}}' 2> /dev/null)
  if [ $? -ne 0 ]; then
    DOCKER_VER=0
  fi
}


install_docker () {
  nvlog "START Installing Docker and NVIDIA Container Toolkit"
  semver_gte $DOCKER_VER $MIN_DOCKER
  if [ $? -eq 1 ]; then
    nvlog "Docker is new enough, checking for nvidia-docker2..."

    if [ $OS_FLAVOR = "ubuntu" ]; then
      nvd2=$(dpkg -l | grep nvidia-docker2 | grep ii)
    else
      nvd2=$(yum list installed | grep nvidia-docker2)
    fi

    if [ "$nvd2" != "" ]; then
      nvlog "nvidia-docker2 found, no install needed"
      nvlog "END Installing Docker and NVIDIA Container Toolkit"
      return
    fi
  fi

  set -e
  if [ $OS_FLAVOR = "ubuntu" ]; then
    # NVIDIA Repo
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    # Docker Repo
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

    sudo apt-get -y update
    sudo apt-get -y install \
      apt-transport-https \
      ca-certificates \
      gnupg-agent \
      software-properties-common
    sudo apt-get -y install \
      nvidia-docker2 \
      docker-ce \
      docker-ce-cli \
      containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo groupadd -f docker
    sudo systemctl restart docker
  elif [ $OS_FLAVOR = "redhat7" ]; then
    # NVIDIA Repos
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
      sudo tee /etc/yum.repos.d/nvidia-docker.repo
    # Docker Repo
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

    sudo yum install -y \
      nvidia-docker2 \
      docker-ce \
      docker-ce-cli \
      containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo groupadd -f docker
    sudo systemctl restart docker
  else # RHEL 8
    # NVIDIA Repos
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
      sudo tee /etc/yum.repos.d/nvidia-docker.repo
    # Docker Repo
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

    # force new-enough containerd.io
    curl -s https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.2.13-3.2.el7.x86_64.rpm \
      --output containerd.rpm
    sudo yum localinstall -y containerd.rpm
    rm containerd.rpm

    sudo yum install -y \
      docker-ce \
      nvidia-container-toolkit
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo groupadd -f docker
    sudo systemctl restart docker
  fi
  set +e

  nvlog "END Installing Docker and NVIDIA Container Toolkit"
}


docker_adduser () {
  nvlog "START Add User to Docker Group"
  if groups $USER | grep -qw '\bdocker\b'; then
    nvlog "User already member of docker group"
    nvlog "END Add User to Docker Group"
    return
  fi

  set -e
  nvlog "Adding user '$USER' to docker group"
  sudo usermod -aG docker $USER
  sudo setfacl -m user:$USER:rw /var/run/docker.sock
  sudo systemctl daemon-reload
  sudo systemctl reload docker
  LOGOUT=1
  set +e

  nvlog "END Add User to Docker Group"
}


build_container () {
  nvlog "START Building Container - magnum-io:${MAGNUM_IO_VERSION}"

  docker build \
    --tag magnum-io:${MAGNUM_IO_VERSION} \
    -f ./magnum-io.Dockerfile .

  nvlog "END Building Container"
}


purge_container () {
  nvlog "START Purge Container - magnum-io:${MAGNUM_IO_VERSION}"

  read -p "DANGER: Are you sure you want to remove container magnum-io:${MAGNUM_IO_VERSION} [y/N]?" -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    nvlog "Removing container magnum-io:${MAGNUM_IO_VERSION}"
    CMD="docker rmi -f magnum-io:${MAGNUM_IO_VERSION}"
    nvlog "${CMD}"
    ${CMD}
  else
    nvlog "Aborting - no container deleted"
  fi

  nvlog "END Purge Container"
}


install_nccl () {
  nvlog "START Install NCCL"

  nvlog "END Install NCCL"
}


diagnostics () {
  nvlog "START Diagnostics"

  nvlog "Run as: $USER"
  nvlog "OS Flavor: $OS_FLAVOR"
  nvlog "OS Release: $OS_RELEASE"
  if [ -f /usr/bin/lsb_release ]; then
    nvlog "lsb_release:"
    /usr/bin/lsb_release -r -d
  fi
  if [ -f /bin/uname ]; then
    nvlog "uname -a"
    /bin/uname -a
  fi

  nvlog "Storage (non-tmpfs, non-loopback)"
  df -h | grep -v dev/loop | grep -v tmpfs

  nvlog "Network test"
  ping -c 1 -W 3 8.8.8.8

  nvlog "Driver detected (0 means not installed): $DRIVER_VER"
  nvlog "NVIDIA SMI:"
  if [ -f /usr/bin/nvidia-smi ]; then
    /usr/bin/nvidia-smi
  else
    echo "nvidia-smi not found, NVIDIA GPU Driver not installed correctly."
  fi

  nvlog "CUDA detected (0 means not installed): $CUDA_VER"
  nvlog "Docker detected (0 means not installed): $DOCKER_VER"

  nvlog "Shared libraries:"
  ldconfig -p | grep 'nvidia\|libnv\|cuda\|libcu'

  nvlog "END Diagnostics"
}


notify_reboot () {
  nvlog
  nvlog
  nvlog "ACTION REQUIRED:"
  nvlog "For the changes to take effect, reboot the machine."
  nvlog
  nvlog "Current working directory: `pwd`/"
  nvlog "script: ${RUNFROM}/${SCRIPT_NAME}"
  nvlog
  nvlog
}


notify_logout () {
  nvlog
  nvlog
  nvlog "ACTION REQUIRED:"
  nvlog "For the changes to take effect, please logout and login again."
  nvlog
  nvlog "Current working directory: `pwd`/"
  nvlog "script: ${RUNFROM}/${SCRIPT_NAME}"
  nvlog
  nvlog
}


usage () {
  more << EOF

NVIDIA Magnum IO Development Environment Installer v${MAGNUM_IO_VERSION}

Usage: ${SCRIPT_NAME} COMMAND

Quick Start:

    ./${SCRIPT_NAME} setup-system

    # To pull the latest from from NGC visit
    # https://ngc.nvidia.com/catalog/containers/nvidia:magnum-io:magnum-io
    # To build locally:
    ./${SCRIPT_NAME} build-container

    ./${SCRIPT_NAME} run-container

Information Commands:

    help
        Display help and usage.
    version
        Display version information. Version:, Latest:,
        and an Error: if the current release version cannot be retrieved.
    diagnostics
        Display software versions and info.

Setup Commands:

    setup-system
        Setup system with NVIDIA Magnum IO Development Environment software.
    install-driver
        Install the NVIDIA GPU driver v${MIN_DRIVER}+.
        Automated install not supported on Red Hat.
    purge-driver
        Purge the NVIDIA GPU driver from the system, before a clean reinstall.
        Automated purge not supported on Red Hat.
    install-docker
        Install Docker CE v${MIN_DOCKER}+.
    setup-docker-user
        Setup user permissions to use docker. Useful when multiple users
        use a machine. Normal done by setup-system for first user.
        The user must have sudo permission.
    install-cuda
        Install the NVIDIA CUDA Toolkit v${MIN_CUDA}+

Containers Commands:

    build-container
        Build the container.
    run-container
        Run container, with HOME directory mounted.
    purge-container
        Removes the container.
    build-dockerfile
        Regenerate Dockerfile using HPCCM.

EOF
}


(
detect_driver
detect_cuda
detect_docker

case "$1" in
  version)
    echo Version: ${MAGNUM_IO_VERSION}
    ;;

  diagnostics)
    require_user
    diagnostics
    ;;

  setup-system)
    require_user
    nvlog "OS Flavor: $OS_FLAVOR"
    nvlog "Driver detected: $DRIVER_VER"
    nvlog "Docker detected: $DOCKER_VER"
    install_driver
    install_cuda
    install_docker
    docker_adduser
    ;;

  install-driver)
    install_driver
    ;;
  purge-driver)
    purge_driver
    ;;

  install-cuda)
    install_cuda
    ;;

  install-docker)
    install_docker
    ;;
  setup-docker-user)
    require_user
    docker_adduser
    ;;

  build-dockerfile)
    nvlog "To rebuild Dockerfile, HPC Container Maker (HPCCM) is used"
    nvlog "Installable with \"pip install hpccm\" or from https://github.com/NVIDIA/hpc-container-maker"
    hpccm --recipe magnum-io-hpccm.py --format docker > magnum-io.Dockerfile
    nvlog "Finished Dockerfile rebuild"
    ;;
  build-container)
    build_container
    ;;
  purge-container)
    purge_container
    ;;
  run-container)
    nvlog "Running docker container magnum-io:${MAGNUM_IO_VERSION} with $HOME directory mounted"
    nvlog 'Container run command:
      docker run --gpus all --rm -it \
        --user "$(id -u):$(id -g)" \
        --volume $HOME:$HOME \
        --volume /run/udev:/run/udev:ro \
        --workdir $HOME \
        magnum-io:'${MAGNUM_IO_VERSION}
    docker run --gpus all --rm -it \
      --user $(id -u):$(id -g) \
      --volume $HOME:$HOME \
      --volume /run/udev:/run/udev:ro \
      --workdir $HOME \
      magnum-io:${MAGNUM_IO_VERSION}
   ;;

  help)
    usage
    ;;

  *)
    usage
    ;;
esac

if [ $REBOOT -ne 0 ]; then
  notify_reboot
  exit
fi

if [ $LOGOUT -ne 0 ]; then
  notify_logout
  exit
fi

)2>&1 | tee -a $LOGFILE