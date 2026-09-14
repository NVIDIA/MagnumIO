# Third-Party Notices

This repository is NVIDIA-authored source. The Python diagnostic tool is
distributed under the Apache License, Version 2.0. The agent skill component
(skills/) is additionally licensed under the Creative Commons Attribution 4.0
International License (CC-BY-4.0). No vendored third-party source code or
dependency manifests were found in the tracked repository at the time of this
review.

The toolkit documents and may invoke external system tools, packages, services,
or interfaces, including:

- NVIDIA CUDA Toolkit, GPUDirect Storage, `gdscheck`, `nvidia-smi`,
  `nvidia_fs`, and `libcufile`
- NVIDIA DOCA, DOCA-OFED, or MLNX_OFED documentation and packages
- Linux kernel, sysfs, procfs, PCI, NVMe, RDMA, mount, and distribution package
  manager interfaces
- Python 3 standard library modules

Those external components are not vendored in this repository and remain under
their own licenses and distribution terms. If future changes add vendored,
copied, modified, generated, or required third-party code, document the source,
license, and review notes here before distribution.
