# NVIDIA Magnum IO

This repository is a nexus for application/framework developers who make use
of Magnum IO and developers of the various components in the Magnum IO
platform. Its purpose is to accelerate development by sharing open source
code, governed by the Apache 2.0 license. The existing structure provides
for samples, tests, building blocks, and more.

Magnum IO is the IO subsystem of the modern accelerated data center.
It pertains to data movement, access, and management. As described in
<https://developer.nvidia.com/magnum-io>
it covers network IO, storage IO, compute in the network, and IO management.

We welcome you to browse the growing body of code here, and to submit your
own contributions as PRs. Signing the license is one of the steps of making
a PR. Over time, a subset of the code here will be functionally validated,
and a subset may be performance evaluated. Not all of the code will be
accepted into the mainline, but PRs will still be visible to other
developers as an inspiration.

Please find the license file in this repo, along with a file which contains
the license-related text that should be included in all source code files.

## Subdirectories

### /dev-env

Contains the source files for the NVIDIA Magnum IO Developer Environment.
NVIDIA publishes the Magnum IO Developer Environment as an NGC Container
at <https://ngc.nvidia.com/catalog/containers/nvidia:magnum-io:magnum-io>
containing the libraries and tools to create Magnum IO accelerated
applications.
This allows developers to begin scaling their applications on a laptop,
desktop, workstation, or in the cloud.

### /gds

Sample code and readers for GPUDirect Storage -
<https://developer.nvidia.com/gpudirect-storage>
