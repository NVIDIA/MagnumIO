# pytorch-numpy-reader

This repository provides a GDS implementation for reading NumPy files as PyTorch extension. 

## Installation
Make sure that you have PyTorch >= 1.6 installed in your Python environment. Then clone this repo into any path and execute `python setup.py install`.
CUDA 10.2 or newer as well as GDS has to be available and the cufile libraries as well as the headers have to be in system search locations.

## Usage
The file `python/example.py` contains an example implementation of how to use this reader inside a PyTorch Dataset object. 
This example expects a directory `source` containing files named `<some-tag>_data.npy` and matching `<some-tag>_label.npy` for a segmentation task. In this case, the samples in the files are expected to be HWC format and we transpose them to CHW format before passing them to pytorch. 
We are using one numpy reader instance for data and another one for label and make sure that the files stay in sync during shuffling.
Please note that if `__getitem__` is called from a subprocess (e.g. by using the multiprocessing API from PyTorch DataLoader), the usual 
CUDA subprocess restrictions apply. For example, process forking is forbidden whereas spawning and using a forkserver is allowed, the latter is prefered. 

