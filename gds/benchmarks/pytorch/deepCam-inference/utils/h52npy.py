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

import os
import numpy as np
import h5py as h5
import argparse as ap
from tqdm import tqdm


def main(pargs):
    
    #check path
    inputpath = pargs.input
    outputpath = pargs.output

    #check inputs
    filenames = [x for x in os.listdir(inputpath) if x.endswith(".h5")]

    #create outputpath if doesn't exist:
    os.makedirs(outputpath, exist_ok=True)

    
    for filename in tqdm(filenames):
        #read input
        inputfile = os.path.join(inputpath, filename)
        with h5.File(inputfile, 'r') as f:
            data = f["climate"]["data"][...].astype(np.float32)
            label = np.stack([f["climate"]["labels_0"][...], f["climate"]["labels_1"][...]], axis=-1)

        #write output
        filenamebase = os.path.splitext(filename)[0]
        datafile = os.path.join(outputpath, filenamebase+'_data.npy')
        if not os.path.isfile(datafile):
            np.save(datafile, data)
        labelfile = os.path.join(outputpath, filenamebase+'_label.npy')
        if not os.path.isfile(labelfile):
            np.save(labelfile, label)


if __name__ == '__main__':
    AP = ap.ArgumentParser()
    AP.add_argument("--input", type=str, help="input directory with hdf5 files")
    AP.add_argument("--output",type=str, help="output directory for the npy files")
    parsed = AP.parse_args()
    
    main(parsed)
