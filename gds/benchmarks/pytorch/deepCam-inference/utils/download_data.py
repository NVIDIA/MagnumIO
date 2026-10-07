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
import io
import requests
import argparse as AP
import numpy as np
import h5py as h5
import concurrent.futures as cf


def process_file(source_path, target_path):
    # get file content
    try:
        response = requests.get(source_path)
    except:
        print(f"Cannot open file {source_path}")
        return (source_path, False)
    
    # load data
    handle = io.BytesIO(response.content)
    with h5.File(handle, "r") as f:
        data = f["climate"]["data"][...].astype(np.float32)
        label = np.stack([f["climate"]["labels_0"][...], f["climate"]["labels_1"][...]], axis=-1)

    # get file basename:
    basename = os.path.basename(source_path)

    # save data and label:
    dataname = os.path.join(target_path, basename.replace(".h5", ".npy"))
    labelname = os.path.join(target_path, basename.replace(".h5", ".npy").replace("data-", "label-"))

    # save stuff
    np.save(dataname, data)
    np.save(labelname, label)

    return (source_path, True)
    

def download_data(target_dir, num_files, overwrite = False):

    # fetch from here
    root_url = "https://portal.nersc.gov/project/dasrepo/deepcam/climate-data/All-Hist/"

    # get list of files
    response = requests.get(root_url + "validation/files.txt")
    filelist = response.content.decode('utf-8').strip().split("\n")
    filelist = sorted([x for x in filelist if x.endswith(".h5")])

    # create directory if doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # only use first n-samples
    if num_files > 0:
        filelist = filelist[0:num_files]

    # check which files are missing
    files_downloaded = [x for x in os.listdir(target_dir) if x.endswith(".npy") and x.startswith("data-")]

    # filter tags
    if not overwrite:
        files_missing = list(filter(lambda x: x.replace(".h5", ".npy") not in files_downloaded, filelist))
    else:
        files_missing = filelist
        
    # perform the loads
    executor = cf.ThreadPoolExecutor(max_workers = args.num_streams)
    
    # starts the download
    print("Starting download")
    retry_count=0
    while files_missing or (retry_count < args.num_retries):
        
        futures = []
        for fname in files_missing:
            futures.append(executor.submit(process_file, root_url + "validation/" + fname, target_dir))

        files_missing = []
        for future in cf.as_completed(futures):
            fname, status = future.result()
            if not status:
                print(f"Re-queueing {fname}")
                files_missing.append(os.path.basename(fname))

        # increment retry counter
        retry_count += 1
    print("done")
            
    if files_missing:
        files_missing_string = "\n".join(files_missing)
        print(f"The following files could not be downloaded: {files_missing_string}")
    

if __name__ == "__main__":
    parser = AP.ArgumentParser(description = 'Download and preprocess files for deepcam benchmark')
    parser.add_argument('--target-dir', type=str, help="directory where to download the files to")
    parser.add_argument('--num-files', type=int, default=-1, help="How many files to download, default will download all files")
    parser.add_argument('--num-streams', type=int, default=1, help="How many parallel streams do we want to employ for downloading")
    parser.add_argument('--num-retries', type=int, default=5, help="Number of retries scheduled per file before it is aborted")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # download data
    download_data(args.target_dir,
                  args.num_files,
                  args.overwrite)
