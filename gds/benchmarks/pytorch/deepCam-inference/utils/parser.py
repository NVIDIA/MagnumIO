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

import argparse as ap

def parse_arguments():
    # set up parser
    AP = ap.ArgumentParser()
    AP.add_argument("--outputfile", type=str, help="Full path to output file.")
    AP.add_argument("--data_dirs", type=str, nargs='+', help="List of directories which hold data. The files will be sharded evenly across all ranks/GPUs.")
    AP.add_argument("--trt_model_dir", type=str, default=None, help="Directory where to store and read TRT models to and from.")
    AP.add_argument("--num_warmup_runs", type=int, default=1, help="Number of warmup experiments to run.")
    AP.add_argument("--num_runs", type=int, default=1, help="Number of experiments to run.")
    AP.add_argument("--batch_size", type=int, default=16, help="Global batch size. Make sure it is bigger than the number of ranks.")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_intra_threads", type=int, default=8, help="Maximum degree of parallelism within reader")
    AP.add_argument("--device_count", type=int, default=None, help="Number of devices, necessary to override torch default detection.")
    AP.add_argument("--device_id", type=int, default=None, help="Select device to run on, if None it is selected automatically.")
    AP.add_argument("--global_shuffle", action='store_true')
    AP.add_argument("--shuffle", action='store_true')
    AP.add_argument("--visualize", action='store_true')
    AP.add_argument("--visualization_output_dir", type=str, default=None, help="Path for storing the visualizations (if requested).")
    AP.add_argument("--preprocess", action='store_true')
    AP.add_argument("--mode", type=str, choices=["train", "inference", "io"], help="Which mode to run the benchmark in")
    AP.add_argument("--enable_gds", action='store_true')
    AP.add_argument("--enable_fp16", action='store_true')
    AP.add_argument("--enable_trt", action='store_true')
    AP.add_argument("--enable_graphs", action='store_true')
    AP.add_argument("--enable_nhwc", action='store_true')
    AP.add_argument("--drop_fs_cache", action='store_true')
    AP.add_argument("--disable_mmap", action='store_true')
    parsed = AP.parse_args()

    # sanitization
    if parsed.mode in {"inference", "train"}:
        parsed.preprocess = True

    if parsed.enable_gds:
        # we do not need to drop caches here and disable mmap
        parsed.drop_fs_cache = False
        parsed.disable_mmap = True
    
    return parsed
