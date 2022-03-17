# MIT License
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import glob
import numpy as np
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

class CamDaliDataloader(object):

    def get_pipeline(self):
        pipeline = Pipeline(self.batchsize, self.num_threads, self.device, seed=333)
        
        with pipeline:
            data = fn.readers.numpy(name="data",
                                    device = self.io_device,
                                    files = self.data_files,
                                    num_shards = self.num_shards,
                                    shard_id = self.shard_id,
                                    stick_to_shard = self.stick_to_shard,
                                    shuffle_after_epoch = self.shuffle,
                                    prefetch_queue_depth = 2,
                                    cache_header_information = True,
                                    register_buffers = True,
                                    dont_use_mmap = (not self.use_mmap) or (self.io_device == "gpu")).gpu()

            if self.load_label:
                label = fn.readers.numpy(name="label",
                                         device = self.io_device,
                                         files = self.label_files,
                                         num_shards = self.num_shards,
                                         shard_id = self.shard_id,
                                         stick_to_shard = self.stick_to_shard,
                                         shuffle_after_epoch = self.shuffle,
                                         prefetch_queue_depth = 2,
	                                 cache_header_information = True,
                                         register_buffers = True,
                                         dont_use_mmap = (not self.use_mmap) or (self.io_device == "gpu")).gpu()

            data = fn.transpose(data,
                                device = "gpu",
                                perm = [2, 0, 1])

            if self.load_label:
                pipeline.set_outputs(data, label)
            else:
                pipeline.set_outputs(data)
        
        return pipeline
        
    
    def init_files(self, root_dirs, prefix_data, prefix_label):
        self.root_dirs = root_dirs
        self.prefix_data = prefix_data
        self.prefix_label = prefix_label

        # get files
        self.data_files = []
        for directory in self.root_dirs:
            self.data_files += sorted(glob.glob(os.path.join(directory, self.prefix_data)))
            
        if self.load_label:
            self.label_files = []
            for directory in self.root_dirs:
                self.label_files += sorted(glob.glob(os.path.join(directory, self.prefix_label)))

        # shuffle globally if requested
        if not self.stick_to_shard and self.shuffle:
            self.rng = np.random.default_rng(seed=333)
            perm = self.rng.permutation(len(self.data_files))
            self.data_files = np.array(self.data_files)[perm].tolist()
            if self.load_label:
                self.label_files = np.array(self.label_files)[perm].tolist()
        
        # get shapes
        self.data_shape = np.load(self.data_files[0]).shape
        if self.load_label:
            self.label_shape = np.load(self.label_files[0]).shape
            
        # clean up old iterator
        if self.iterator is not None:
            del(self.iterator)
            self.iterator = None
        
        # clean up old pipeline
        if self.pipeline is not None:
            del(self.pipeline)
            self.pipeline = None

        # io devices
        self.io_device = "gpu" if self.read_gpu else "cpu"
            
        # define pipeline
        self.pipeline = self.get_pipeline()

        # build pipeline
        self.pipeline.build()
        
        # build pipes
        self.length = len(self.data_files)

        # create iterator but do not prepare first batch
        tags = ['data', 'label'] if self.load_label else ['data']
        self.iterator = DALIGenericIterator([self.pipeline], tags,
                                            reader_name="data", auto_reset = True,
	                                    prepare_first_batch = False,
                                            last_batch_policy = LastBatchPolicy.DROP)
               
        
    def __init__(self, root_dirs, prefix_data, prefix_label,
                 channels, batchsize, num_threads = 1, device = -1,
                 num_shards = 1, shard_id = 0, stick_to_shard = True,
                 lazy_init = False, read_gpu = False, use_mmap = True,
                 shuffle = False, preprocess = True):
    
        # read filenames first
        self.channels = channels
        self.batchsize = batchsize
        self.num_threads = num_threads
        self.device = device
        self.io_device = "gpu" if read_gpu else "cpu"
        self.use_mmap = use_mmap
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.read_gpu = read_gpu
        self.pipeline = None
        self.iterator = None
        self.lazy_init = lazy_init
        self.load_label = prefix_label is not None

        # sharding
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.stick_to_shard = stick_to_shard

        # init files
        self.init_files(root_dirs, prefix_data, prefix_label)
        

    @property
    def shapes(self):
        if self.load_label:
            return self.data_shape, self.label_shape
        else:
            return self.data_shape

    @property
    def sample_sizes(self):
        data_size = np.prod(self.data_shape) * 4
        label_size = 0 if not self.load_label else np.prod(self.label_shape) * 4
        return data_size, label_size

    
    def __iter__(self):
        if self.load_label:
            for token in self.iterator:
                data = token[0]['data']
                label = token[0]['label']

                if self.preprocess:
                    data = data[:, self.channels, ...]
                    label = torch.squeeze(label[..., 0])
            
                yield data, label, ""
        else:
            for token in self.iterator:
                data = token[0]['data']
                
                if self.preprocess:
                    data = data[:, self.channels, ...]
                
                yield data, None, ""
