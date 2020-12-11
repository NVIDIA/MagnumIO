# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#custom reader
import numpy_reader as nr

#dataset class
class SegmentationDataset(Dataset):

    #set a new path for files
    def init_files(self, source):
        self.source = source
        self.files = [x.replace("_data.npy", "") for x in sorted(os.listdir(self.source)) if x.endswith("_data.npy")]
        
        if self.shuffle:
            np.random.shuffle(self.files)
            
        self.length = len(self.files)

    
    def __init__(self, source, num_intra_threads = 1, device = -1, shuffle = False):
        self.shuffle = shuffle

        #init files
        self.init_files(source)
        
        #init numpy loader
        filename = os.path.join(self.source, self.files[0])
        #data
        self.npr_data = nr.numpy_reader(split_axis = False, device = device)
        self.npr_data.num_intra_threads = num_intra_threads
        self.npr_data.parse(filename + "_data.npy")
        #label
        self.npr_label = nr.numpy_reader(split_axis = False, device = device)
        self.npr_label.num_intra_threads = num_intra_threads
        self.npr_label.parse(filename + "_label.npy")        
        
        print("Initialized dataset with ", self.length, " samples.")

    def __len__(self):
        return self.length

    @property
    def shapes(self):
        return self.npr_data.shape, self.npr_label.shape
        
    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])
    
        try:
            #load data
            self.npr_data.init_file(filename + "_data.npy")
            X = self.npr_data.get_sample(0)
            self.npr_data.finalize_file()

            #load label
            self.npr_label.init_file(filename + "_label.npy")
            Y = self.npr_label.get_sample(0)
            self.npr_label.finalize_file()
        
        except OSError:
            print("Could not open file " + filename)
            sleep(5)
        
        #preprocess
        X = X.permute(2, 0, 1)
        
        return X, Y, filename
