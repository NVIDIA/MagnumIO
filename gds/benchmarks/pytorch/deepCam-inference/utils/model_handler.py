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

# base modules
import os
import sys

# torch modules
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# tensorrt modules
import tensorrt as trt
import torch_tensorrt

# custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture import deeplab_xception


#reload model helper
def reload_model(model_path, model, device_id):
    # load checkpoint
    checkpoint = torch.load(model_path, map_location = device_id)

    # we need to do some key hacking for the model dict
    model_dict = {}
    for k in checkpoint:
        model_dict[k.replace("module.","")] = checkpoint[k]

    #load model
    model.load_state_dict(model_dict)


class ModelHandler(object):
    
    def __init__(self, pargs, channels, local_batch_size, device, comm_size, comm_rank, run_in_stream=False):
        # extract parameters
        self.use_fp16 = pargs.enable_fp16
        self.use_trt = pargs.enable_trt
        self.use_graphs = pargs.enable_graphs
        self.use_nhwc = pargs.enable_nhwc
        self.do_inference = (pargs.mode == "inference")
        self.do_train = (pargs.mode == "train")
        model_path = "/share/model.pth"
        trt_model_dir = pargs.trt_model_dir
        self.device = device
        
        # create stream
        self.pyt_stream = torch.cuda.Stream(self.device, -1)
        self.fw_stream = self.pyt_stream if run_in_stream else torch.cuda.current_stream()
        
        #init data parallel model
        net = deeplab_xception.DeepLabv3_plus(nInputChannels=4, n_classes=3, os=16, pretrained=False).to(self.device)
        if self.use_nhwc:
            net = net.to(memory_format=torch.channels_last)
        if self.do_train:
            net = DDP(net, device_ids=[self.device.index])
        else:
            reload_model(model_path, net, self.device)

        #broadcast
        net.eval()
        if self.use_fp16 and self.do_inference:
            net.half()

        # load trt model
        if self.use_trt and not self.do_train:
            # Torch-TensorRT debug level
            #torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)

            # set device
            torch_tensorrt.set_device(self.device.index)
            
            # filename
            trtfile = f"model_fp16_bs{local_batch_size}_gpu{self.device.index}.trt" \
                if self.use_fp16 else f"model_fp32_bs{local_batch_size}_gpu{self.device.index}.trt"
            if trt_model_dir is not None:
                trtfile = os.path.join(trt_model_dir, trtfile)
            if os.path.isfile(trtfile):
                if comm_rank == 0:
                    print("Loading TRT model")
                net_trt = torch.jit.load(trtfile, map_location=self.device)
            else:
                if comm_rank == 0:
                    print("Compiling TRT model")
                    
                # Torch-TensorRT compile settings
                input_dtype = torch.half if self.use_fp16 else torch.float
                input_shape = (local_batch_size, len(channels), 768, 1152)
                input_format = torch.channel_last if self.use_nhwc else torch.contiguous_format

                # JIT the model
                net_script = torch.jit.script(net)

                # TRT compile the model
                net_trt = torch_tensorrt.compile(net_script,
                                                 inputs=[torch_tensorrt.Input(input_shape, dtype=input_dtype, format=input_format)],
                                                 enabled_precisions={input_dtype},
                                                 device=self.device)

                if trt_model_dir is not None:
                    os.makedirs(trt_model_dir, exist_ok=True)
                    torch.jit.save(net_trt, trtfile)

            # switch to model
            self.net_fw = net_trt
        else:
            print("Using PyTorch model")
            self.net_fw = net

        if self.use_graphs:
            print("Capturing Graph")
            self.static_inputs = torch.ones((local_batch_size, len(channels), 768, 1152), dtype=torch.float32).to(device)
            if self.use_nhwc:
                self.static_inputs = self.static_inputs.contiguous(memory_format = torch.channels_last)
            if self.use_fp16:
                self.static_inputs = self.static_inputs.half()

            self.pyt_stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self.pyt_stream):
                with torch.no_grad():
                    for _ in range(10):
                        self.static_outputs = self.net_fw(self.static_inputs)
                    self.pyt_stream.synchronize()

                    # capture
                    self.graph = torch.cuda.CUDAGraph()
                    self.graph.capture_begin()
                    self.static_outputs = self.net_fw(self.static_inputs)
                    self.graph.capture_end()
            
            self.pyt_stream.synchronize()
            print("Graph Capture Done")

            
    def sync(self):
        self.fw_stream.synchronize()

        
    def run(self, inputs, labels):
        #training?
        if self.do_train:
            with torch.cuda.stream(self.fw_stream):
                #FW
                with amp.autocast(enabled = self.use_fp16):
                    outputs = self.net_fw(inputs)
            result = outputs, labels
            
        #inference?
        if self.do_inference:

            with torch.no_grad():
                    
                #convert to FP16 if requested
                with torch.cuda.stream(self.fw_stream):
                    if self.use_fp16:
                        inputs  = inputs.half()
                    if self.use_nhwc:
                        inputs = inputs.to(memory_format=torch.channels_last)

                    if self.use_graphs:
                        self.static_inputs.copy_(inputs)
                        self.graph.replay()
                        outputs = self.static_outputs.clone()
                    else:
                        #pass forward
                        outputs = self.net_fw.forward(inputs)
            result = outputs, None
        else:
            result = None, None
            
        return result
