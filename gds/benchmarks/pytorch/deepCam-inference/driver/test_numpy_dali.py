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

# Basics
import os
import sys
import numpy as np
import time
import subprocess as sp
import psutil

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import torch.distributed as dist

#MPI
from mpi4py import MPI

#print helper
def printr(msg, comm, rank=0):
    if comm.Get_rank() == rank:
        print(msg)

    
# Custom
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from utils import utils
from utils import losses
from utils import model_handler as mh
from utils import parser

#DALI
from data import cam_numpy_dali_dataset as cam


def main(pargs):

    #init MPI
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    num_devices = torch.cuda.device_count() if pargs.device_count is None else pargs.device_count
    comm_local_rank = comm_rank % num_devices

    # parameters fro prediction
    visualize = pargs.visualize
    use_gds = pargs.enable_gds
    use_fp16 = pargs.enable_fp16
    use_trt = pargs.enable_trt
    use_graphs = pargs.enable_graphs
    use_nhwc = pargs.enable_nhwc
    do_inference = (pargs.mode == "inference")
    do_train = (pargs.mode == "train")
    drop_fs_cache = pargs.drop_fs_cache
    preprocess = pargs.preprocess
    channels = [0,1,2,10]
    batch_size = pargs.batch_size
    local_batch_size = batch_size // comm_size
    max_threads = pargs.max_inter_threads

    if visualize and pargs.visualization_output_dir is None:
        raise ValueError("Please specify a valid --visualization_output_dir if you want to visualize the results.")
    
    # enable distributed training if requested
    if do_train:
        addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        comm_addr = addrport.split(":")[0]
        comm_port = "29500"
        os.environ["MASTER_ADDR"] = comm_addr
        os.environ["MASTER_PORT"] = comm_port
        dist.init_process_group(backend = "nccl",
                                rank = comm_rank,
                                world_size = comm_size)
    
    # parameters for visualization
    if (pargs.visualization_output_dir is not None) and visualize:
        predict_dir = os.path.join(pargs.visualization_output_dir, "predict")
        os.makedirs(predict_dir, exist_ok=True)
        truth_dir = os.path.join(pargs.visualization_output_dir, "true")
        os.makedirs(truth_dir, exist_ok=True)    
    
    # Initialize run
    rng = np.random.RandomState(seed=333)
    torch.manual_seed(333)
    
    # Define architecture
    if torch.cuda.is_available():
        printr("Using GPUs", comm, 0)
        if pargs.device_id is not None:
            device = torch.device("cuda", pargs.device_id)
        else:
            device = torch.device("cuda", comm_local_rank)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    else:
        printr("Using CPUs", comm, 0)
        device = torch.device("cpu")

    # create model handler
    model = mh.ModelHandler(pargs,
                            channels,
                            local_batch_size,
                            device,
                            comm_size,
                            comm_rank,
                            run_in_stream = use_gds)

    # set up for training
    if do_train:
        model.net_fw.train()
        gscaler = amp.GradScaler(enabled = use_fp16)
        optimizer = optim.AdamW(model.net_fw.parameters(), lr = 1.e-4)
        criterion = losses.fp_loss
        
    # Get data
    data_dirs = pargs.data_dirs
    data_loader = cam.CamDaliDataloader(data_dirs,
                                        prefix_data = "data-*.npy",
                                        prefix_label = "label-*.npy" if do_train or visualize else None,
                                        channels = channels,
                                        batchsize = local_batch_size,
                                        num_threads = max_threads,
                                        device = device.index,
                                        num_shards = comm_size,
                                        shard_id = comm_rank,
                                        stick_to_shard = not pargs.global_shuffle,
                                        lazy_init = True,
                                        read_gpu = use_gds,
                                        use_mmap = not pargs.disable_mmap,
                                        shuffle = pargs.shuffle,
                                        preprocess = preprocess)
    
    #create vizc instance
    if visualize:
        viz = vizc.CamVisualizer()

    
    printr("starting benchmark", comm, 0)

    # for cpu itilization measurements
    cpu_util = []

    #do multiple experiments if requested
    for nr in range(-pargs.num_warmup_runs, pargs.num_runs):

        # flush the caches
        if drop_fs_cache and (comm_rank == 0):
            print("Dropping caches")
            with open("/proc/sys/vm/drop_caches", "w") as outfile:
                sp.run(["echo", "1"], stdout=outfile)
        
        #sync up
        printr(f"Running iteration {nr}", comm, 0)
        comm.barrier()
        
        #start time
        tstart = time.time()
        it = 0

        # set baseline cpu % interval
        psutil.cpu_percent(interval=None) 

        # do the loop
        for inputs, labels, source in data_loader:
            
            #increase iteration count
            it += 1

            # run model
            outputs, labels = model.run(inputs, labels)
            
            #training?
            if do_train:
                with torch.cuda.stream(model.fw_stream):
                    with amp.autocast(enabled = use_fp16):
                        loss = criterion(outputs, labels, [1., 1., 1.])
                    
	            #BW
                    gscaler.scale(loss).backward()
                    gscaler.step(optimizer)
                    gscaler.update()

            if do_inference:
                with torch.cuda.stream(model.fw_stream):
                    with torch.no_grad():
                        predictions = torch.max(outputs, 1)[1]
            
                #do we want to plot?
                if visualize:
        
                    #extract tensors as numpy arrays
                    datatens = inputs.cpu().detach().numpy()
                    predtens = predictions.cpu().detach().numpy()
                    labeltens = labels.cpu().detach().numpy()
                
                    for i in range(0,len(source)):
                        print("visualizing " + source[i])
                        npypath = source[i]
                        npybase = os.path.basename(npypath)
                        year = npybase[5:9]
                        month = npybase[10:12]
                        day = npybase[13:15]
                        hour = npybase[16:18]
                        
                        viz.plot(os.path.join(predict_dir, os.path.splitext(os.path.basename(npybase))[0]),
                                 "Predicted",
                                 np.squeeze(datatens[i,0,...]),
                                 np.squeeze(predtens[i,...]),
                                 year=year,
                                 month=month,
                                 day=day,
                                 hour=hour)
                    
                        viz.plot(os.path.join(truth_dir, os.path.splitext(os.path.basename(npybase))[0]),
                                 "Ground truth",
                                 np.squeeze(datatens[i,0,...]),
                                 np.squeeze(labeltens[i,...]),
                                 year=year,
                                 month=month,
                                 day=day,
                                 hour=hour)

        # cpu %: measure here so that estimate is more conservative (higher)
        cpu_util.append(psutil.cpu_percent(interval=None))
        
        #sync up
        model.sync()
        comm.barrier()

        # communicate cpu utilization
        cpu_util_arr = np.array(cpu_util, dtype=np.float32)
        cpu_util_arr = np.stack(comm.allgather(cpu_util_arr), axis=0)

        # compute average per rank:
        cpu_util_arr = np.mean(cpu_util_arr, axis=1)
        
        #end time: measure here so that estimate is more conservative (lower)
        tend = time.time()
        printr("inference complete\n", comm, 0)
        printr("total time: {:.2f} seconds for {} samples".format(tend - tstart, it * batch_size), comm, 0)
        printr("iteration time: {:.4f} seconds/sample".format((tend - tstart)/float(it * batch_size)), comm, 0)
        printr("throughput: {:.2f} samples/second".format(float(it * batch_size)/(tend - tstart)), comm, 0)
        data_size, label_size = data_loader.sample_sizes
        sample_size = (data_size + label_size) / 1024 / 1024 / 1024
        printr("bandwidth: {:.2f} GB/s".format(float(it * batch_size * sample_size) / (tend - tstart)), comm, 0)
        printr(f"cpu utilization: {np.mean(cpu_util_arr):.2f}% (min: {np.min(cpu_util_arr):.2f}%, max: {np.max(cpu_util_arr):.2f}%)", comm, 0)
        
        #write results to file
        if (nr >= 0) and (comm_rank == 0):
            mode = ('a' if nr > 0 else 'w+')
            with open(pargs.outputfile, mode) as f:
                f.write("run {}:\n".format(nr + 1))
                f.write("total time: {:.2f} seconds for {} samples\n".format(tend - tstart, it * batch_size))
                f.write("iteration time: {:.4f} seconds/sample\n".format((tend - tstart)/float(it * batch_size)))
                f.write("throughput: {:.2f} samples/second\n".format(float(it * batch_size)/(tend - tstart)))
                f.write("bandwidth: {:.2f} GB/s\n".format(float(it * batch_size * sample_size) / (tend - tstart)))
                f.write("cpu utilization: {:.2f}%\n".format(np.mean(cpu_util_arr)))
                f.write("\n")

    # wait for everyone to finish
    comm.barrier()


if __name__ == "__main__":
    
    main(parser.parse_arguments())
