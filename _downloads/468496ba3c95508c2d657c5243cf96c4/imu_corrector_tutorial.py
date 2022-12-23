"""
IMU Corrector Tutorial
======================

"""

######################################################################
# Uncomment this if you're using google colab to run this script
#

# !pip install pypose
# !pip install pykitti

######################################################################
# In this tutorial, we will be implementing a simple IMUCorrector
# using ``torch.nn`` modules and ``pypose.IMUPreintegrator``.
# The functionality of our ``IMUCorrector`` is to take an input noisy IMU sensor reading,
# and output the corrected IMU integration result. 
# In some way, ``IMUCorrector`` is an improved ``IMUPreintegrator``.
#
# We will show that, we can combine ``pypose.module.IMUPreintegrator`` into network training smoothly.
# 
# **Skip the first two part if you have seen it in the imu integrator tutorial**
# 

import torch
import pykitti
import numpy as np
import pypose as pp
from torch import nn
import tqdm, argparse
from datetime import datetime
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

######################################################################
# 1. Dataset Defination
# --------------------------
# First we will define the ``KITTI_IMU`` dataset as a ``data.Dataset`` in torch, for easy usage. 
# We're using the ``pykitti`` package.
# This package provides a minimal set of tools for working with the KITTI datasets.
# To access a data sequence, use:
# ::
#
#   dataset = pykitti.raw(root, dataname, drive)
# 
# Some of the data attributes we used below are:
# 
# * ``dataset.timestamps``:    Timestamps are parsed into a list of datetime objects
# * ``dataset.oxts``:          List of OXTS packets and 6-dof poses as named tuples
# 
# For more details about the data format, please refer to their github page 
# `here <https://github.com/utiasSTARS/pykitti#references>`_.
# 
# A sequence will be seperated into many segments. The number of segments is controlled by ``step_size``.
# Each segment of the sequence will return the measurements like ``dt``, ``acc``, and ``gyro``
# for a few frames, defined by duration.
# 

class KITTI_IMU(Data.Dataset):
    def __init__(self, root, dataname, drive, duration=10, step_size=1, mode='train'):
        super().__init__()
        self.duration = duration
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1
        assert mode in ['evaluate', 'train',
                        'test'], "{} mode is not supported.".format(mode)

        self.dt = torch.tensor([datetime.timestamp(self.data.timestamps[i+1]) -
                               datetime.timestamp(self.data.timestamps[i]) 
                               for i in range(self.seq_len)])
        self.gyro = torch.tensor([[self.data.oxts[i].packet.wx, 
                                   self.data.oxts[i].packet.wy,
                                   self.data.oxts[i].packet.wz] 
                                   for i in range(self.seq_len)])
        self.acc = torch.tensor([[self.data.oxts[i].packet.ax, 
                                  self.data.oxts[i].packet.ay,
                                  self.data.oxts[i].packet.az] 
                                  for i in range(self.seq_len)])
        self.gt_rot = pp.euler2SO3(torch.tensor([[self.data.oxts[i].packet.roll, 
                                                  self.data.oxts[i].packet.pitch, 
                                                  self.data.oxts[i].packet.yaw] 
                                                  for i in range(self.seq_len)]))
        self.gt_vel = self.gt_rot @ torch.tensor([[self.data.oxts[i].packet.vf, 
                                                   self.data.oxts[i].packet.vl, 
                                                   self.data.oxts[i].packet.vu] 
                                                   for i in range(self.seq_len)])
        self.gt_pos = torch.tensor(
            np.array([self.data.oxts[i].T_w_imu[0:3, 3] for i in range(self.seq_len)]))

        start_frame = 0
        end_frame = self.seq_len
        if mode == 'train':
            end_frame = np.floor(self.seq_len * 0.5).astype(int)
        elif mode == 'test':
            start_frame = np.floor(self.seq_len * 0.5).astype(int)

        self.index_map = [i for i in range(
            0, end_frame - start_frame - self.duration, step_size)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            'dt': self.dt[frame_id: end_frame_id],
            'acc': self.acc[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id+1: end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id+1: end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id+1: end_frame_id+1],
            'init_pos': self.gt_pos[frame_id][None, ...],
            # TODO: the init rotation might be used in gravity compensation
            'init_rot': self.gt_rot[frame_id: end_frame_id],
            'init_vel': self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.gt_pos[:1],
                'rot': self.gt_rot[:1],
                'vel': self.gt_vel[:1]}

######################################################################
# 2. Utility Functions
# --------------------------
# These are several utility functions. You can skip to the parameter definations
# and come back when necessary.

######################################################################
# ``imu_collate``
# ~~~~~~~~~~~~~~~~
# ``imu_collate`` is used in batch operation, to stack data in multiple frames together.
# 


def imu_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data]).unsqueeze(-1)

    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }

######################################################################
# ``move_to``
# ~~~~~~~~~~~~~~~~
# ``move_to`` used to move different object to CUDA device.
#


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to", obj)

######################################################################
# ``plot_gaussian``
# ~~~~~~~~~~~~~~~~~~
# ``plot_gaussian`` used to plot an ellipse measuring uncertainty, 
# bigger ellipse means bigger uncertainty.
# 


def plot_gaussian(ax, means, covs, color=None, sigma=3):
    ''' Set specific color to show edges, otherwise same with facecolor.'''
    ellipses = []
    for i in range(len(means)):
        eigvals, eigvecs = np.linalg.eig(covs[i])
        axis = np.sqrt(eigvals) * sigma
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * np.arctan(slope) / np.pi
        ellipses.append(Ellipse(means[i, 0:2], axis[0], axis[1], angle=angle))
    ax.add_collection(PatchCollection(ellipses, edgecolors=color, linewidth=1))


######################################################################
# 3. Define IMU Corrector
# -------------------------
# Here we define the ``IMUCorrecter`` module. It has two parts, the ``net`` and the ``imu``,
#   * ``net`` is a network that resemble an autoencoder. 
#     It consists of a sequence of linear layer and activation layer.
#     It will return the IMU measurements correction. Add this correction to the original IMU sensor data,
#     we will get the corrected sensor reading.
#   * ``imu`` is a ``pypose.module.IMUPreintegrator``. Use the corrected sensor reading from previous step as 
#     the input to the ``IMUPreintegrator``, we can get a more accurate IMU integration result.
# 

class IMUCorrector(nn.Module):
    def __init__(self, size_list= [6, 64, 128, 128, 128, 6]):
        super().__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i], size_list[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=False)

    def forward(self, data, init_state):
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        B, F = feature.shape[:2]

        output = self.net(feature.reshape(B*F,6)).reshape(B, F, 6)
        corrected_acc = output[...,:3] + data["acc"]
        corrected_gyro = output[...,3:] + data["gyro"]

        return self.imu(init_state = init_state, 
                        dt = data['dt'], 
                        gyro = corrected_gyro, 
                        acc = corrected_acc, 
                        rot = data['gt_rot'].contiguous())

######################################################################
# 4. Define the Loss Function
# ----------------------------
# The loss function consists of two parts: position loss and rotation loss.
# 
# For position loss, we used ``torch.nn.functional.mse_loss``, which is the mean squared error.
# See the `docs <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_
# for more detail.
# 
# For rotation loss, we first compute pose error between the output rotation and the ground truth rotation, 
# then taking the norm of the lie algebra of the pose error.
#
# Finally, we add the two loss together as our combined loss.
#

def get_loss(inte_state, data):
    pos_loss = torch.nn.functional.mse_loss(inte_state['pos'][:,-1,:], data['gt_pos'][:,-1,:])
    rot_loss = (data['gt_rot'][:,-1,:] * inte_state['rot'][:,-1,:].Inv()).Log().norm()

    loss = pos_loss + rot_loss
    return loss, {'pos_loss': pos_loss, 'rot_loss': rot_loss}


######################################################################
# 5. Define the Training Process
# ------------------------------
# This is the training process, which has three steps:
#   #. **Step 1**: Run forward function, to get the current network output
#   #. **Step 2**: Collect loss, for doing backward in **Step 3**
#   #. **Step 3**: Get gradients and do optimization

def train(network, train_loader, epoch, optimizer, device="cuda:0"):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    network.train()
    running_loss = 0
    t_range = tqdm.tqdm(train_loader)
    for i, data in enumerate(t_range):

        # Step 1: Run forward function
        data = move_to(data, device)
        init_state = {
            "pos": data['init_pos'], 
            "rot": data['init_rot'][:,:1,:],
            "vel": data['init_vel'],}
        state = network(data, init_state)

        # Step 2: Collect loss
        losses, _ = get_loss(state, data)
        running_loss += losses.item()

        # Step 3: Get gradients and do optimization
        t_range.set_description(f'iteration: {i:04d}, losses: {losses:.06f}')
        t_range.refresh()
        losses.backward()
        optimizer.step()

    return (running_loss/i)


######################################################################
# 6. Define the Testing Process
# -----------------------------
# This is the testing process, which has two steps:
#   #. **Step 1**: Run forward function, to get the current network output
#   #. **Step 2**: Collect loss, to evaluate the network performance


def test(network, loader, device = "cuda:0"):
    network.eval()
    with torch.no_grad():
        running_loss = 0
        for i, data in enumerate(tqdm.tqdm(loader)):

            # Step 1: Run forward function
            data = move_to(data, device)
            init_state = {
            "pos": data['init_pos'], 
            "rot": data['init_rot'][:,:1,:],
            "vel": data['init_vel'],}
            state = network(data, init_state)

            # Step 2: Collect loss
            losses, _ = get_loss(state, data)
            running_loss += losses.item()

        print("the running loss of the test set %0.6f"%(running_loss/i))

    return (running_loss/i)

######################################################################
# 7. Define Parameters
# -------------------------
# Here we define all the parameters we will use.
# See the help message for the usage of each parameter.

parser = argparse.ArgumentParser()
parser.add_argument("--device", 
                    type=str, 
                    default='cuda:0', 
                    help="cuda or cpu")
parser.add_argument("--batch-size", 
                    type=int, 
                    default=4, 
                    help="batch size")
parser.add_argument("--max_epoches", 
                    type=int, 
                    default=100, 
                    help="max_epoches")
parser.add_argument("--dataroot", 
                    type=str, 
                    default='../dataset', 
                    help="dataset location downloaded")
parser.add_argument("--dataname", 
                    type=str, 
                    default='2011_09_26', 
                    help="dataset name")
parser.add_argument("--datadrive", 
                    nargs='+', 
                    type=str, 
                    default=[ "0001"], 
                    help="data sequences")
parser.add_argument('--load_ckpt', 
                    default=False, 
                    action="store_true")
args, unknown = parser.parse_known_args(); print(args)

######################################################################
# 8. Define Dataloaders
# -------------------------
# 

train_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0], 
                          duration=10, mode='train')
test_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0], 
                         duration=10, mode='test')
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                               collate_fn=imu_collate, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                              collate_fn=imu_collate, shuffle=False)

######################################################################
# 9. Main Training Loop
# -------------------------
# Here we will run our main training loop. 
# First, like in pytorch, we will define the network, optimizer and scheduler.
# 
# If you are not familiar with the process of training a network,
# we would recommand you reading one of the PyTorch tutorial, like 
# `this <https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>`_.
#
# For each epoch, we run both the training and testing once and collect the running loss.
# We can see from the output message below: the running losss is reducing,
# which means our IMUCorrecter is working. 
# 

network = IMUCorrector().to(args.device)
optimizer = torch.optim.Adam(network.parameters(), lr = 5e-6)  # to use with ViTs
scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.1, patience = 10) # default setup

for epoch_i in range(args.max_epoches):
    train_loss = train(network, train_loader, epoch_i, optimizer, device = args.device)
    test_loss = test(network, test_loader, device = args.device)
    scheduler.step(train_loss)
    print("train loss: %f test loss: %f "%(train_loss, test_loss))

######################################################################
# And that's it. We'are done with our IMUCorrecter tutorials. Thanks for reading. 
# 