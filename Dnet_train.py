from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from bg_former import bg_former
import torch.optim as optim
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchviz import make_dot
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os
from torchsummary import summary
from Dnet_Linemod import Dnet
from Linemod_tasknet import Linemod_tasknet

class Linemod_dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img, labels, quat):
        'Initialization'
        self.labels = labels
        self.img = img
        self.quat = quat

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.img)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample


        # Load data and get label
        img = torch.from_numpy(self.img[index]).float()
        label = self.labels[index]
        quat = torch.from_numpy(self.quat[index]).float()

        return img, label, quat

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forword(self, *input):
        return GradReverse.apply(*input)

def train_p1():
    task_net = Linemod_tasknet().to(device)
    task_net.load_state_dict(torch.load("Linemod_tasknet.pt"))
    d_net = Dnet().to(device)

    d_net_out =



def generate_data(n_test):
    img = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/img.pkl"), 'rb'))
    quat = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/quaternion.pkl"), 'rb'))
    classes = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/class.pkl"), 'rb'))
    # classes = to_categorical(classes['class'])

    np.random.seed(2021)
    test_idx = np.random.choice(np.arange(1313*11), n_test*11 ,replace=False)
    train_idx = []
    for i in range(img['train_img'].shape[0]):
        if i not in test_idx:
            train_idx.append(i)

    img_train = img['train_img'][train_idx]
    img_test = img['train_img'][test_idx]
    quat_train = quat['train_quaternion'][train_idx]
    quat_test = quat['train_quaternion'][test_idx]
    classes_train = classes['class'][train_idx]
    classes_test = classes['class'][test_idx]
    return img_train, img_test, quat_train,quat_test,classes_train, classes_test

def linmod_data(n_test):
    img = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/img.pkl"), 'rb'))
    nmap = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/nmap.pkl"), 'rb'))
    dmap = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/dmap.pkl"), 'rb'))
    quat = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/quaternion.pkl"), 'rb'))
    classes = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/class.pkl"), 'rb'))
    #classes = to_categorical(classes['class'])

    nmap = nmap['nmap']
    nmap = np.reshape(nmap, (nmap.shape[0] * nmap.shape[1],64, 64, 3))
    dmap = dmap['dmap']
    dmap =np.reshape(dmap, (dmap.shape[0] * dmap.shape[1],64, 64))

    np.random.seed(2021)
    test_idx = np.random.choice(np.arange(1313 * 11), n_test * 11, replace=False)
    train_idx = []
    for i in range(img['train_img'].shape[0]):
        if i not in test_idx:
            train_idx.append(i)

    img_train = img['train_img'][train_idx]
    img_test = img['train_img'][test_idx]
    # print(img_train[0,32,32,:])
    nmap_train = nmap[train_idx]
    nmap_test = nmap[test_idx]
    dmap_train = dmap[train_idx]
    dmap_test = dmap[test_idx]

    quat_train = quat['train_quaternion'][train_idx]
    quat_test = quat['train_quaternion'][test_idx]
    classes_train = classes['class'][train_idx]
    classes_test = classes['class'][test_idx]
    # print('nmap train shape',nmap_train.shape)
    # print('dmap train shape',dmap_train.shape)

    return img_train, img_test, nmap_train, nmap_test, dmap_train, dmap_test, quat_train, quat_test, classes_train, classes_test


img_train, img_test, nmap_train, nmap_test, dmap_train, dmap_test, quat_train, quat_test, classes_train, classes_test = linmod_data(100)
img_train = np.expand_dims(img_train[0,:,:,:], 0)
dmap_train = np.expand_dims(dmap_train[0,:,:], 0)
dmap_train = np.expand_dims(dmap_train, 0)

img_train = torch.from_numpy(np.transpose(img_train, (0, 3, 1, 2))).float()
dmap_train = torch.from_numpy(dmap_train).float()
print(img_train.shape)
print(dmap_train.shape)


print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda")

dnet = Dnet().to(device)
task_net = Linemod_tasknet().to(device)
task_net.load_state_dict(torch.load("Linemod_tasknet.pt"))
gr = GRL()




summary(model, [(3, 64, 64),(1,64,64)])
img_train = img_train.to(device)
dmap_train = dmap_train.to(device)
out = model(img_train, dmap_train)
out = out.cpu().detach().numpy()
out = np.squeeze(out)
out = np.transpose(out, (1,2,0))
print(out)
print(out.shape)
plt.imshow(out)
plt.show()
