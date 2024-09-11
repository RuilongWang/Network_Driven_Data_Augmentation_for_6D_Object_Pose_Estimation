"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Sergey Zakharov
"""

import os
import torch

from utils import data
from data.train import PatchDatasetRGB2CORR
import cv2
from torch.autograd import Function

import torch.optim as optim
import torch.nn as nn

from Dnet_Linemod import Dnet_6D
from p3d_render import p3d_renderer
import utilities
from data.model import Model
import matplotlib.pyplot as plt
import numpy as np


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


def train(cfg, test_function, create_cnn):
    """
    Training function
    Args:
        cfg: Config file
        test_function: Evaluation function
        create_cnn: Neural network class
    """

    # Setup device
    device_name = data.read_cfg_string(cfg, 'optimization', 'device', default='cpu')
    if device_name == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #torch.cuda.empty_cache()

    # Set model and optimizer
    restore_path = data.read_cfg_string(cfg, 'input', 'restore_net', default='')
    dnet_restore_path = data.read_cfg_string(cfg, 'input', 'restore_Dnet', default='')
    dpod = create_cnn(pretrained=True).to(device)

    renderer = p3d_renderer()
    Dnet_model = Dnet_6D(renderer=renderer).to(device)

    # Criterions
    class_weights = torch.ones(32).to(device)##??##
    class_weights[0] = 0.01
    class_weights_uv = torch.ones(256).to(device)
    class_weights_uv[0] = 0.01

    criterion_uv = nn.CrossEntropyLoss(weight=class_weights_uv)
    criterion_mask_id = nn.CrossEntropyLoss(weight=class_weights)
    dnet_parameters_to_train = list(filter(lambda p: p.requires_grad, Dnet_model.parameters()))
    dpod_parameters_to_train = list(filter(lambda p: p.requires_grad, dpod.parameters()))
    optimizer1 = optim.Adam(dnet_parameters_to_train, weight_decay=0.00004, lr=0.00001)
    optimizer2 = optim.Adam(dpod_parameters_to_train, weight_decay=0.00004, lr=0.00003)

    if len(restore_path) > 0:
        dpod.load_state_dict(torch.load(restore_path), strict=False)
        print('Loaded dpod model:' + restore_path)
    if len(dnet_restore_path) > 0:
        Dnet_model.load_state_dict(torch.load(dnet_restore_path), strict=False)
        print('Loaded dnet model:' + dnet_restore_path)

    # Logs
    log_dir = data.read_cfg_string(cfg, 'test', 'dir', default='log')
    os.makedirs(log_dir, exist_ok=True)

    # Prepare the data
    uv_model = Model()
    uv_model.load("db_mini/models_uv/obj_04.ply")

    batch_size = data.read_cfg_int(cfg, 'train', 'batch_size', default=32)
    cpu_threads = data.read_cfg_int(cfg, 'optimization', 'cpu_threads', default=3)
    trainset = PatchDatasetRGB2CORR(cfg, uv_model)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_threads)
    mesh = utilities.load_one_mesh(2)#4-cat




    # Run the optimizer
    epochs = data.read_cfg_int(cfg, 'train', 'epochs', default=1000)
    for epoch in range(1001, 1001+epochs):

        #Training loop Phase 1
        Dnet_model.train()
        dpod.train(mode=False)
        for param in Dnet_model.parameters():
            param.requires_grad = True
        for param in dpod.parameters():
            param.requires_grad = False

        for i, (rgb, corr, mask, R, T) in enumerate(trainloader):
            rgb, corr, mask, R, T = rgb.to(device), corr.to(device), mask.to(device), R.to(device), T.to(device)
            meshes = mesh.extend(rgb.shape[0])
            # print(pose.shape)

            optimizer1.zero_grad()
            rendering = Dnet_model(rgb, R, T, meshes, mask)

            ###########

            ###############

            gr = grad_reverse(rendering)

            pred_u, pred_v, pred_mask = dpod(gr)
            loss_u = criterion_uv(pred_u, corr[:, 0])
            loss_v = criterion_uv(pred_v, corr[:, 1])
            loss_mask = criterion_mask_id(pred_mask, mask)

            loss = loss_u + loss_v + loss_mask
            loss.backward()
            optimizer1.step()
            loss_corr = loss_u + loss_v

            if (i * len(rgb)) % 80 == 0:
                image = rgb[:, :, :, :].cpu().detach().numpy()
                image = np.transpose(image, (0, 2, 3, 1))

                rendering_show = rendering.permute(0, 2, 3, 1)
                decoded_imgs = rendering_show[:, :, :, :].cpu().detach().numpy()

                plt.imsave('data/temp/img/04orig_' + str(epoch) + '_' + str(i * len(rgb)) + '.png',
                           np.clip(image[0].reshape(120, 160, 3), 0, 1))
                plt.imsave('data/temp/img/04render_' + str(epoch) + '_' + str(i * len(rgb)) + '.png',
                           np.clip(decoded_imgs[0].reshape(120, 160, 3), 0, 1))

            print('Train Epoch: {} Phase 1 [{}/{} ({:.0f}%)]\tLosses: - Corr: {:.6f}, - Mask: {:.6f}'.format(
                epoch, i * len(rgb), len(trainloader.dataset),
                       100. * i / len(trainloader), loss_corr.item(), loss_mask.item()))

        #########################################################################################################
        # Training loop Phase 2
        Dnet_model.train(mode=False)
        dpod.train(mode=True)
        for param in Dnet_model.parameters():
            param.requires_grad = False
        for param in dpod.parameters():
            param.requires_grad = True

        for i, (rgb, corr, mask, R, T) in enumerate(trainloader):
            rgb, corr, mask, R, T = rgb.to(device), corr.to(device), mask.to(device), R.to(device), T.to(device)
            meshes = mesh.extend(rgb.shape[0])
            # print(pose.shape)

            optimizer2.zero_grad()
            rendering = Dnet_model(rgb, R, T, meshes, mask)

            pred_u, pred_v, pred_mask = dpod(rendering)
            loss_u = criterion_uv(pred_u, corr[:, 0])
            loss_v = criterion_uv(pred_v, corr[:, 1])
            loss_mask = criterion_mask_id(pred_mask, mask)

            loss = loss_u + loss_v + loss_mask
            loss.backward()
            optimizer2.step()
            loss_corr = loss_u + loss_v

            print('Train Epoch: {} Phase 2 [{}/{} ({:.0f}%)]\tLosses: - Corr: {:.6f}, - Mask: {:.6f}'.format(
                epoch, i * len(rgb), len(trainloader.dataset),
                       100. * i / len(trainloader), loss_corr.item(), loss_mask.item()))

            # optimizer2.zero_grad()
            # pred_u, pred_v, pred_mask = dpod(rgb)
            # loss_u = criterion_uv(pred_u, corr[:, 0])
            # loss_v = criterion_uv(pred_v, corr[:, 1])
            # loss_mask = criterion_mask_id(pred_mask, mask)
            #
            # loss = loss_u + loss_v + loss_mask
            # loss.backward()
            # optimizer2.step()
            #
            # loss_corr = loss_u + loss_v
            #
            # print('Train Epoch: {} Phase 3 [{}/{} ({:.0f}%)]\tLosses: - Corr: {:.6f}, - Mask: {:.6f}'.format(
            #     epoch, i * len(rgb), len(trainloader.dataset),
            #            100. * i / len(trainloader), loss_corr.item(), loss_mask.item()))

        for i, (rgb, corr, mask, _, _) in enumerate(trainloader):
            # rgb, corr, mask = rgb.to(device), corr.to(device), mask.to(device)
            # optimizer2.zero_grad()
            # pred_u, pred_v, pred_mask = dpod(rgb)
            # loss_u = criterion_uv(pred_u, corr[:, 0])
            # loss_v = criterion_uv(pred_v, corr[:, 1])
            # loss_mask = criterion_mask_id(pred_mask, mask)
            #
            # loss = loss_u + loss_v + loss_mask
            # loss.backward()
            # optimizer2.step()
            #
            # loss_corr = loss_u + loss_v

            loss_corr, loss_mask = dpod.optimize(rgb.to(device),
                                                 corr.to(device),
                                                 mask.to(device))

            print('Train Epoch: {} Phase 3 [{}/{} ({:.0f}%)]\tLosses: - Corr: {:.6f}, - Mask: {:.6f}'.format(
                epoch, i * len(rgb), len(trainloader.dataset),
                       100. * i / len(trainloader), loss_corr.item(), loss_mask.item()))

        # Save networks, analyze performance
        if epoch != 0 and epoch % data.read_cfg_int(cfg, 'train', 'analyze_epoch', default=100) == 0:
            # Store net
            net_dir = os.path.join(log_dir, 'net')
            os.makedirs(net_dir, exist_ok=True)
            torch.save(Dnet_model.state_dict(), os.path.join(net_dir, 'Dnet_model_{}.pt'.format(epoch)))
            torch.save(dpod.state_dict(), os.path.join(net_dir, 'dpod_model_{}.pt'.format(epoch)))
            print('Saved network')

            # Test performance
        if epoch != 0 and epoch % 10 ==0:
            test_function(cfg, cnn=dpod)
