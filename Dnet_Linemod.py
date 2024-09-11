from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from bg_former import bg_former, bg_former6D
from p3d_render import p3d_renderer, CropImage, ResizeImage
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchviz import make_dot
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch3d.renderer import (
    DirectionalLights,
    BlendParams,
    TexturesVertex,

)

class Dnet(nn.Module):
    def __init__(self, renderer):
        super(Dnet, self).__init__()
        self.cb1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.cb2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )

        self.cb3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.cb4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        ##encoded
        self.cb5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )

        ########decoders#######
        #bg decoder#
        self.bg_convtp1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.bgconv1 =nn.Sequential(
            nn.Conv2d(128+64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.bgconv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.bg_convtp2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.bg_conv3 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.bg_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.bg_add_conv1 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),

        )

        # light decoder###############
        self.light_convtp1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.lightconv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.lightconv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.light_convtp2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.light_conv3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.light_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.light_flatten = nn.Flatten()
        self.light_dropout = nn.Dropout(0.5)
        self.light_add1 = nn.Sequential(
            nn.Linear(64*64*64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()

        )
        self.light_add2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()

        )
        self.light_dir = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.Sigmoid()

        )

        self.light_amb = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.ReLU()#0.8-1

        )
        self.light_diff = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.ReLU()  # 0.8-1

        )
        self.light_spec = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.ReLU()  # 0.8-1

        )



        ##################

        self.tex_color = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.Tanh()

        )

        self.rot_perturbation = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.Tanh()

        )

        self.upsample = nn.Upsample(scale_factor=10, mode='bilinear')
        self.bg_former = bg_former()
        self.renderer = renderer
        self.crop_img = CropImage()
        #self.meshes = meshes
        self.blend_param = BlendParams(0, 1e-4)
        self.lights = DirectionalLights(device=torch.device("cuda:0"))





    def forward(self, img, R, bbox, label, batch_meshes, dmap):
        ####encoder####
        cb1 = self.cb1(img)
        mp1 = F.max_pool2d(cb1, 2)
        cb2 = self.cb2(mp1)
        cb3 = self.cb3(cb2)

        mp2 = F.max_pool2d(cb3, 2)
        cb4 = self.cb4(mp2)
        cb5 = self.cb5(cb4)
        ################
        ###bg_decoder###
        bg_up1 = self.bg_convtp1(cb5)
        bg_merge1 = torch.cat((cb3, bg_up1), dim = 1)
        bg_conv1 = self.bgconv1(bg_merge1)
        bg_conv2 = self.bgconv2(bg_conv1)
        bg_up2 = self.bg_convtp2(bg_conv2)
        bg_merge2 = torch.cat((cb1, bg_up2), dim = 1)
        bg_conv3 = self.bg_conv3(bg_merge2)
        bg_conv4 = self.bg_conv4(bg_conv3)
        bg_add_conv1 = self.bg_add_conv1(bg_conv4)

        #print(bg_add_conv1[0,:,0,0])
        bg_add_conv1 = torch.clamp(bg_add_conv1, 0,1)
        #bg_add_up = self.upsample(bg_add_conv1)
        #print('bg_add_up shape', bg_add_up.shape)
        #self.lights.ambient_color = bg_add_conv1
        T = torch.ones((R.shape[0], 3)).to(torch.device("cuda:0"))
        T = T * torch.tensor(np.array([0.0, 0.0, 400.0])).to(torch.device("cuda:0"))
        # print(R.shape)
        # print(T.shape)

        ######lights decoder############
        light_up1 = self.light_convtp1(cb5)
        light_merge1 = torch.cat((cb3, light_up1), dim=1)
        light_conv1 = self.lightconv1(light_merge1)
        light_conv2 = self.lightconv2(light_conv1)
        light_up2 = self.light_convtp2(light_conv2)
        light_merge2 = torch.cat((cb1, light_up2), dim=1)
        light_conv3 = self.light_conv3(light_merge2)
        light_conv4 = self.light_conv4(light_conv3)
        light_flatten = self.light_flatten(light_conv4)
        light_dropout = self.light_dropout(light_flatten)
        light_add_linear1 = self.light_add1(light_dropout)
        light_add_linear2 = self.light_add2(light_add_linear1)
        light_dir = self.light_dir(light_add_linear2)
        light_amb = self.light_amb(light_add_linear2)
        light_diff = self.light_diff(light_add_linear2)
        light_spec = self.light_spec(light_add_linear2)

        light_dir = torch.clamp(light_dir, 0.8, 1)
        light_amb = torch.clamp(light_amb, 0.8, 1)
        light_diff = torch.clamp(light_diff, 0.8, 1)
        light_spec = torch.clamp(light_spec, 0.8, 1)

        self.lights.direction = light_dir
        self.lights.ambient_color = light_amb
        self.lights.diffuse_color = light_diff
        self.lights.specular_color = light_spec

        tex_color = self.tex_color(light_add_linear2)
        tex_color = torch.clamp(tex_color, -0.1, 0.1)

        rot_perturbation = self.rot_perturbation(light_add_linear2)
        rot_perturbation = torch.clamp(rot_perturbation, -0.02, 0.02) * 3.1415
        #print(rot_perturbation)

        R_euler = matrix_to_euler_angles(R, convention='XYZ')
        #print(R_euler)
        R_euler += rot_perturbation
        #print(R_euler)

        R = euler_angles_to_matrix(R_euler, convention='XYZ')
        #print(rot_perturbation)
        #
        # zrot = torch.tensor([-1, -1, 1], device='cuda').reshape(3, 1)
        # rot_perturbation = rot_perturbation * zrot
        # rot_perturbation = torch.transpose(rot_perturbation, 1, 2)
        #print(rot_perturbation.shape)

        #R += rot_perturbation


        # lmesh = []
        # #print(len(batch_meshes))
        # for i in range(len(batch_meshes)):
        #     tex = textures[i].get_verts_features()[0] + tex_color[i]
        #     #tex = torch.unsqueeze(tex, 0)
        #     mesh = batch_meshes[i]
        #     mesh.textures = TexturesVertex(verts_features=tex)
        #     lmesh.append(mesh)
        # #print(len(lmesh))

        new_batch_meshes = join_meshes_as_batch(batch_meshes)
        tex = new_batch_meshes.textures.verts_features_list()

        new_tex = TexturesVertex(verts_features=[tex[i] + tex_color[i] for i in range(len(tex))])
        new_batch_meshes.textures = new_tex


        #rendered = self.renderer(meshes_world=batch_meshes, R=R, T=T, blend_params=BlendParams(0, 1e-4, bg_add_up))
        rendered = self.renderer(meshes_world=new_batch_meshes, R=R, T=T, lights=self.lights, blend_params=BlendParams(0, 1e-4, [0,0,0]))
        #print(rendered.grad_fn)
        cropped = self.crop_img(rendered, bbox)
        bg_former = self.bg_former(cropped, dmap, bg_add_conv1)





        return bg_former



class Dnet_6D(nn.Module):
    def __init__(self, renderer):
        super(Dnet_6D, self).__init__()
        self.cb1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.cb2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )

        self.cb3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.cb4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        ##encoded
        self.cb5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )

        ########decoders#######
        #bg decoder#
        self.bg_convtp1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.bgconv1 =nn.Sequential(
            nn.Conv2d(128+64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.bgconv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.bg_convtp2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.bg_conv3 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.bg_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.bg_add_conv1 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Tanh(),

        )

        # light decoder###############
        self.light_convtp1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.lightconv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.lightconv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.light_convtp2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.light_conv3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.light_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.light_flatten = nn.Flatten()
        self.light_dropout = nn.Dropout(0.5)
        # self.light_add1 = nn.Sequential(
        #     nn.Linear(120*160*64, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU()
        #
        # )
        self.light_add1 = nn.Sequential(
            nn.Linear(120 * 160 * 64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()

        )
        self.light_add2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()

        )
        self.light_dir = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.Sigmoid()

        )

        self.light_amb = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.ReLU()#0.8-1

        )
        self.light_diff = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.ReLU()  # 0.8-1

        )
        self.light_spec = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.ReLU()  # 0.8-1

        )

        # ##################
        # # tex decoder###############
        # self.tex_convtp1 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        # )
        # self.texconv1 = nn.Sequential(
        #     nn.Conv2d(128 + 64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        # )
        #
        # self.texconv2 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        # )
        #
        # self.tex_convtp2 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        # )
        #
        # self.tex_conv3 = nn.Sequential(
        #     nn.Conv2d(64 + 64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        # )
        # self.tex_conv4 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        # )
        # self.tex_flatten = nn.Flatten()
        # self.tex_dropout = nn.Dropout(0.5)
        #
        # self.tex_add1 = nn.Sequential(
        #     nn.Linear(120 * 160 * 64, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU()
        #
        # )
        # self.tex_add2 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU()
        #
        # )
        self.tex_color = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.Tanh()

        )
        self.rot_perturbation = nn.Sequential(
            nn.Linear(64, 3),
            nn.BatchNorm1d(3),
            nn.Tanh()

        )



        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.bg_former = bg_former6D()
        self.renderer = renderer
        self.resize_img = ResizeImage()
        #self.meshes = meshes
        self.blend_param = BlendParams(0, 1e-4)
        self.lights = DirectionalLights(device=torch.device("cuda:0"))





    def forward(self, img, R, T, batch_meshes, mask):
        ####encoder####
        cb1 = self.cb1(img)
        mp1 = F.max_pool2d(cb1, 2)
        cb2 = self.cb2(mp1)
        cb3 = self.cb3(cb2)

        mp2 = F.max_pool2d(cb3, 2)
        cb4 = self.cb4(mp2)
        cb5 = self.cb5(cb4)
        ################
        ###bg_decoder###
        bg_up1 = self.bg_convtp1(cb5)
        bg_merge1 = torch.cat((cb3, bg_up1), dim = 1)
        bg_conv1 = self.bgconv1(bg_merge1)
        bg_conv2 = self.bgconv2(bg_conv1)
        bg_up2 = self.bg_convtp2(bg_conv2)
        bg_merge2 = torch.cat((cb1, bg_up2), dim = 1)
        bg_conv3 = self.bg_conv3(bg_merge2)
        bg_conv4 = self.bg_conv4(bg_conv3)
        bg_add_conv1 = self.bg_add_conv1(bg_conv4)

        #print(bg_add_conv1[0,:,0,0])

        bg_add_conv1 = torch.clamp(bg_add_conv1, -0.25,0.25)
        #print(bg_add_conv1[0,:,0,0])
        #bg_add_up = self.upsample(bg_add_conv1)
        #print('bg_add_up shape', bg_add_up.shape)
        #self.lights.ambient_color = bg_add_conv1
        # T = torch.ones((R.shape[0], 3)).to(torch.device("cuda:0"))
        # T = T * torch.tensor(np.array([0.0, 0.0, 400.0])).to(torch.device("cuda:0"))
        # print(R.shape)
        # print(T.shape)

        ######lights decoder############
        light_up1 = self.light_convtp1(cb5)
        light_merge1 = torch.cat((cb3, light_up1), dim=1)
        light_conv1 = self.lightconv1(light_merge1)
        light_conv2 = self.lightconv2(light_conv1)
        light_up2 = self.light_convtp2(light_conv2)
        light_merge2 = torch.cat((cb1, light_up2), dim=1)
        light_conv3 = self.light_conv3(light_merge2)
        light_conv4 = self.light_conv4(light_conv3)

        #####light_conv4
        light_flatten = self.light_flatten(light_conv4)
        #print(light_flatten.shape)
        light_dropout = self.light_dropout(light_flatten)
        light_add_linear1 = self.light_add1(light_dropout)
        light_add_linear2 = self.light_add2(light_add_linear1)
        light_dir = self.light_dir(light_add_linear2)
        light_amb = self.light_amb(light_add_linear2)
        light_diff = self.light_diff(light_add_linear2)
        light_spec = self.light_spec(light_add_linear2)

        #light_dir = torch.clamp(light_dir, 0.8, 1)
        light_amb = torch.clamp(light_amb, 0.6, 0.8)
        light_diff = torch.clamp(light_diff, 0.4, 1)
        light_spec = torch.clamp(light_spec, 0.4, 1)

        self.lights.direction = light_dir
        self.lights.ambient_color = light_amb
        self.lights.diffuse_color = light_diff
        self.lights.specular_color = light_spec

        ######tex decoder############
        # tex_up1 = self.tex_convtp1(cb5)
        # tex_merge1 = torch.cat((cb3, tex_up1), dim=1)
        # tex_conv1 = self.texconv1(tex_merge1)
        # tex_conv2 = self.texconv2(tex_conv1)
        # tex_up2 = self.tex_convtp2(tex_conv2)
        # tex_merge2 = torch.cat((cb1, tex_up2), dim=1)
        # tex_conv3 = self.tex_conv3(tex_merge2)
        # tex_conv4 = self.tex_conv4(tex_conv3)
        # tex_flatten = self.tex_flatten(tex_conv4)
        # tex_dropout = self.tex_dropout(tex_flatten)
        # tex_add_linear1 = self.tex_add1(tex_dropout)
        # tex_add_linear2 = self.tex_add2(tex_add_linear1)
        # tex_color = self.tex_color(tex_add_linear2)
        tex_color = self.tex_color(light_add_linear2)
        tex_color = torch.clamp(tex_color, -0.15, 0.15)

        rot_perturbation = self.rot_perturbation(light_add_linear2)
        rot_perturbation = torch.clamp(rot_perturbation, -0.02, 0.02) * 3.1415
        R_euler = matrix_to_euler_angles(R, convention='XYZ')
        R_euler += rot_perturbation
        R = euler_angles_to_matrix(R_euler, convention='XYZ')


        #rendered = self.renderer(meshes_world=batch_meshes, R=R, T=T, blend_params=BlendParams(0, 1e-4, bg_add_up))


        #meshes = batch_meshes.extend(R.shape[0])
        tex = batch_meshes.textures.verts_features_list()

        new_tex = TexturesVertex(verts_features=[tex[i] + tex_color[i] for i in range(len(tex))])
        batch_meshes.textures = new_tex


        # new_tex = tex[0] + tex_color[0]
        # new_tex = torch.unsqueeze(new_tex, 0)
        # batch_meshes.textures = TexturesVertex(verts_features=new_tex)

        rendered = self.renderer(meshes_world=batch_meshes, R=R, T=T, lights=self.lights, blend_params=BlendParams(0, 1e-4, [0,0,0]))
        #print(rendered.grad_fn)
        resized = self.resize_img(rendered)
        bg_former = self.bg_former(resized, mask, bg_add_conv1, img)





        return bg_former