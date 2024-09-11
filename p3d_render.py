from __future__ import print_function
import torch
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize
import pickle as pkl
import utilities
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# Data structures and functions for rendering

from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
)
import sys
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print('set device cuda')

class CropImage(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, img, bbox):
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]

        v_total = []
        img = img.permute(0, 3, 1, 2)

        for i in range(img.shape[0]):
            # print(x1[i])
            # print(y1[i])
            # print(x2[i])
            # print(y2[i])

            crop_img = img[i, :3,  y1[i]+82:y2[i]+82, x1[i]:x2[i]]

            resize = transforms.Resize([64,64])
            crop_img = resize(crop_img)
            v_total.append(crop_img)
        v_total = torch.stack(v_total)
        #v_total = v_total.permute(0, 2, 3, 1)
        #print(v_total.shape)
        return v_total

class ResizeImage(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, img):

        v_total = []
        img = img.permute(0, 3, 1, 2)

        for i in range(img.shape[0]):
           # 640,640 -> 480,640
            #print(img[i].shape)
            crop_img = img[i, :3, 82:562, :]

           #480,640 -> 240,320
            resize = transforms.Resize([120,160], interpolation=Image.NEAREST)
            crop_img = resize(crop_img)
            v_total.append(crop_img)
        v_total = torch.stack(v_total)
        #v_total = v_total.permute(0, 2, 3, 1)
        #print(v_total.shape)
        return v_total

class Linemod_DRender_Datagenerator(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img, labels, quat, campose,bbox, dmap):
        'Initialization'
        self.labels = labels
        self.img = img
        self.quat = quat
        #self.meshes = meshes
        self.campose = campose
        self.bbox = bbox
        self.dmap = dmap


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

        campose = torch.from_numpy(self.campose[index]).float()
        bbox = torch.from_numpy(self.bbox[index]).int()
        dmap = torch.from_numpy(self.dmap[index]).float()

        return img, label, quat, campose, bbox, dmap

class Drender(nn.Module):
    def __init__(self, meshes, renderer):
        super().__init__()
        self.meshes = meshes
        self.device = device
        self.renderer = renderer
        self.crop_img = CropImage()

        # Create an optimizable parameter for the light info of the camera.
        self.light_position = nn.Parameter(
            torch.from_numpy(np.array(((0, 0, 1), (1,0,0)), dtype=np.float32)).to(self.device))
        self.lights = DirectionalLights(device=self.device, ambient_color=[[0, 1, 0]], specular_color=[[1, 1, 1]])
        bg = torch.ones((4, 640, 640, 3))
        self.blend_param = BlendParams(0,1e-4,bg)

    def forward(self, R, T, bbox, label):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        mesheslist = []
        for l in label:
            mesheslist.append(self.meshes[l])
        print(len(mesheslist))
        batch_meshes = join_meshes_as_batch(mesheslist)
        # light_color = torch.tensor((1,0,0),(0,1,0))
        self.lights.direction = self.light_position

        image = self.renderer(meshes_world=batch_meshes, R=R, T=T, lights=self.lights, blend_params = self.blend_param)
        cropped = self.crop_img(image, bbox)




        return cropped

def p3d_renderer():
    # fx = 572.4114 / 5
    # fy = 573.5704 / 5
    # px = 325.2611 / 10
    # py = 325 / 10
    fx = 572.4114
    fy = 573.5704
    px = 325.2611
    py = 325.2611
    # py = 242.04899
    f = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)  # dim = (1, 2)
    p = torch.tensor((px, py), dtype=torch.float32).unsqueeze(0)  # dim = (1, 2)
    cameras = PerspectiveCameras(device=device, focal_length=f, principal_point=p,image_size=((640,640),))


    # Change specular color to green and change material shininess
    materials = Materials(
        device=device,

        # specular_color=[[0.0, 1.0, 0.0]],
        shininess=2
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    #bg = torch.rand(4,640,640, 3)
    #blend_param = BlendParams(0,1e-4,bg)
    raster_settings = RasterizationSettings(
        image_size=(640,640),
        blur_radius=0.0,
        faces_per_pixel=1,
        max_faces_per_bin=60000
    )

    #lights = DirectionalLights(device=device, direction=[[1, 1, 1]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            #blend_params=blend_param,
            materials = materials,
            #lights=lights
        )
    )
    return renderer



def test():
    meshes = utilities.load_meshes()

    renderer = p3d_renderer()
    model = Drender(meshes=meshes, renderer=renderer).to(device)
    R = torch.tensor(np.array([-0.52573111, 0.85065081, 0.0, 0.84825128, 0.52424812, -0.07505775, -0.06384793, -0.03946019, -0.99717919]).reshape(1,3,3))
    zrot = torch.tensor(np.array([-1, -1, 1]).reshape(1, 3, 1))
    R = R * zrot
    R = torch.transpose(R, 2, 1)

    # R = torch.cat((R,R),0).to(device)
    R1 = torch.vstack((R, R)).to(device)
    print(R.shape)
    T = torch.tensor(np.array([0.0, 0.0, 400.0]).reshape(1, 3))
    T1 = torch.vstack((T, T)).to(device)
    #T = torch.cat((T, T), 0).to(device)

    bbox = pkl.load(open(os.path.abspath(r"D:\HHIproject\DecepetionNetConda\dataset\cropped_linemod\train_bbox.pkl"), 'rb'))
    bbox = bbox['train_bbox']
    bbox1 = torch.tensor(np.array(bbox[1313*8+1,:]))
    bbox2 = torch.tensor(np.array(bbox[1313*8+1,:]))

    bbox = torch.vstack((bbox1, bbox2)).to(device)
    print(bbox)
    label = torch.reshape(torch.tensor((8,8)), (2,1))
    print(label.shape)

    out = model(R=R1, T=T1, bbox=bbox, label=label)
    print(out.shape)
    images = out[:, :, :, :3].cpu().detach().numpy()
    # image0 = images[0]
    # print(image0.shape)
    # image0 = image0[67-10:67+331+10, 152-10:152+300+10, :]
    # image0 = resize(image0, (64, 64, 3), anti_aliasing=True)

    org = plt.imread(r'D:\HHIproject\DecepetionNetConda\dataset\lm_train\train\000014\cropped_rgb\000071.png')
    f, axarr = plt.subplots(3)
    axarr[0].imshow(images[0])
    axarr[1].imshow(images[1])
    axarr[2].imshow(org)
    plt.show()

def test_renderer():
    renderer = p3d_renderer()
    meshes, texes = utilities.load_meshes()
    # print(texes[0].get_verts_features())
    # lmesh = []
    # for i in range(2):
    #     tex = torch.add(texes[i].get_verts_features()[0], torch.tensor([0.5, 1, 0.1]).to(device))
    #     tex = torch.unsqueeze(tex, 0)
    #     mesh = meshes[i]
    #     mesh.textures = TexturesVertex(verts_features=tex)
    #     lmesh.append(mesh)
    # #meshes = meshes.extend(2)

    #print(meshes.textures.get_verts_features())
    # meshes.textures.change_verts_features(torch.tensor([0.5, 1, 0.1]).to(device))

    meshes = join_meshes_as_batch([meshes])
    tex = meshes.textures.verts_features_list()
    new_tex = TexturesVertex(verts_features=[t + torch.tensor([0.5, 1, 0.1]).to(device) for t in tex])
    meshes.textures = new_tex



    R = torch.tensor(np.array(
        [-0.82175528, -0.56984056, 0.0, -0.53710394, 0.77454648, -0.33406154, 0.19036181, -0.27451683, -0.94255127]).reshape(1, 3, 3))
    zrot = torch.tensor(np.array([-1, -1, 1]).reshape(1, 3, 1))
    R = R * zrot
    R = torch.transpose(R, 2, 1)

    # R = torch.cat((R,R),0).to(device)
    R1 = torch.vstack((R, R)).to(device)
    print(R.shape)
    T = torch.tensor(np.array([0.0, 0.0, 400.0]).reshape(1, 3))
    T1 = torch.vstack((T, T)).to(device)



    images = renderer(meshes, R=R1, T=T1)
    #images = images[:, ..., :3].cpu().numpy()
    images = images[:, 80:560, :, :3].cpu().numpy()
    print(images.shape)




    f, axarr = plt.subplots(2)
    axarr[0].imshow(images[0])
    axarr[1].imshow(images[1])
    plt.show()
# test_renderer()





