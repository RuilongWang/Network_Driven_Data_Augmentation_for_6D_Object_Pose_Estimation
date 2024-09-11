import pickle as pkl
import os
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_ply
from plyfile import PlyData, PlyElement
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch3d.renderer import (
    TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_batch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print('set device cuda')

def load_meshes_list():
    DATA_DIR = "./data/linemod_meshes/"
    objlist = [f for f in os.listdir(DATA_DIR)]
    print(objlist)
    mesh_list = []
    for obj in objlist:
        print('loading',obj)
        path = os.path.join(DATA_DIR, obj)
        verts, faces = load_ply(path)
        # Initialize each vertex to be white in color.
        plydata = PlyData.read(path)
        red = plydata.elements[0].data['red'] / 255.0
        print(red.shape)
        green = plydata.elements[0].data['green'] / 255.0
        blue = plydata.elements[0].data['blue'] / 255.0
        rgb = torch.tensor(np.dstack((red, green, blue)), dtype=torch.float32)
        print(rgb.shape)
        textures = TexturesVertex(verts_features=rgb.to(device))
        mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

        mesh_list.append(mesh)

    return mesh_list

def load_meshes():
    DATA_DIR = "./data/linemod_meshes/"
    objlist = [f for f in os.listdir(DATA_DIR)]
    print(objlist)
    verts_list = []
    faces_list = []
    tex_list = []
    for obj in objlist:
        print('loading',obj)
        path = os.path.join(DATA_DIR, obj)
        verts, faces = load_ply(path)
        verts_list.append(verts.to(device))
        faces_list.append(faces.to(device))
        # Initialize each vertex to be white in color.
        plydata = PlyData.read(path)
        red = plydata.elements[0].data['red'] / 255.0
        #print(red.shape)
        green = plydata.elements[0].data['green'] / 255.0
        blue = plydata.elements[0].data['blue'] / 255.0
        rgb = torch.tensor(np.squeeze(np.dstack((red, green, blue))), dtype=torch.float32).to(device)
        print(rgb.shape)
        tex_list.append(rgb)



    textures = TexturesVertex(verts_features=tex_list)
    #
    mesh = Meshes(verts=verts_list, faces=faces_list, textures=textures)
    # mesh = Meshes(verts=verts_list, faces=faces_list)

    #print(tex_list.shape)



    return mesh, textures

def load_one_mesh(index):
    DATA_DIR = "C:/Users/Ruilo/PycharmProjects/Pytorch3D_test\data/linemod_meshes/"
    objlist = [f for f in os.listdir(DATA_DIR)]
    print(objlist)
    verts_list = []
    faces_list = []
    tex_list = []
    obj = objlist[index]

    print('loading',obj)
    path = os.path.join(DATA_DIR, obj)
    verts, faces = load_ply(path)
    verts_list.append(verts.to(device))
    faces_list.append(faces.to(device))
    # Initialize each vertex to be white in color.
    plydata = PlyData.read(path)
    red = plydata.elements[0].data['red'] / 255.0
    #print(red.shape)
    green = plydata.elements[0].data['green'] / 255.0
    blue = plydata.elements[0].data['blue'] / 255.0
    rgb = torch.tensor(np.squeeze(np.dstack((red, green, blue))), dtype=torch.float32).to(device)
    print(rgb.shape)
    tex_list.append(rgb)



    textures = TexturesVertex(verts_features=tex_list)
    #
    mesh = Meshes(verts=verts_list, faces=faces_list, textures=textures)
    #mesh = Meshes(verts=verts_list, faces=faces_list)



    return mesh

def quat2eulerdegree(quat):
    r = R.from_quat(quat)
    euler_deg = r.as_euler('zyx', degrees=True)
    return euler_deg

# meshes = load_meshes()
# test = meshes[2]
# print(len(test))




    # filename14 = os.path.join(DATA_DIR, "obj_000014.ply")
    # filename1 = os.path.join(DATA_DIR, "obj_000001.ply")
    #
    # verts14, faces14 = load_ply(filename14)
    # verts1, faces1 = load_ply(filename1)
    #
    # # Initialize each vertex to be white in color.
    # plydata14 = PlyData.read(filename14)
    # red14 = plydata14.elements[0].data['red'] / 255.0
    # green14 = plydata14.elements[0].data['green'] / 255.0
    # blue14 = plydata14.elements[0].data['blue'] / 255.0
    # rgb14 = torch.tensor(np.dstack((red14, green14, blue14)), dtype=torch.float32)
    #
    # plydata1 = PlyData.read(filename1)
    # red1 = plydata1.elements[0].data['red'] / 255.0
    # green1 = plydata1.elements[0].data['green'] / 255.0
    # blue1 = plydata1.elements[0].data['blue'] / 255.0
    # rgb1 = torch.tensor(np.dstack((red1, green1, blue1)), dtype=torch.float32)
    #
    # # verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    # textures14 = TexturesVertex(verts_features=rgb14.to(device))
    # textures1 = TexturesVertex(verts_features=rgb1.to(device))
    #
    # mesh14 = Meshes(verts=[verts14.to(device)], faces=[faces14.to(device)], textures=textures14)
    # mesh1 = Meshes(verts=[verts1.to(device)], faces=[faces1.to(device)], textures=textures1)
    # mesh_list = [mesh1, mesh14]
    #
    # meshes = join_meshes_as_batch([mesh14, mesh1])
    #return meshes