"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Sergey Zakharov
"""

import os
import glob
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import cv2
import numpy as np

from utils import data
import utils.evaluation as eval
from matplotlib import pyplot as plt


class PatchDatasetRGB2CORR(Dataset):
    def __init__(self, cfg, uv_model):
        self.uv_model = uv_model
        self.cam = data.read_cfg_cam(cfg, 'test', 'intrinsics', default=None)
        self.models = data.read_cfg_string(cfg, 'train', 'models_train', default=None).split(',')#06 object
        self.path = data.read_cfg_string(cfg, 'train', 'path_train_data', default=None)#db_mini/train/uv/
        self.bg_path = data.read_cfg_string(cfg, 'train', 'backgrounds', default=None)

        cv2.setNumThreads(0)

        # Load GTs
        print('Loading yaml...')

        # Find min/max frame numbers
        self.images = []
        for root, directories, filenames in os.walk(self.path):
            if os.path.basename(root) in self.models:
                for filename in filenames:
                    if 'img' in filename:
                        self.images.append(os.path.join(root, filename))

        # Set background path
        if self.bg_path:
            self.backgrounds = glob.glob(os.path.join(self.bg_path, '*.jpg'))

        # Read camera parameters from config
        self.scale = data.read_cfg_float(cfg, 'train', 'image_scale', default=.5)
        self.resolution = data.read_cfg_resolution(cfg, 'train', 'resolution', default=(640, 480))
        self.resolution_scaled = tuple(int(ti * self.scale) for ti in self.resolution)
        self.corr_type = data.read_cfg_string(cfg, 'input', 'corr_type', default='uv')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Create BG data
        np.random.seed()
        bg_id = self.backgrounds[np.random.randint(len(self.backgrounds))]
        rgb_bg = cv2.imread(bg_id)
        # rgb_bg = cv2.resize(rgb_bg, (640, 480)) / 255.0
        rgb_bg = cv2.resize(rgb_bg, (640, 480))

        rgb, corr, mask_id = self.add_interest_object(rgb_bg)
        # plt.figure(1)
        # plt.imshow(rgb)
        # plt.show()
        # cv2.waitKey(3000)
        # plt.imsave('data/temp/img/orig_' + str(idx) + '.png', rgb)

        pose = eval.predict_pose_uv(self.cam, corr.astype(np.uint8), self.uv_model)
        #print(self.uv_model)
        R = pose[:3, :3]
        zrot = np.array([-1, -1, 1]).reshape(3, 1)
        R = R * zrot
        R = np.transpose(R, (1, 0))

        T = pose[:, 3]
        T = T * [-1, -1, 1]

        # Resize to net size
        rgb = cv2.resize(rgb, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        corr = cv2.resize(corr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        mask_id = cv2.resize(mask_id, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        # BGR -> RGB
        rgb = rgb[:, :, ::-1]
        # plt.figure(1)
        # plt.imshow(rgb)
        # plt.show()
        # cv2.waitKey(3000)




        # Transform to torch format
        rgb = F.to_pil_image(rgb.astype(np.uint8))
        #rgb = data.preprocess_rgb(rgb)

        #rgb = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).type(torch.float32)
        rgb = data.preprocess_rgb(rgb)
        corr = torch.from_numpy(corr).permute(2, 0, 1).long()
        mask_id = torch.from_numpy(mask_id).long()
        R = torch.from_numpy(R).type(torch.float32)
        T = torch.from_numpy(T).type(torch.float32)
        # image = rgb.cpu().detach().numpy()
        # image = np.transpose(image, (1,2,0))
        # plt.figure(1)
        # plt.imshow(image)
        # plt.show()
        # cv2.waitKey(3000)




        if self.corr_type == 'uv':
            corr = corr[:2]

        return rgb, corr, mask_id, R, T

    def add_interest_object(self, rgb_bg, number=1):
        """
        Add interest object on the background image
        Args:
            rgb_bg: Background image
            number: Number of added object

        Returns: RGB, Correspondence image, Mask

        """
        rgb_clean = np.zeros((self.resolution[1], self.resolution[0], 3))
        mask_id = np.zeros((self.resolution[1], self.resolution[0], 1))
        corr = np.zeros((self.resolution[1], self.resolution[0], 3))
        dist_ref = 0.6

        for i in range(number):
            idx = np.random.randint(len(self.images))
            patch_rgb = cv2.imread(self.images[idx], -1) / 255
            patch_mask = (patch_rgb[:, :, 3] != 0).astype(np.uint8)
            patch_normals = (cv2.imread(self.images[idx].replace('img', 'nor')) / 255) * 2 - 1
            patch_corr = cv2.imread(self.images[idx].replace('img', 'corr'), 1)



            dist_new = np.random.uniform(low=0.6, high=1.3, size=None)
            scale = dist_ref / dist_new

            patch_rgb = cv2.resize(patch_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            patch_mask = cv2.resize(patch_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)
            patch_normals = cv2.resize(patch_normals, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            patch_a = np.expand_dims(patch_rgb[:, :, 3], axis=2)
            patch_corr = cv2.resize(patch_corr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # plt.figure(1)
            # plt.imshow(patch_rgb)
            # plt.show()
            # cv2.waitKey(30000)

            # normalize normals
            normals_length = np.linalg.norm(patch_normals, axis=2)
            normals_length = np.stack((normals_length, normals_length, normals_length), axis=2)
            patch_normals /= normals_length

            dx_px = np.random.randint(0 + patch_rgb.shape[1] / 2, rgb_bg.shape[1] - patch_rgb.shape[1] / 2) - self.resolution_scaled[0] + 1
            dy_px = np.random.randint(0 + patch_rgb.shape[0] / 2, rgb_bg.shape[0] - patch_rgb.shape[0] / 2) - self.resolution_scaled[1] + 1
            # print(0 + patch_rgb.shape[1] / 2)
            # print(rgb_bg.shape[1] - patch_rgb.shape[1] / 2)
            # print(self.resolution_scaled[0])

            row_top = int(self.resolution_scaled[1] - patch_rgb.shape[0] / 2 + dy_px)
            row_bottom = int(self.resolution_scaled[1] + patch_rgb.shape[0] / 2 + dy_px)
            column_left = int(self.resolution_scaled[0] - patch_rgb.shape[1] / 2 + dx_px)
            column_right = int(self.resolution_scaled[0] + patch_rgb.shape[1] / 2 + dx_px)

            # augment rgb##############
            #patch_rgb_aug = self.add_random_lights(patch_rgb, patch_normals)
            patch_rgb_aug = (patch_rgb * 255).astype(np.uint8)

            # RGB
            # print(corr.shape)
            # print(patch_corr.shape)
            # print(patch_corr[:,-1])
            # old_patch_corr = cv2.imread('db_mini/train/uv/06_old/synth_0000_corr.png', -1)
            # print(old_patch_corr.shape)

            corr[row_top: row_bottom, column_left: column_right][patch_mask] = patch_corr[patch_mask]
            rgb_bg[row_top: row_bottom, column_left:column_right] = patch_rgb_aug[:, :, :3] * patch_a + (1 - patch_a) * rgb_bg[row_top: row_bottom, column_left:column_right]
            #rgb_clean[row_top: row_bottom, column_left:column_right] = patch_rgb[:, :, :3] * patch_a + (1 - patch_a) * rgb_clean[row_top: row_bottom, column_left:column_right]

            # ID mask
            model_name = os.path.basename(os.path.split(self.images[idx])[0])
            label = int(model_name)
            mask_id[row_top: row_bottom, column_left: column_right][patch_mask] = label

            # for i in range(corr.shape[0]):
            #     for j in range(corr.shape[1]):
            #         # if corr[i][j] != 0:
            #         print(corr[i][j])
            # cv2.imshow('1', corr)

            # plt.figure(1)
            # plt.imshow(rgb_bg)
            # plt.show()
            # cv2.waitKey(30000)

        return rgb_bg, corr, mask_id

    def add_random_lights(self, image, normals):
        """
        Add lighting to the RGB image
        Args:
            image: RGB image
            normals: Surface normals image

        Returns: RGB image

        """
        new = np.copy(image)

        light_xyz = np.random.uniform(-1.0, 1.0, (3,))
        light_xyz /= np.linalg.norm(light_xyz)

        diffuse_color = np.random.uniform(.85, 1.0, (3,))
        specular_color = [1, 1, 1]

        ambient = 1 * diffuse_color
        diffuse = np.dot(normals, light_xyz).clip(min=0)[:, :, np.newaxis] * diffuse_color

        camera = [0, 0, -1]
        specularStrength = 0.8
        reflection = light_xyz - 2 * np.dot(normals, light_xyz).clip(min=0)[:, :, np.newaxis] * normals
        specular = np.dot(reflection, camera).clip(min=0)[:, :, np.newaxis] ** 32 * specularStrength * specular_color
        new[:, :, :3] = ((ambient + diffuse + specular) * new[:, :, :3])
        new[new > 1] = 1

        new = (new * 255).astype(np.uint8)

        return new
