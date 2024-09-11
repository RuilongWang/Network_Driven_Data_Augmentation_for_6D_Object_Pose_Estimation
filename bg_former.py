import torch
import torch.nn as nn
import torch.nn.functional as F

class bg_former(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, img, mask, bg_color):

        #bg_where = tf.equal(mask, zero)

        #bg_where = torch.where(mask == 0, 1., 0.)
        # bg_where = torch.unsqueeze(bg_where,1)

        zeros = torch.zeros_like(img)
        #ones = torch.ones_like(img)
        bg_where = torch.where(img == zeros, 1., 0.)
        bg_where = bg_where[:,0,:,:]
        bg_where = torch.unsqueeze(bg_where, 1)

        # print('mask', mask.shape)#[8, 64, 64]
        # print('where', bg_where.shape)#[8, 1, 64, 64]
        # print('color', bg_color.shape)
        #print('img', img.shape)#[8, 3, 64, 64]

        #bg_indices = tf.cast(bg_where, tf.float32)

        # tf.print(bg_color)

        bg = torch.multiply(bg_where, bg_color)
        added = bg + img

        return added


class bg_former6D(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, img, mask, bg_color, org_img):

        #bg_where = tf.equal(mask, zero)
        zeros = torch.zeros_like(img)
        bg_where = torch.where(img == zeros, 1., 0.)
        bg_where = bg_where[:, 0, :, :]
        bg_where = torch.unsqueeze(bg_where,1)
        # print('mask', mask.shape)
        # print('where', bg_where.shape)
        # print('color', bg_color.shape)
        # print('img', img.shape)

        #bg_indices = tf.cast(bg_where, tf.float32)

        # tf.print(bg_color)

        bg = torch.multiply(bg_where, bg_color)
        org_bg = torch.multiply(bg_where, org_img)
        added = bg + img + org_bg

        return added

