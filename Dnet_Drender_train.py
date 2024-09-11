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
from p3d_render import p3d_renderer, Linemod_DRender_Datagenerator
from pytorch3d.structures import Meshes, join_meshes_as_batch

from Linemod_tasknet import Linemod_tasknet,Linemod_dataset
import utilities

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def generate_data(n_test):
    img = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/img.pkl"), 'rb'))
    quat = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/quaternion.pkl"), 'rb'))
    classes = pkl.load(open(os.path.abspath("D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/class.pkl"), 'rb'))
    # classes = to_categorical(classes['class'])
    cam_pose = pkl.load(
        open(os.path.abspath(r"D:\HHIproject\DecepetionNetConda\dataset\cropped_linemod\train_rot_mat_transposed.pkl"),
             'rb'))

    cam_pose = cam_pose['train_rot_mat_transposed']
    # zrot = np.array([-1, -1, 1]).reshape(1, 3, 1)
    # cam_pose = cam_pose * zrot
    # cam_pose = np.transpose(cam_pose, (0,2,1))


    bbox = pkl.load(
        open(os.path.abspath(r"D:\HHIproject\DecepetionNetConda\dataset\cropped_linemod\train_bbox.pkl"), 'rb'))

    dmap = pkl.load(open(os.path.abspath(r"D:\HHIproject\DecepetionNetConda\dataset\cropped_linemod\dmap.pkl"), 'rb'))
    dmap = dmap['dmap']
    dmap = np.reshape(dmap, (dmap.shape[0] * dmap.shape[1], 64, 64))

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
    campose_train = cam_pose[train_idx]
    campose_test = cam_pose[test_idx]
    bbox_train = bbox['train_bbox'][train_idx]
    bbox_test = bbox['train_bbox'][test_idx]
    dmap_train = dmap[train_idx]
    dmap_test = dmap[test_idx]

    return img_train, img_test, quat_train,quat_test,classes_train, classes_test,campose_train,campose_test,bbox_train,bbox_test, dmap_train,dmap_test


def train_p1(Dmodel, Tmodel, device, train_loader, optimizerP1, epoch, writer, all_meshes):
    Dmodel.train()
    Tmodel.train(mode=False)
    for param in Dmodel.parameters():
        param.requires_grad = True
    for param in Tmodel.parameters():
        param.requires_grad = False
    correct = 0
    for batch_idx, (img, target1, target2, campose, bbox, dmap) in enumerate(train_loader):
        img, target_classification, target_quat, campose, bbox, dmap = img.to(device), target1.to(device,dtype=torch.int64), target2.to(device), campose.to(device), bbox.to(device), dmap.to(device)
        # mesheslist = []
        #
        # for l in target_classification:
        #     mesheslist.append(all_meshes[l])

        meshes = [all_meshes[target_classification]]
        #textures = [all_textures[target_classification]]
        #textures_list = []
        #
        # for t in target_classification:
        #     textures_list.append(all_textures[t])
        # print(target_classification)
        # print(len(textures_list))



        #batch_meshes = join_meshes_as_batch(meshes)

        #batch_meshes.verts_padded()
        optimizerP1.zero_grad()
        rendering = Dmodel(img, campose, bbox, target_classification,meshes, dmap)
        gr = grad_reverse(rendering)
        #batch_meshes.detach()



        classification, quat = Tmodel(gr)
        classify_loss = F.nll_loss(classification, target_classification)
        quat_loss = F.mse_loss(quat, target_quat)
        loss = classify_loss + 5*quat_loss
        loss.backward()
        optimizerP1.step()
        # if epoch == 1 and batch_idx == 1:
        #     make_dot(model(img), params=dict(list(model.named_parameters()))).render("torchviz", format="png")
        #     writer.add_graph(model, input_to_model=img, verbose=True)
        pred = classification.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target_classification.view_as(pred)).sum().item()

        if batch_idx % 10 == 0 and batch_idx != 0:
            print('Phase 1 Train Epoch: {} [{}/{} ({:.0f}%)]\tClassLoss: {:.6f}\tQuatLoss: {:.6f}\tAcc: {}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), classify_loss.item(), quat_loss.item(),
                       correct / (batch_idx * len(img))))
        if (batch_idx * len(img)) % 1000 == 0:
            n = 7
            plt.figure(figsize=(20, 4))
            img = img[:, :, :, :].cpu().detach().numpy()
            img = np.transpose(img, (0, 2, 3, 1))

            rendering_show = rendering.permute(0, 2, 3, 1)
            decoded_imgs = rendering_show[:, :, :, :].cpu().detach().numpy()

            for i in range(1, n + 1):
                # Display original
                ax = plt.subplot(2, n, i)

                plt.imshow(img[i].reshape(64, 64, 3))


                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Display reconstruction
                ax = plt.subplot(2, n, i + n)

                plt.imshow(decoded_imgs[i].reshape(64, 64, 3))

                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                plt.savefig(str(epoch)+'_'+str(batch_idx))
            ######for debug########
            # print('rendering shape', rendering.shape)
            #
            # print(campose)
            # print(bbox)
            # rendering_show = rendering.permute(0, 2, 3, 1)
            # images = rendering_show[:, :, :, :].cpu().detach().numpy()
            # img = img[:, :, :, :].cpu().detach().numpy()
            # img = np.transpose(img, (0, 2, 3, 1))
            # f, axarr = plt.subplots(6)
            # axarr[0].imshow(images[0])
            # axarr[1].imshow(images[1])
            # axarr[2].imshow(images[2])
            # axarr[3].imshow(img[0])
            # axarr[4].imshow(img[1])
            # axarr[5].imshow(img[2])
            # plt.savefig(str(batch_idx))
            # plt.show()


def train_p2(Dmodel, Tmodel, device, train_loader, optimizerP2, epoch, writer, all_meshes):
    Dmodel.train(mode=False)
    Tmodel.train()
    for param in Tmodel.parameters():
        param.requires_grad = True
    for param in Dmodel.parameters():
        param.requires_grad = False
    correct = 0
    for batch_idx, (img, target1, target2, campose, bbox, dmap) in enumerate(train_loader):
        img, target_classification, target_quat, campose, bbox, dmap = img.to(device), target1.to(device, dtype=torch.int64), target2.to(device), campose.to(device), bbox.to(device), dmap.to(device)
        # mesheslist = []
        #
        # for l in target_classification:
        #     mesheslist.append(all_meshes[l])

        meshes = [all_meshes[target_classification]]


        # batch_meshes = join_meshes_as_batch(meshes)

        #batch_meshes.verts_padded()
        optimizerP2.zero_grad()
        rendering = Dmodel(img, campose, bbox, target_classification,meshes, dmap)

        #batch_meshes.detach()



        classification, quat = Tmodel(rendering)
        classify_loss = F.nll_loss(classification, target_classification)
        quat_loss = F.mse_loss(quat, target_quat)
        loss = classify_loss + 5*quat_loss
        loss.backward()
        optimizerP2.step()
        # if epoch == 1 and batch_idx == 1:
        #     make_dot(model(img), params=dict(list(model.named_parameters()))).render("torchviz", format="png")
        #     writer.add_graph(model, input_to_model=img, verbose=True)
        pred = classification.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target_classification.view_as(pred)).sum().item()

        if batch_idx % 10 == 0 and batch_idx != 0:



            print('Phase 2 Train Epoch: {} [{}/{} ({:.0f}%)]\tClassLoss: {:.6f}\tQuatLoss: {:.6f}\tAcc: {}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), classify_loss.item(),quat_loss.item(), correct / (batch_idx * len(img))))

def train_p3(Tmodel, device, train_loader, optimizerP2, epoch):

    Tmodel.train()
    for param in Tmodel.parameters():
        param.requires_grad = True
    correct = 0
    for batch_idx, (img, target1, target2, campose, bbox, dmap) in enumerate(train_loader):
        img, target_classification, target_quat, campose, bbox , dmap= img.to(device), target1.to(device,dtype=torch.int64), target2.to(device), campose.to(device), bbox.to(device), dmap.to(device)

        optimizerP2.zero_grad()


        classification, quat = Tmodel(img)
        classify_loss = F.nll_loss(classification, target_classification)
        quat_loss = F.mse_loss(quat, target_quat)
        loss = classify_loss + 5*quat_loss
        loss.backward()
        optimizerP2.step()
        # if epoch == 1 and batch_idx == 1:
        #     make_dot(model(img), params=dict(list(model.named_parameters()))).render("torchviz", format="png")
        #     writer.add_graph(model, input_to_model=img, verbose=True)
        pred = classification.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target_classification.view_as(pred)).sum().item()

        if batch_idx % 10 == 0 and batch_idx != 0:



            print('Phase 2 Train Epoch: {} [{}/{} ({:.0f}%)]\tClassLoss: {:.6f}\tQuatLoss: {:.6f}\tAcc: {}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), classify_loss.item(),quat_loss.item(), correct / (batch_idx * len(img))))

def test_with_realLinemod():
    print("test with real linemod")
    device = torch.device("cuda")
    test_img = pkl.load(
        open(os.path.abspath('D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/test_img.pkl'), 'rb'))
    test_quat = pkl.load(
        open(os.path.abspath('D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/test_quaternion.pkl'), 'rb'))
    test_class = pkl.load(
        open(os.path.abspath('D:/HHIproject/DecepetionNetConda/dataset/cropped_linemod/test_class.pkl'), 'rb'))
    test_img = np.transpose(test_img['test_img'], (0, 3, 1, 2))
    test_quat = test_quat['test_quaternion']
    test_class = test_class['test_class']

    real_test_data = Linemod_dataset(test_img, test_class, test_quat)
    real_test_loader = torch.utils.data.DataLoader(real_test_data, batch_size=64, shuffle=True)

    model = Linemod_tasknet().to(device)

    model.load_state_dict(torch.load("Linemod_task_trained_e40.pt"))

    model.eval()
    test_loss = 0
    correct = 0
    mean_angle_loss = 0
    classify_loss = 0
    test_loader = real_test_loader
    with torch.no_grad():
        for data, target1, target2 in test_loader:
            data, target_classification, target_quat = data.to(device), target1.to(device,
                                                                                   dtype=torch.int64), target2.to(
                device)
            classification, quat = model(data)
            classify_loss += F.nll_loss(classification, target_classification, reduction='sum').item()
            quat_loss = F.mse_loss(quat, target_quat, reduction='sum').item()
            print(quat.shape)
            euler_deg = utilities.quat2eulerdegree(quat.cpu().detach().numpy())
            print(euler_deg.shape)
            target_euler_deg = utilities.quat2eulerdegree(target_quat.cpu().detach().numpy())
            mean_angle_loss += F.l1_loss(torch.from_numpy(euler_deg).to(device),
                                         torch.from_numpy(target_euler_deg).to(device), reduction='sum')
            test_loss += classify_loss + quat_loss

            pred = classification.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target_classification.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(mean_angle_loss)
    mean_angle_loss /= 3 * len(test_loader.dataset)
    classify_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f},Average class loss: {:.4f},Average angle loss: {:.4f} Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, classify_loss, mean_angle_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))




def main():

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 32}
    lr = 0.0001
    epochs = 20
    save_model = True
    in_training = True
    resume_training = True


    if use_cuda:
        cuda_kwargs = {'num_workers': 6,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    writer = SummaryWriter()

    img_train, img_test, quat_train,quat_test,classes_train, classes_test,campose_train,campose_test,bbox_train,bbox_test,dmap_train,dmap_test  = generate_data(300)

    print('training and testing data generated')
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    img_test = np.transpose(img_test, (0, 3, 1, 2))

    training_data = Linemod_DRender_Datagenerator(img_train, classes_train, quat_train, campose_train, bbox_train, dmap_train)
    test_data = Linemod_DRender_Datagenerator(img_test, classes_test, quat_test, campose_test, bbox_test,dmap_test)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True)

    meshes, textures = utilities.load_meshes()
    renderer = p3d_renderer()

    Dnet_model = Dnet(renderer=renderer).to(device)
    task_model = Linemod_tasknet().to(device)
    if resume_training:
        Dnet_model.load_state_dict(torch.load("Linemod_Dnet_tex_trained_e20.pt"))
        task_model.load_state_dict(torch.load("Linemod_task_tex_trained_e20.pt"))
    else:
        task_model.load_state_dict(torch.load("Linemod_tasknet.pt"))
        print('taks net loaded')

    optimizer1 = optim.Adam(Dnet_model.parameters(), lr=lr)
    optimizer2 = optim.Adam(task_model.parameters(), lr=lr)

    if in_training:
        for epoch in range(1, 1+epochs):

            train_p1(Dnet_model, task_model, device, train_loader, optimizer1, epoch, writer, meshes)
            train_p2(Dnet_model, task_model, device, train_loader, optimizer2, epoch, writer, meshes)
            train_p3(task_model, device, train_loader, optimizer2, epoch)
            #test(model, device, test_loader, in_training)

        if save_model:
            torch.save(Dnet_model.state_dict(), "Linemod_Dnet_tex_trained_e20.pt")
            torch.save(task_model.state_dict(), "Linemod_task_tex_trained_e20.pt")

            print('model saved')
test_with_realLinemod()
# main()