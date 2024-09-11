from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchviz import make_dot
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os
import utilities

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

        ## TODO:
        # meshes = join_meshes_as_batch(all_meshes[label])

        return img, label, quat


class Quat_activation(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, input):
        coe = torch.sqrt(1 / torch.sum(torch.pow(input, 2), dim=-1, keepdim=True))
        #print(coe.grad_fn)
        output = torch.multiply(input, coe)
        return output


class Linemod_tasknet(nn.Module):
    def __init__(self):
        super(Linemod_tasknet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10816, 128)
        self.fc2 = nn.Linear(128, 11)
        self.fc3 = nn.Linear(128, 4)
        self.quat_activation = Quat_activation()




    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        classification = self.fc2(x)
        classification_output = F.log_softmax(classification, dim=1)
        quat = self.fc3(x)
        quat_output = self.quat_activation(quat)


        return classification_output, quat_output


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

def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target1, target2) in enumerate(train_loader):
        data, target_classification, target_quat = data.to(device), target1.to(device,dtype=torch.int64), target2.to(device)
        optimizer.zero_grad()
        classification, quat = model(data)
        classify_loss = F.nll_loss(classification, target_classification)
        quat_loss = F.mse_loss(quat, target_quat)
        loss = classify_loss + quat_loss
        loss.backward()
        optimizer.step()
        if epoch == 1 and batch_idx == 1:
            make_dot(model(data), params=dict(list(model.named_parameters()))).render("torchviz", format="png")
            writer.add_graph(model, input_to_model=data, verbose=True)


        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, in_training):
    if not in_training:


        model.load_state_dict(torch.load("Linemod_tasknet.pt"))


    model.eval()
    test_loss = 0
    correct = 0
    mean_angle_loss=0
    classify_loss=0
    with torch.no_grad():
        for data, target1, target2 in test_loader:
            data, target_classification, target_quat = data.to(device), target1.to(device,
                                                                                   dtype=torch.int64), target2.to(
                device)
            classification, quat = model(data)
            classify_loss += F.nll_loss(classification, target_classification, reduction='sum').item()
            quat_loss = F.mse_loss(quat, target_quat, reduction='sum').item()

            euler_deg = utilities.quat2eulerdegree(quat.cpu().detach().numpy())
            print(quat[2])

            target_euler_deg = utilities.quat2eulerdegree(target_quat.cpu().detach().numpy())
            print('target', target_quat[2])
            angle_loss = F.l1_loss(torch.from_numpy(euler_deg[2]).to(device), torch.from_numpy(target_euler_deg[2]).to(device), reduction='sum')
            print(angle_loss)
            mean_angle_loss +=  F.l1_loss(torch.from_numpy(euler_deg).to(device), torch.from_numpy(target_euler_deg).to(device), reduction='sum')
            test_loss += classify_loss + quat_loss

            pred = classification.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target_classification.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(mean_angle_loss)
    mean_angle_loss /=3*len(test_loader.dataset)
    classify_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f},Average class loss: {:.4f},Average angle loss: {:.4f} Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, classify_loss, mean_angle_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #04,Average class loss: 3.3426,Average angle loss: 44.0291 Accuracy: 1341/2200 (61%)
    #,Average class loss: 0.8934,Average angle loss: 41.2071 Accuracy: 1681/2200 (76%) 40e

def main():

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 64}
    lr = 0.0001
    epochs = 50
    save_model = True
    in_training = False


    if use_cuda:
        cuda_kwargs = {'num_workers': 6,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    writer = SummaryWriter()

    img_train, img_test, quat_train, quat_test, classes_train, classes_test = generate_data(100)
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    img_test = np.transpose(img_test, (0, 3, 1, 2))

    training_data = Linemod_dataset(img_train, classes_train, quat_train)
    test_data = Linemod_dataset(img_test, classes_test, quat_test)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    model = Linemod_tasknet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if in_training:
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, writer)
            test(model, device, test_loader, in_training)

        if save_model:
            torch.save(model.state_dict(), "Linemod_tasknet.pt")
    else:
        print("test with real linemod")
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


        test(model, device, real_test_loader, in_training)

def scratch():
    img_train, img_test, quat_train, quat_test, classes_train, classes_test = generate_data(100)
    training_data = Linemod_dataset(img_train, classes_train, quat_train)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=False)

    train_features, train_labels, train_quat = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    quat = train_quat[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    print(f"Label: {quat}")
# main()