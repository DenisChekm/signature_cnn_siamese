import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from datetime import datetime

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import random
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

train_dir = "../sign_data/train"
train_csv = "../sign_data/train_data.csv"
test_csv = "../sign_data/test_data.csv"
test_dir = "../sign_data/test"


class Dataset(Dataset):
    # default constuctor for assigning values
    def __init__(self, train_dir=None, train_csv=None, transform=None):
        self.train_dir = train_dir
        self.train_data = pd.read_csv(train_csv)
        self.train_data.columns = ['image1', 'image2', 'class']
        self.transform = transform

    def __getitem__(self, idx):  ## __getitem__ returns a sample data given index, idx=index

        img1_path = os.path.join(self.train_dir, self.train_data.iat[idx, 0])
        img2_path = os.path.join(self.train_dir, self.train_data.iat[idx, 1])

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        img1 = img1.convert(
            'L')  # L mode image, that means it is a single channel image - normally interpreted as greyscale.
        img2 = img2.convert('L')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([int(self.train_data.iat[idx, 2])], dtype=np.float32))

    def __len__(self):  ## __len__ returns the size of the dataset..
        return len(self.train_data)


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 50, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # L1 ImgIn shape=(?, 28, 28, 1)      # (n-f+2*p/s)+1
        #    Conv     -> (?, 24, 24, 50)
        #    Pool     -> (?, 12, 12, 50)

        self.conv2 = nn.Conv2d(50, 60, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # L2 ImgIn shape=(?, 12, 12, 50)
        #    Conv      ->(?, 8, 8, 60)
        #    Pool      ->(?, 4, 4, 60)

        self.conv3 = nn.Conv2d(60, 80, kernel_size=3)
        # L3 ImgIn shape=(?, 4, 4, 60)
        #    Conv      ->(?, 2, 2, 80)

        self.batch_norm1 = nn.BatchNorm2d(50)
        self.batch_norm2 = nn.BatchNorm2d(60)

        #         self.dropout1 = nn.Dropout2d()

        # L4 FC 2*2*80 inputs -> 250 outputs
        self.fc1 = nn.Linear(32000, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        #         print(x.size())
        x = x.view(x.size()[0], -1)
        #         print('Output2')
        #         print(x.size()) #32000 thats why the input of fully connected layer is 32000
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)

        return output1, output2


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def train(nn_model, device, train_dataloader, optimizer, epoch):
    nn_model.train()

    criterion = ContrastiveLoss()

    batch_i_log = 50
    steps_count = len(train_dataloader) - 1
    loss_contrastive = 0
    for batch_i, batch_data in enumerate(train_dataloader):
        img0, img1, label = batch_data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)

        optimizer.zero_grad()
        output1, output2 = nn_model(img0, img1)

        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()

        if batch_i % batch_i_log == 0:
            print("Train Epoch {}; Step {}/{}; Current loss {:.6f}"
                  .format(epoch, batch_i, steps_count, loss_contrastive.item()))
        elif batch_i == steps_count:
            print("Train Epoch {}; Step {}/{}; Current loss {:.6f}"
                  .format(epoch, batch_i, steps_count, loss_contrastive.item()))

    #return loss_contrastive


def test(nn_model, device, test_loader):
    nn_model.eval()

    criterion = ContrastiveLoss()
    correct = 0
    threshold = 0.41
    running_loss = 0.0
    with torch.no_grad():
        for batch_data in test_loader:
            img0, img1, label = batch_data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            output1, output2 = nn_model(img0, img1)
            loss = criterion(output1, output2, label)

            euclidean_distance = F.pairwise_distance(output1, output2)
            if label == torch.FloatTensor([[0]]):
                label = "Orig"
            else:
                label = "Forg"

            predict = "Forg"
            if euclidean_distance < threshold:
                predict = "Orig"

            if label[0] == predict[0]:
                correct += 1

            running_loss += loss.item()

    test_acc = 100 * correct / len(test_loader)
    print('Test Accuracy: {}/{} ({:.4f}%) Test Loss: {}'.format(
        correct, len(test_loader.dataset), test_acc, running_loss))


def main():
    # Training settings
    cpu_cores_number = 4
    epochs_count = 10
    batch_size = 32
    seed = 10
    torch.manual_seed(seed)

    dataset = Dataset(train_dir, train_csv,
                      transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]))
    train_dataloader = DataLoader(dataset,
                                  num_workers=cpu_cores_number,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataset = Dataset(test_dir, test_csv,
                           transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]))
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)



    nn_model = SiameseNetwork().to(device)
    optimizer = optim.RMSprop(nn_model.parameters(), lr=1e-4, alpha=0.99)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    for epoch in range(1, epochs_count + 1):
        train(nn_model, device, train_dataloader, optimizer, epoch)
        test(nn_model, device, test_dataloader)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

    model_name = "model_3.pt"
    torch.save(nn_model.state_dict(), model_name)
    print("Model Saved Successfully")


if __name__ == '__main__':
    main()
