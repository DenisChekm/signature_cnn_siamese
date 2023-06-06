import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from datetime import datetime

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
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

train_dir = "../input/sign_data/sign_data/train"
train_csv = "../input/sign_data/sign_data/train_data.csv"
test_csv = "../input/sign_data/sign_data/test_data.csv"
test_dir = "../input/sign_data/sign_data/test"


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

    def forward1(self, x):
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
        output1 = self.forward1(input1)
        # forward pass of input 2
        output2 = self.forward1(input2)

        return output1, output2


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def train(epochs_num):
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=3e-4)
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99)
    loss = []

    for epoch in range(0, epochs_num):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 50 == 0:
                print("Epoch {}; Step {}; Current loss {}\n".format(epoch, i, loss_contrastive.item()))
                loss.append(loss_contrastive.item())
        #print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
    return net


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test():
    count = 0
    test_acc = 0
    for i, data in enumerate(test_dataloader, 0):
        x0, x1, label = data
        # concat = torch.cat((x0, x1), 0)
        output1, output2 = model(x0.to(device), x1.to(device))
        eucledian_distance = F.pairwise_distance(output1, output2)

        if label == torch.FloatTensor([[0]]):
            label = "Original Pair Of Signature"
        else:
            label = "Forged Pair Of Signature"

        pred = "Forged Pair Of Signature"
        if eucledian_distance < 0.41:
            pred = "Original Pair Of Signature"

        if label[0] == pred[0]:
            test_acc += 1
        # imshow(torchvision.utils.make_grid(concat))
        print("Predicted Eucledian Distance:-", eucledian_distance.item())
        print("Actual Label:-", label)
        print("Predicted Label: ", pred)
        print("======================================================")
        # count += 1
        # if count == 1000:
        #     break

    test_acc /= len(test_dataloader)
    return test_acc


if __name__ == '__main__':
    df_train = pd.read_csv(train_csv)
    print(df_train.sample(10))
    print('==============')
    df_test = pd.read_csv(test_csv)
    print(df_test.sample(10))
    print('==============')
    print(df_train.shape)
    print('==============')
    print(df_test.shape)
    print('==============')
    print(df_train[4:5])
    print('==============')
    image1_path = os.path.join(train_dir, df_train.iat[4, 0])
    print(image1_path)

    dataset = Dataset(train_dir, train_csv,
                      transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]))
    train_dataloader = DataLoader(dataset,
                                  shuffle=True,
                                  num_workers=1,
                                  batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    # print("Start Time =", current_time)
    # model = train(10)
    # print(f'Test Accuracy: {test_acc:.4f}')
    # now = datetime.now()
    # torch.save(model.state_dict(), "model_2.pt")
    print("Model Saved Successfully")
    # current_time = now.strftime("%H:%M:%S")
    # print("End Time =", current_time)

    # Load the saved model
    model = SiameseNetwork().to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("model_2.pt"))
    else:
        model.load_state_dict(torch.load("model_2.pt", map_location=torch.device('cpu')))

    test_dataset = Dataset(test_dir, test_csv,
                           transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]))

    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=True)

    test_acc = test()
    print(test_acc)
