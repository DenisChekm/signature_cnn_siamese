import numpy as np
import pandas as pd

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torchvision

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

train_dir = "../sign_data/train"
train_csv = "../sign_data/train_data.csv"
test_csv = "../sign_data/test_data.csv"
test_dir = "../sign_data/test"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CPU_CORES_NUM = 4
BATCH_SIZE = 128
NUM_EPOCHS = 10
threshold = 0.41
MODEL_NAME = 'model_from_pdf.pt'

train_losses = []
eval_losses = []
eval_accu = []


class SignatureDataset(Dataset):
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

        # 'L' mode image, that means it is a single channel image - normally interpreted as greyscale.
        img1 = img1.convert('L')
        img2 = img2.convert('L')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([int(self.train_data.iat[idx, 2])], dtype=np.float32))

    def __len__(self):  ## __len__ returns the size of the dataset..
        return len(self.train_data)


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # input 1
        output1 = self.forward_once(input1)
        # input 2
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


def train_valid(train_dataloader, val_dataloader):
    nn_model = SiameseNetwork().to(DEVICE)
    criterion = ContrastiveLoss()
    optimizer = optim.RMSprop(nn_model.parameters(), lr=1e-4, alpha=0.99)

    batch_i_log = 50
    num_train_batches = len(train_dataloader) - 1
    min_valid_loss = np.inf

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = 0.0
        nn_model.train()
        for batch_i, batch_data in enumerate(train_dataloader):
            img0, img1, labels = batch_data
            img0, img1, labels = img0.to(DEVICE), img1.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            output1, output2 = nn_model(img0, img1)

            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_i % batch_i_log == 0:
                print("Train Epoch {}; Step {}/{}; Current loss {:.6f}"
                      .format(epoch, batch_i, num_train_batches, loss.item()))
            elif batch_i == num_train_batches:
                print("Train Epoch {}; Step {}/{}; Current loss {:.6f}"
                      .format(epoch, batch_i, num_train_batches, loss.item()))

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        val_loss = 0.0
        nn_model.eval()
        with torch.no_grad():
            for batch_data in val_dataloader:
                img0, img1, labels = batch_data
                img0, img1, labels = img0.to(DEVICE), img1.to(DEVICE), labels.to(DEVICE)

                output1, output2 = nn_model(img0, img1)
                loss = criterion(output1, output2, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        eval_losses.append(avg_val_loss)
        print("Epoch {}; Avg Train loss={:.6f}; Avg Val loss={:.6f}\n".format(epoch, avg_train_loss, avg_val_loss))

        if min_valid_loss > avg_val_loss:
            print("Val loss decreased ({:.6f}->{:.6f})\t Saving the model\n".format(min_valid_loss, avg_val_loss))
            min_valid_loss = avg_val_loss
            torch.save(nn_model.state_dict(), MODEL_NAME)


def test(test_dataset):
    nn_model = SiameseNetwork().to(DEVICE)
    if torch.cuda.is_available():
        nn_model.load_state_dict(torch.load(MODEL_NAME))
    else:
        nn_model.load_state_dict(torch.load(MODEL_NAME, map_location=torch.device('cpu')))

    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=False)
    correct = 0
    nn_model.eval()

    with torch.no_grad():
        for batch_data in test_loader:
            img0, img1, label = batch_data
            img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)

            output1, output2 = nn_model(img0, img1)
            euclidean_distance = F.pairwise_distance(output1, output2)

            is_label_orig = False
            if label.item() == 0:
                is_label_orig = True

            is_predict_orig = False
            if euclidean_distance < threshold:
                is_predict_orig = True

            if is_label_orig == is_predict_orig:
                correct += 1

    test_acc = 100. * correct / len(test_loader)
    # eval_accu.append(test_acc)
    # print("Test acc {:.4f}%\n".format(test_acc))
    return test_acc


def main():
    # Training settings
    torch.manual_seed(seed=10)

    train_dataset = SignatureDataset(train_dir, train_csv,
                                     transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()]))
    train_dataloader = DataLoader(train_dataset, num_workers=CPU_CORES_NUM, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = SignatureDataset(test_dir, test_csv,
                                    transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()]))
    val_dataloader = DataLoader(test_dataset, num_workers=CPU_CORES_NUM, batch_size=BATCH_SIZE, shuffle=True)

    print(DEVICE)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    train_valid(train_dataloader, val_dataloader)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

    plt.plot(train_losses, '-o')
    plt.plot(eval_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.show()

    # Testing model
    acc = test(test_dataset)
    print(acc)
    # plt.plot(train_accu, '-o')
    # plt.plot(eval_accu, '-o')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(['Valid'])
    # plt.title('Valid Accuracy')
    # plt.show()


if __name__ == '__main__':
    main()
