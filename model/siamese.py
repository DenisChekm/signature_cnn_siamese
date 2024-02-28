import numpy as np

import os
import random
from time import time
from psutil import cpu_count

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# label_dict = {1.0: 'Forged', 0.0: 'Original'}

OUTPUT_DIR = os.getcwd() + '/savedmodels/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

BEST_MODEL = OUTPUT_DIR + 'best_loss.pt'


class Config:
    SEED = 42
    EPOCHS = 20
    EARLY_STOPPING_EPOCH = 5
    BATCH_SIZE = 32
    NUM_WORKERS = cpu_count(logical=False)  # 4
    PRINT_FREQ = 54  # 50
    CANVAS_SIZE = (952, 1360)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def fully_connected_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU()
    )


def output_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Sigmoid()
    )


def _no_cat_embeddings(embedding1, embedding2):
    return embedding1, embedding2


def _cat_embeddings(embedding1, embedding2):
    return torch.cat([embedding1, embedding2], dim=1)


def weights_init(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            kaiming_normal_(module.weight, nonlinearity='relu')


class SignatureNet(nn.Module):
    def __init__(self, is_distance):
        super(SignatureNet, self).__init__()

        self.conv = nn.Sequential(
            conv_block(1, 64, 11, stride=4),
            nn.MaxPool2d(3, 2),

            conv_block(64, 128, 5, padding=2),
            nn.MaxPool2d(3, 2),

            conv_block(128, 256, 3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        self.adap_avg_pool = nn.AdaptiveAvgPool2d(3)

        self.fc = nn.Sequential(
            fully_connected_block(256 * 3 * 3, 1024),
            nn.Dropout(p=0.4),
            fully_connected_block(1024, 512),
            fully_connected_block(512, 256)
        )

        if is_distance:
            self.classifier = nn.PairwiseDistance()
            self.pass_params = _no_cat_embeddings
        else:
            self.classifier = output_block(256 * 2, 1)
            self.pass_params = _cat_embeddings

    def __forward_once(self, img):
        x = self.conv(img)
        x = self.adap_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def forward(self, img1, img2):
        # Inputs of shape (batch x channels x height x width), and between [0, 1]
        embedding1 = self.__forward_once(img1)
        embedding2 = self.__forward_once(img2)
        output = self.classifier(self.pass_params(embedding1, embedding2))
        return output

    def load_best_model(self):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(BEST_MODEL))
        else:
            self.load_state_dict(torch.load(BEST_MODEL, map_location=torch.device('cpu')))
        return self.to(DEVICE)


def _train_loop(model, train_loader, loss_function, optimizer: torch.optim.Optimizer, epoch):
    losses = AverageMeter()
    batches = len(train_loader)

    model.train()
    start = time()
    for batch, (img1, img2, labels) in enumerate(train_loader):
        predictions = model(img1, img2).squeeze()
        loss = loss_function(predictions, labels)

        losses.update(loss.item(), labels.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch + 1) % Config.PRINT_FREQ == 0:
            print(f'Epoch [{epoch}][{batch + 1}/{batches}] Elapsed: {time_since(start, float(batch + 1) / batches)}'
                  f' Loss: {losses.val:.4f}({losses.avg:.4f})')

    return losses.avg, losses.list


def make_predictions(model, dataloader, loss_fn):
    predictions, targets = [], []
    loss = 0
    model.eval()
    with (torch.no_grad()):
        for img1, img2, label in dataloader:
            img1 = img1.to(DEVICE).float()
            img2 = img2.to(DEVICE).float()
            label = label.to(DEVICE)

            targets.append(label)
            prediction = model(img1, img2).squeeze()
            loss += loss_fn(prediction, label).item()
            predictions.append(prediction)

    targets = torch.cat(targets)
    predictions = torch.round(torch.cat(predictions))
    loss /= len(dataloader)
    return targets, predictions, loss


def _validation_loop(model, dataloader, loss_fn):
    start_time = time()
    trues, predictions, val_loss = make_predictions(model, dataloader, loss_fn)
    acc = accuracy_score(trues, predictions)
    elapsed_time = time() - start_time
    return val_loss, acc, elapsed_time


def _test_best_model(is_distance, loss_fn):
    test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS,
                             pin_memory=True, drop_last=True)

    model = SignatureNet(is_distance).load_best_model()
    start_time = time()
    trues, predictions, test_loss = make_predictions(model, test_loader, loss_fn)
    report = classification_report(trues, predictions)
    matrix = confusion_matrix(trues, predictions)

    elapsed_time = time() - start_time
    print(f'Test [avg_loss {test_loss:.6f}'
          # f' acc {test_acc:.6f}, precision {p:.4f}, recall {r:.4f}, f1 {f1:.4f}'
          f'] - time: {elapsed_time:.0f}s')
    print(report)
    print(matrix)
    # return test_loss, report, matrix, elapsed_time


def fit(is_distance: bool, loss_function, optimizer, epochs_count: int):
    seed_torch(seed=Config.SEED)

    train_dataset = SignatureDataset("train", Config.CANVAS_SIZE, dim=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True)

    val_dataset = SignatureDataset("val", Config.CANVAS_SIZE, dim=(256, 256))
    validation_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                                   num_workers=Config.NUM_WORKERS,
                                   pin_memory=True,
                                   drop_last=True)

    model = SignatureNet(is_distance).to(DEVICE)
    print(model)
    weights_init(model)
    optim = optimizer(model.parameters())

    std_losses = []
    best_loss = np.inf
    early_stop_epoch_count = 0

    for epoch in range(epochs_count):
        start_time = time()
        avg_train_loss, losses_list = _train_loop(model, train_loader, loss_function, optim, epoch)
        elapsed = time() - start_time

        std = np.std(losses_list)
        std_losses.append(std)

        val_loss, acc, val_time = _validation_loop(model, validation_loader, loss_function)

        print(f'Epoch {epoch} - Train [avg_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s]; '
              f'Val [avg_loss {val_loss:.6f}, acc {acc:.6f}, time: {val_time:.0f}s]')

        torch.save({model.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_epoch_count = 0
            print(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({model.state_dict()}, BEST_MODEL)
        else:
            early_stop_epoch_count += 1
            if early_stop_epoch_count == Config.EARLY_STOPPING_EPOCH:
                print(f"Early stopping. No better loss value for last {Config.EARLY_STOPPING_EPOCH} epochs")
                break

    _test_best_model(is_distance, loss_function)
