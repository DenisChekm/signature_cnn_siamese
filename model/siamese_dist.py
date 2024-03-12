import numpy as np

import os
import random
from time import time
from psutil import cpu_count

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Linear, BatchNorm1d, Module, MaxPool2d, PairwiseDistance, \
    Dropout, AdaptiveAvgPool2d
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, Adamax, RMSprop, SGD
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset
from utils.preprocess_image import PreprocessImage
from utils.config import Config
from model.loss.my_contrasive_loss import ContrastiveLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = cpu_count(logical=False)  - 1

OUTPUT_DIR = "./savedmodels/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
BEST_MODEL = OUTPUT_DIR + 'best_loss.pt'

contrastive_loss = ContrastiveLoss()
THRESHOLD = 0.5


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return Sequential(
        Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        BatchNorm2d(out_channels),
        ReLU()
    )


def linear_block(in_features, out_features):
    return Sequential(
        Linear(in_features, out_features),
        BatchNorm1d(out_features),
        ReLU()
    )


def weights_init(model: Module):
    for module in model.modules():
        if isinstance(module, Conv2d):  # (Conv2d, Linear)):
            kaiming_normal_(module.weight, nonlinearity='relu')


class SignatureNet(Module):

    def __init__(self):
        super(SignatureNet, self).__init__()

        self.conv = Sequential(
            conv_block(1, 96, 11, stride=4, padding=2),
            MaxPool2d(3, 2),
            conv_block(96, 128, 5, padding=2),
            MaxPool2d(3, 2),
            conv_block(128, 256, 3, padding=1),
            MaxPool2d(3, 2)
        )
        self.adap_avg_pool = AdaptiveAvgPool2d(3)

        self.fc = Sequential(
            linear_block(256 * 3 * 3, 512),
            Dropout(p=0.4),
            linear_block(512, 128)
        )

        self.euclidean_distance = PairwiseDistance()

    def forward_once(self, img):
        x = img.view(-1, 1, 150, 220).div(255)
        x = self.conv(x)
        x = self.adap_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def forward(self, img1, img2):
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        return self.euclidean_distance(embedding1, embedding2)

    def load_model(self, file_name):
        self.to(DEVICE)
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(file_name)['model'])
        else:
            self.load_state_dict(torch.load(file_name, map_location=torch.device('cpu'))['model'])

    def load_best_model(self):
        self.load_model(BEST_MODEL)

    def __train_loop(self, train_loader, loss_function, optimizer, epoch, print_fn_callback):
        losses = AverageMeter()
        batches = len(train_loader)
        targets, predictions = [], []
        start = time()

        self.train()
        for batch, (img1, img2, labels) in enumerate(train_loader):
            labels = labels.to(DEVICE)
            targets.append(labels)

            euclidian_dist = self(img1.to(DEVICE), img2.to(DEVICE))
            predictions.append(euclidian_dist)
            loss = loss_function(euclidian_dist, labels)
            losses.update(loss.item(), labels.size(0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch + 1) % Config.PRINT_FREQ == 0:
                print_fn_callback(
                    f'Эпоха {epoch} [{batch + 1}/{batches}] Время: {time_since(start, float(batch + 1) / batches)}'
                    f' Ошибка: {losses.val:.5f}({losses.avg:.5f})')

        targets = torch.cat(targets)
        predictions = torch.cat(predictions)
        predictions = torch.where(torch.gt(predictions, THRESHOLD), 1.0, 0.0)
        targets, predictions = targets.to('cpu'), predictions.to('cpu')
        acc = accuracy_score(targets, predictions)
        std = np.std(losses.list)
        return losses.avg, std, acc

    def make_predictions(self, dataloader, loss_fn):
        targets, predictions = [], []
        losses = AverageMeter()

        self.eval()
        with torch.no_grad():
            for img1, img2, label in dataloader:
                label = label.to(DEVICE)
                targets.append(label)

                euclidian_distance = self(img1.to(DEVICE), img2.to(DEVICE))
                predictions.append(euclidian_distance)

                loss = loss_fn(euclidian_distance, label)
                losses.update(loss.item(), label.size(0))

        targets = torch.cat(targets)
        predictions = torch.cat(predictions)
        predictions = torch.where(torch.gt(predictions, THRESHOLD), 1.0, 0.0)
        loss /= len(dataloader)
        std = np.std(losses.list)
        return targets, predictions, losses.avg, std

    def __validation_loop(self, dataloader, loss_fn):
        trues, predictions, val_loss, std = self.make_predictions(dataloader, loss_fn)
        trues, predictions = trues.to('cpu'), predictions.to('cpu')
        acc = accuracy_score(trues, predictions)
        return val_loss, std, acc

    def fit(self, batch_size: int, epochs_number: int, print_fn):
        seed_torch(seed=Config.SEED)

        train_dataset = SignatureDataset("train", Config.CANVAS_SIZE, dim=(256, 256))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, drop_last=True)

        validation_dataset = SignatureDataset("val", Config.CANVAS_SIZE, dim=(256, 256))
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=NUM_WORKERS,
                                       pin_memory=True, drop_last=True)

        self.to(DEVICE)
        weights_init(self)

        # loss_function = ContrastiveLoss()
        optim = Adamax(self.parameters())

        best_loss = np.inf
        avg_train_losses, avg_val_losses, = [], []
        std_train_losses, std_val_losses, = [], []
        acc_train_list, acc_val_list = [], []
        early_stop_epoch_count = 0

        for epoch in range(epochs_number):
            start_time = time()
            avg_train_loss, std_train_loss, train_acc = self.__train_loop(train_loader, contrastive_loss, optim, epoch,
                                                                          print_fn)
            train_loop_time = time() - start_time

            avg_train_losses.append(avg_train_loss)
            std_train_losses.append(std_train_loss)
            acc_train_list.append(train_acc)

            start_time = time()
            avg_val_loss, std_val_loss, val_acc = self.__validation_loop(validation_loader, contrastive_loss)
            val_loop_time = time() - start_time

            avg_val_losses.append(avg_val_loss)
            std_val_losses.append(std_val_loss)
            acc_val_list.append(val_acc)

            print_fn(
                f'Эпоха {epoch} - время: {train_loop_time:.0f}s - loss: {avg_train_loss:.4f} - std_loss: {std_train_loss:.4f} - acc: {train_acc:.4f}'
                f' - время: {val_loop_time:.0f}s - val_loss {avg_val_loss:.4f} - val_std_loss: {std_val_loss:.4f} - val_acc: {val_acc:.4f}')

            torch.save({'model': self.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_epoch_count = 0
                print_fn(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': self.state_dict()}, BEST_MODEL)
            else:
                early_stop_epoch_count += 1
                if early_stop_epoch_count == Config.EARLY_STOPPING_EPOCH:
                    print_fn(
                        f"Ранняя остановка. За последние {Config.EARLY_STOPPING_EPOCH} эпох значение ошибки валидации не уменьшилось")
                    break

        self.load_best_model()
        return avg_train_losses, avg_val_losses, acc_train_list, acc_val_list  #std_train_losses, std_val_losses

    def test(self, batch_size: int, print_fn):
        test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)

        start_time = time()
        trues, predictions, test_loss, losses = self.make_predictions(test_loader, contrastive_loss)
        trues, predictions = trues.to('cpu'), predictions.to('cpu')
        report = classification_report(trues, predictions, output_dict=True)
        matrix = confusion_matrix(trues, predictions)
        elapsed_time = time() - start_time
        print_fn(
            f'Тест - время: {elapsed_time:.0f}s - avg_loss: {test_loss:.5f} - std_loss: {np.std(losses):.5f} - acc": {report["accuracy"]:.5f}')
        return report, matrix

    def test_model_by_name(self, model_name, batch_size: int, print_fn):
        self.load_model(model_name)
        test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)

        start_time = time()
        trues, predictions, test_loss, losses = self.make_predictions(test_loader, contrastive_loss)
        trues, predictions = trues.to('cpu'), predictions.to('cpu')
        report = classification_report(trues, predictions)
        matrix = confusion_matrix(trues, predictions)
        elapsed_time = time() - start_time
        print_fn(f'Тест - время: {elapsed_time:.0f}s - avg_loss: {test_loss:.5f} - std_loss: {np.std(losses):.5f}')
        return report, matrix

    def test_best_model(self, batch_size: int, print_fn_callback):
        return self.test_model_by_name(BEST_MODEL, batch_size, print_fn_callback)

    def predict(self, image_path1, image_path2):
        img1 = torch.tensor(PreprocessImage.transform_image(image_path1, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)
        img2 = torch.tensor(PreprocessImage.transform_image(image_path2, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)

        self.eval()
        with torch.no_grad():
            euclidian_distance = self(img1, img2).item()
        prediction = 'Подделанная' if euclidian_distance > THRESHOLD else 'Настоящая'
        return prediction
