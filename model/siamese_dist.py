import numpy as np

import os

from time import time
from psutil import cpu_count

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Linear, BatchNorm1d, Module, MaxPool2d, PairwiseDistance, \
    Dropout, AdaptiveAvgPool2d
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, Adamax, RMSprop, SGD
from torch.utils.data import DataLoader
# from torchsummary import summary

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset
from utils.preprocess_image import PreprocessImage
from utils.config import Config
from model.loss.euclidian_contrasive_loss import ContrastiveLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = cpu_count(logical=False)

OUTPUT_DIR = "./savedmodels/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
BEST_MODEL = OUTPUT_DIR + 'best_loss.pt'


def get_predictions_by_euclidian_distance(euclidian_distance: torch.Tensor):
    return torch.where(torch.gt(euclidian_distance, Config.THRESHOLD), 1.0, 0.0)  # real = 0.0, forg = 1.0


def get_predicted_rus_label(euclidian_distance: float) -> str:
    return 'Подделанная' if euclidian_distance > Config.THRESHOLD else 'Настоящая'


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
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
    for m in model.modules():
        if isinstance(m, Conv2d):  # (Conv2d, Linear)):
            kaiming_normal_(m.weight, nonlinearity='relu')
            # if m.bias is not None:
            #     constant_(m.bias, 0)
        # elif isinstance(m, BatchNorm2d):
        #     constant_(m.weight, 1)
        #     constant_(m.bias, 0)


class SignatureNet(Module):

    def __init__(self):
        super(SignatureNet, self).__init__()

        # Alexnet
        self.conv = Sequential(
            conv_block(1, 96, 11, stride=4),  # stride=1),
            MaxPool2d(3, 2),
            conv_block(96, 256, 5, stride=1, padding=2),
            MaxPool2d(3, 2),
            conv_block(256, 384, 3, padding=1),
            conv_block(384, 256, 3, padding=1),
            MaxPool2d(3, 2)
        )

        _adaptive_output = 6
        self.adap_avg_pool = AdaptiveAvgPool2d(_adaptive_output)

        self.fc = Sequential(
            linear_block(256 * _adaptive_output * _adaptive_output, 1024),
            Dropout(p=0.5),
            linear_block(1024, 128)
        )

        # Vgg16
        # self.conv = Sequential(
        #     conv_block(1, 64, 3, stride=1, padding=1),
        #     conv_block(64, 64, 3, stride=1, padding=1),
        #     MaxPool2d(2, 2),
        #
        #     conv_block(64, 128, 3, stride=1, padding=1),
        #     conv_block(128, 128, 3, stride=1, padding=1),
        #     MaxPool2d(2, 2),
        #
        #     conv_block(128, 256, 3, stride=1, padding=1),
        #     conv_block(256, 256, 3, stride=1, padding=1),
        #     conv_block(256, 256, 3, stride=1, padding=1),
        #     MaxPool2d(2, 2),
        #
        #     conv_block(256, 512, 3, stride=1, padding=1),
        #     conv_block(512, 512, 3, stride=1, padding=1),
        #     conv_block(512, 512, 3, stride=1, padding=1),
        #     MaxPool2d(2, 2),
        #
        #     conv_block(512, 512, 3, stride=1, padding=1),
        #     conv_block(512, 512, 3, stride=1, padding=1),
        #     conv_block(512, 512, 3, stride=1, padding=1),
        #     MaxPool2d(2, 2)
        # )
        #
        # self.adap_avg_pool = AdaptiveAvgPool2d(7)
        #
        # self.fc = Sequential(
        #     linear_block(512 * 7 * 7, 1024),
        #     Dropout(p=0.5),
        #     linear_block(1024, 128)
        # )

        self.euclidean_distance = PairwiseDistance()

        weights_init(self)

    def forward_once(self, img):
        x = img.view(-1, 1, 155, 220).div(255)
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

    def __train_loop(self, train_loader, loss_function, optimizer, epoch, output_fn):
        targets, predictions = [], []
        losses = AverageMeter()
        batches = len(train_loader)
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
                output_fn(
                    f'Эпоха {epoch} [{batch + 1}/{batches}] Время: {time_since(start, float(batch + 1) / batches)}'
                    f' Ошибка: {losses.val:.5f}({losses.avg:.5f})')

        targets = torch.cat(targets).to('cpu')
        predictions = get_predictions_by_euclidian_distance(torch.cat(predictions)).to('cpu')
        acc = accuracy_score(targets, predictions)
        std = np.std(losses.list)
        return losses.avg, std, acc

    def make_predictions(self, dataloader, loss_fn):
        targets, euclidian_distances = [], []
        losses = AverageMeter()

        self.eval()
        with torch.no_grad():
            for img1, img2, label in dataloader:
                label = label.to(DEVICE)
                targets.append(label)

                euclidian_distance = self(img1.to(DEVICE), img2.to(DEVICE))
                euclidian_distances.append(euclidian_distance)

                loss = loss_fn(euclidian_distance, label)
                losses.update(loss.item(), label.size(0))

        targets = torch.cat(targets).to('cpu')
        predictions = get_predictions_by_euclidian_distance(torch.cat(euclidian_distances)).to('cpu')
        std_loss = np.std(losses.list)
        return targets, predictions, losses.avg, std_loss

    def __validation_loop(self, dataloader, loss_fn):
        targets, predictions, val_loss, std = self.make_predictions(dataloader, loss_fn)
        acc = accuracy_score(targets, predictions)
        return val_loss, std, acc

    def fit(self, batch_size: int, epochs_number: int, print_fn):
        train_dataset = SignatureDataset("train", Config.CANVAS_SIZE)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                                  drop_last=True)

        valid_dataset = SignatureDataset("val", Config.CANVAS_SIZE)
        validation_loader = DataLoader(valid_dataset, batch_size, num_workers=NUM_WORKERS, pin_memory=True,
                                       drop_last=True)

        self.to(DEVICE)

        loss_fn = ContrastiveLoss()
        optim = Adamax(self.parameters())

        avg_train_losses, avg_val_losses, std_train_losses, std_val_losses = [], [], [], []
        train_accuracies, val_accuracies = [], []
        best_loss = np.inf
        early_stop_epoch_count = 0

        for epoch in range(epochs_number):
            start_time = time()
            avg_train_loss, std_train_loss, train_acc = self.__train_loop(train_loader, loss_fn, optim, epoch, print_fn)
            train_loop_time = time() - start_time

            avg_train_losses.append(avg_train_loss)
            std_train_losses.append(std_train_loss)
            train_accuracies.append(train_acc)

            start_time = time()
            avg_val_loss, std_val_loss, val_acc = self.__validation_loop(validation_loader, loss_fn)
            val_loop_time = time() - start_time

            avg_val_losses.append(avg_val_loss)
            std_val_losses.append(std_val_loss)
            val_accuracies.append(val_acc)

            print_fn(
                f'Эпоха {epoch} - время: {train_loop_time:.0f}s - loss: {avg_train_loss:.5f} - std_loss: {std_train_loss:.4f} - acc: {train_acc:.4f}'
                f' - время: {val_loop_time:.0f}s - val_loss {avg_val_loss:.5f} - val_std_loss: {std_val_loss:.4f} - val_acc: {val_acc:.4f}')

            torch.save({'model': self.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_epoch_count = 0
                print_fn(f'Эпоха {epoch} - Сохранена лучшая ошибка: {best_loss:.5f}')
                torch.save({'model': self.state_dict()}, BEST_MODEL)
            else:
                early_stop_epoch_count += 1
                if early_stop_epoch_count == Config.EARLY_STOPPING_EPOCH:
                    print_fn(
                        f"Ранняя остановка. За последние {Config.EARLY_STOPPING_EPOCH} эпох значение ошибки валидации не уменьшилось")
                    break

        self.load_best_model()
        return avg_train_losses, avg_val_losses, train_accuracies, val_accuracies  # std_train_losses, std_val_losses

    def test(self, batch_size: int, output_fn):
        test_dataset = SignatureDataset("test", Config.CANVAS_SIZE)
        test_loader = DataLoader(test_dataset, batch_size, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

        loss_fn = ContrastiveLoss()
        start_time = time()

        trues, predictions, test_loss, losses = self.make_predictions(test_loader, loss_fn)
        report = classification_report(trues, predictions)  # , output_dict=True)
        matrix = confusion_matrix(trues, predictions)

        elapsed_time = time() - start_time
        # output_fn(f'Тест - время: {elapsed_time:.0f}s - avg_loss: {test_loss:.5f} - std_loss: {np.std(losses):.5f} - acc": {report["accuracy"]:.5f}')
        output_fn(f'Тест - время: {elapsed_time:.0f}s - avg_loss: {test_loss:.5f} - std_loss: {np.std(losses):.5f}')
        return report, matrix

    def test_model_by_name(self, model_name: str, batch_size: int, output_fn):
        self.load_model(model_name)  # добавить проверку существования модели
        report, matrix = self.test(batch_size, output_fn)
        return report, matrix

    def test_best_model(self, batch_size: int, print_fn_callback):
        return self.test_model_by_name(BEST_MODEL, batch_size, print_fn_callback)

    def predict(self, image_path1, image_path2):
        img1 = torch.tensor(PreprocessImage.transform_image(image_path1, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)
        img2 = torch.tensor(PreprocessImage.transform_image(image_path2, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)

        self.eval()
        with torch.no_grad():
            euclidian_distance = self(img1, img2).item()
        prediction = get_predicted_rus_label(euclidian_distance)
        return prediction
