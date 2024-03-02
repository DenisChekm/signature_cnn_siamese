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

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay, \
    ConfusionMatrixDisplay

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset
from utils.preprocess_image import PreprocessImage
from utils.config import Config
from model.loss.my_contrasive_loss import MyContrastiveLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "C:/Users/denle/PycharmProjects/signature_cnn_siamese/savedmodels/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

BEST_MODEL = OUTPUT_DIR + 'best_loss.pt'
label_dict = {1.0: 'Подделанная', 0.0: 'Настоящая'}


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
        if isinstance(module, Conv2d):
            kaiming_normal_(module.weight, nonlinearity='relu')


class SignatureNet(Module):
    def __init__(self):
        super(SignatureNet, self).__init__()

        self.conv = Sequential(
            conv_block(1, 64, 11, stride=4, padding=0),
            # conv_block(1, 96, 5, stride=4),
            MaxPool2d(3, 2),

            conv_block(64, 128, 5, padding=2),
            MaxPool2d(3, 2),

            conv_block(128, 256, 3, padding=1),
            # conv_block(384, 384, 3, padding=1),
            # conv_block(384, 256, 3, padding=1),
            MaxPool2d(3, 2)
        )

        self.adap_avg_pool = AdaptiveAvgPool2d(3)

        self.fc = Sequential(
            linear_block(256 * 3 * 3, 512),
            Dropout(p=0.4),
            linear_block(512, 256)
        )

        self.euclidean_distance = PairwiseDistance()

    def forward_once(self, img):
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        x = img.view(-1, 1, 150, 220).div(255)
        x = self.adap_avg_pool(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def forward(self, img1, img2):
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        return self.euclidean_distance(embedding1, embedding2)

    @staticmethod
    def load_model(file_name):
        model = SignatureNet().to(DEVICE)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(file_name)['model'])
        else:
            model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu'))['model'])
        return model  # TO DO move .to(DEVICE) here

    @staticmethod
    def load_best_model():
        return SignatureNet.load_model(BEST_MODEL)

    def __train_loop(self, train_loader, loss_function, optimizer, epoch, print_fn_callback):
        losses = AverageMeter()
        batches = len(train_loader)
        start = time()

        self.train()
        for batch, (img1, img2, labels) in enumerate(train_loader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            labels = labels.to(DEVICE)

            euclidian_dist = self(img1, img2)
            loss = loss_function(euclidian_dist, labels)
            losses.update(loss.item(), labels.size(0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch + 1) % Config.PRINT_FREQ == 0:
                print_fn_callback(
                    f'Epoch [{epoch}][{batch + 1}/{batches}] Elapsed: {time_since(start, float(batch + 1) / batches)}'
                    f' Loss: {losses.val:.4f}({losses.avg:.4f})')

        return losses.avg, losses.list

    def make_predictions(self, dataloader, loss_fn):
        predictions, targets = [], []
        loss = 0

        self.eval()
        with torch.no_grad():
            for img1, img2, label in dataloader:
                img1 = img1.to(DEVICE)
                img2 = img2.to(DEVICE)
                label = label.to(DEVICE)

                targets.append(label)
                euclidian_distance = self(img1, img2)
                loss += loss_fn(euclidian_distance, label).item()
                predictions.append(euclidian_distance)

        targets = torch.cat(targets)
        predictions = torch.where(torch.gt(torch.cat(predictions), 1), 1, 0)  # gr( ,margin)
        loss /= len(dataloader)
        return targets, predictions, loss

    def __validation_loop(self, dataloader, loss_fn):
        start_time = time()
        trues, predictions, val_loss = self.make_predictions(dataloader, loss_fn)
        acc = accuracy_score(trues, predictions)
        elapsed_time = time() - start_time
        return val_loss, acc, elapsed_time

    def fit(self, batch_size: int, epochs_number: int, print_fn_callback):
        seed_torch(seed=Config.SEED)
        num_workers = cpu_count(logical=False)  # - 1

        train_dataset = SignatureDataset("train", Config.CANVAS_SIZE, dim=(256, 256))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)

        validation_dataset = SignatureDataset("val", Config.CANVAS_SIZE, dim=(256, 256))
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                       num_workers=num_workers, pin_memory=True, drop_last=True)

        self.to(DEVICE)
        weights_init(self)

        loss_function = MyContrastiveLoss()
        optim = Adamax(self.parameters())

        best_loss = np.inf
        std_train_losses, std_val_losses, = [], []
        avg_train_losses, avg_val_losses, = [], []
        early_stop_epoch_count = 0

        for epoch in range(epochs_number):
            start_time = time()
            avg_train_loss, losses_list = self.__train_loop(train_loader, loss_function, optim, epoch,
                                                            print_fn_callback)
            elapsed = time() - start_time

            avg_train_losses.append(avg_train_loss)
            std = np.std(losses_list)
            std_train_losses.append(std)

            val_loss, acc, val_time = self.__validation_loop(validation_loader, loss_function)
            avg_val_losses.append(val_loss)
            print_fn_callback(
                f'Epoch {epoch} - Train [avg_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s]; '
                f'Val [avg_loss {val_loss:.6f}, acc {acc:.6f}, time: {val_time:.0f}s]')

            torch.save({'model': self.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')

            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_epoch_count = 0
                print_fn_callback(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': self.state_dict()}, BEST_MODEL)
            else:
                early_stop_epoch_count += 1
                if early_stop_epoch_count == Config.EARLY_STOPPING_EPOCH:
                    print_fn_callback(
                        f"Early stopping. No better loss value for last {Config.EARLY_STOPPING_EPOCH} epochs")
                    break

        report, matrix = test_best_model(batch_size, num_workers, loss_function, print_fn_callback)
        return avg_train_losses, avg_val_losses, report, matrix

    def predict(self, image_path1, image_path2):
        self.eval()

        img1 = torch.tensor(PreprocessImage.transform_image(image_path1, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)
        img2 = torch.tensor(PreprocessImage.transform_image(image_path2, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)

        with torch.no_grad():
            euclidian_distance = self(img1, img2)
            euclidian_distance = euclidian_distance.item()
            euclidian_distance = 1.0 if euclidian_distance > 1 else 0.0  # > margin
            prediction = label_dict[euclidian_distance]
        return prediction


def test_model_by_name(model_name, batch_size: int, num_workers: int, loss_fn, print_fn_callback):
    seed_torch(seed=Config.SEED)
    model = SignatureNet.load_model(model_name)
    test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,  # shuffle убрать?
                             num_workers=num_workers, pin_memory=True, drop_last=True)

    start_time = time()
    trues, predictions, test_loss = model.make_predictions(test_loader, loss_fn)
    report = classification_report(trues, predictions)
    # In the binary case, we can extract true positives, etc. as follows:
    # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    matrix = confusion_matrix(trues, predictions)
    elapsed_time = time() - start_time
    print_fn_callback(f'Test [avg_loss {test_loss:.6f}] - time: {elapsed_time:.0f}s')
    print_fn_callback(report)
    print_fn_callback(matrix)
    return report, matrix


def test_best_model(batch_size: int, num_workers: int, loss_fn, print_fn_callback):
    return test_model_by_name(BEST_MODEL, batch_size, num_workers, loss_fn, print_fn_callback)
