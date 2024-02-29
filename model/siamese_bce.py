import numpy as np

import os
import random
from time import time
from datetime import datetime
from psutil import cpu_count

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Linear, BatchNorm1d, Module, MaxPool2d, Sigmoid, BCELoss, \
    Dropout, AdaptiveAvgPool2d
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, Adamax, RMSprop, SGD
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset
from utils.preprocess_image import PreprocessImage

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay, \
    ConfusionMatrixDisplay

label_dict = {1.0: 'Подделанная', 0.0: 'Настоящая'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "C:/Users/denle/PycharmProjects/signature_cnn_siamese/savedmodels/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

BEST_MODEL = OUTPUT_DIR + 'best_loss.pt'


class Config:
    SEED = 42
    EARLY_STOPPING_EPOCH = 5
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
    return Sequential(
        Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        BatchNorm2d(out_channels),
        ReLU()
    )


def linear_block(in_features, out_features):
    return Sequential(
        Linear(in_features, out_features),
        BatchNorm1d(out_features),
        ReLU()
    )


def output_block(in_features):
    return Sequential(
        Linear(in_features, 1),
        Sigmoid()
    )


def weights_init(model: Module):
    for module in model.modules():
        if isinstance(module, Conv2d):
            kaiming_normal_(module.weight, nonlinearity='relu')


def tp_tn_fp_fn(true_labels, predicted_labels):
    tp = torch.logical_and(true_labels == 0, predicted_labels == 0).sum().item()
    tn = torch.logical_and(true_labels == 1, predicted_labels == 1).sum().item()
    fp = torch.logical_and(true_labels == 0, predicted_labels == 1).sum().item()
    fn = torch.logical_and(true_labels == 1, predicted_labels == 0).sum().item()
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


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

        self.classifier = output_block(256 * 2)

    def __forward_once(self, img):
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        x = img.view(-1, 1, 150, 220).div(255)
        x = self.adap_avg_pool(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def forward(self, img1, img2):
        embedding1 = self.__forward_once(img1)
        embedding2 = self.__forward_once(img2)
        return self.classifier(torch.cat([embedding1, embedding2], dim=1))

    @staticmethod
    def __test_by_model(model, data_loader):
        tp, tn, fp, fn = 0, 0, 0, 0

        model.eval()
        with torch.no_grad():
            for img1, img2, labels in data_loader:
                img1 = img1.to(DEVICE).float()
                img2 = img2.to(DEVICE).float()
                labels = labels.to(DEVICE)

                prediction = model(img1, img2).squeeze()
                tp_tn_fp_fn_batch = tp_tn_fp_fn(labels, torch.round(prediction))
                tp += tp_tn_fp_fn_batch['tp']
                tn += tp_tn_fp_fn_batch['tn']
                fp += tp_tn_fp_fn_batch['fp']
                fn += tp_tn_fp_fn_batch['fn']

        result = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        return result

    @staticmethod
    def load_best_model():
        model = SignatureNet().to(DEVICE)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(BEST_MODEL)['model'])
        else:
            model.load_state_dict(torch.load(BEST_MODEL, map_location=torch.device('cpu'))['model'])
        return model  # TO DO move .to(DEVICE) here


def _make_predictions(model, dataloader, loss_fn):
    predictions, targets = [], []
    loss = 0

    model.eval()
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            label = label.to(DEVICE)

            targets.append(label)
            prediction = model(img1, img2).squeeze()
            loss += loss_fn(prediction, label).item()
            predictions.append(prediction)

    targets = torch.cat(targets)
    predictions = torch.round(torch.cat(predictions))
    loss /= len(dataloader)
    return targets, predictions, loss


def _train_loop(train_loader, model, loss_function, optimizer, epoch, print_fn_callback):
    losses = AverageMeter()
    batches = len(train_loader)
    start = time()

    model.train()
    for batch, (img1, img2, labels) in enumerate(train_loader):
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        labels = labels.to(DEVICE)

        predictions = model(img1, img2).squeeze()
        loss = loss_function(predictions, labels)
        losses.update(loss.item(), labels.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch + 1) % Config.PRINT_FREQ == 0:
            print_fn_callback(
                f'Epoch [{epoch}][{batch + 1}/{batches}] Elapsed: {time_since(start, float(batch + 1) / batches)}'
                f' Loss: {losses.val:.4f}({losses.avg:.4f})')

    return losses.avg, losses.list


def _validation_loop(model, dataloader, loss_fn):
    start_time = time()
    trues, predictions, val_loss = _make_predictions(model, dataloader, loss_fn)
    acc = accuracy_score(trues, predictions)
    elapsed_time = time() - start_time
    return val_loss, acc, elapsed_time


def test_best_model(batch_size: int, num_workers: int, loss_fn, print_fn_callback):
    model = SignatureNet.load_best_model()
    test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)

    start_time = time()
    trues, predictions, test_loss = _make_predictions(model, test_loader, loss_fn)
    report = classification_report(trues, predictions)
    # In the binary case, we can extract true positives, etc. as follows:
    # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    matrix = confusion_matrix(trues, predictions)
    elapsed_time = time() - start_time
    print_fn_callback(f'Test [avg_loss {test_loss:.6f}] - time: {elapsed_time:.0f}s')
    print_fn_callback(report)
    print_fn_callback(matrix)
    return report, matrix


def fit(batch_size: int, epochs_number: int, print_fn_callback):
    # мб проверить сколько процессов занято и часть незанятых подлкючить
    num_workers = cpu_count(logical=False)  # - 1
    seed_torch(seed=Config.SEED)

    train_dataset = SignatureDataset("train", Config.CANVAS_SIZE, dim=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    val_dataset = SignatureDataset("val", Config.CANVAS_SIZE, dim=(256, 256))
    validation_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)

    model = SignatureNet().to(DEVICE)
    weights_init(model)

    loss_function = BCELoss()
    optim = Adamax(model.parameters())

    best_loss = np.inf
    std_train_losses, std_val_losses, = [], []
    avg_train_losses, avg_val_losses, = [], []
    early_stop_epoch_count = 0

    print_fn_callback(f"Начало обучения: {datetime.now():%d.%m.%Y %H:%M:%S%z}")
    for epoch in range(epochs_number):
        start_time = time()
        avg_train_loss, losses_list = _train_loop(train_loader, model, loss_function, optim, epoch, print_fn_callback)
        elapsed = time() - start_time

        avg_train_losses.append(avg_train_loss)
        std = np.std(losses_list)
        std_train_losses.append(std)

        val_loss, acc, val_time = _validation_loop(model, validation_loader, loss_function)
        avg_val_losses.append(val_loss)
        print_fn_callback(
            f'Epoch {epoch} - Train [avg_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s]; '
            f'Val [avg_loss {val_loss:.6f}, acc {acc:.6f}, time: {val_time:.0f}s]')

        torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_epoch_count = 0
            print_fn_callback(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict()}, BEST_MODEL)
        else:
            early_stop_epoch_count += 1
            if early_stop_epoch_count == Config.EARLY_STOPPING_EPOCH:
                print_fn_callback(f"Early stopping. No better loss value for last {Config.EARLY_STOPPING_EPOCH} epochs")
                break

    report, matrix = test_best_model(batch_size, num_workers, loss_function, print_fn_callback)
    return avg_train_losses, avg_val_losses, report, matrix


def predict(model, image_path1, image_path2):
    model.eval()

    img1 = torch.tensor(PreprocessImage.transform_image(image_path1, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)
    img2 = torch.tensor(PreprocessImage.transform_image(image_path2, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)

    with torch.no_grad():
        prediction = model(img1, img2)  # .squeeze()
        prediction = label_dict[torch.round(prediction).item()]

    return prediction
