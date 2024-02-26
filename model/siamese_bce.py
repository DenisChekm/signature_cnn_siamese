import numpy as np

import os
import random
import time

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Linear, BatchNorm1d, Module, MaxPool2d, Sigmoid, BCELoss, \
    Dropout
from torch.optim import Adam, Adamax, RMSprop, SGD
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay, \
    ConfusionMatrixDisplay

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# label_dict = {1.0: 'Forged', 0.0: 'Original'}

OUTPUT_DIR = os.getcwd() + '/savedmodels/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class Config:
    SEED = 42
    EPOCHS = 20
    EARLY_STOPPING_EPOCH = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 4
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


def output_block(in_features, out_features):
    return Sequential(
        Linear(in_features, out_features),
        Sigmoid()
    )


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
            conv_block(1, 64, 11, stride=4),
            # conv_block(1, 96, 5, stride=4),
            MaxPool2d(3, 2),

            conv_block(64, 128, 5, padding=2),
            MaxPool2d(3, 2),

            conv_block(128, 256, 3, padding=1),
            # conv_block(384, 384, 3, padding=1),
            # conv_block(384, 256, 3, padding=1),
            MaxPool2d(3, 2)
        )

        self.fc = Sequential(
            linear_block(256 * 3 * 5, 1024),
            Dropout(p=0.4),
            linear_block(1024, 512),
            linear_block(512, 256)
        )

        self.classifier = output_block(256 * 2, 1)

    def forward_once(self, img):
        x = self.conv(img)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)

        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        return self.classifier(torch.cat([embedding1, embedding2], dim=1))

    @staticmethod
    def __train_epoch(train_loader, model, loss_function, optimizer, epoch):
        losses = AverageMeter()

        batches = len(train_loader)
        model.train()
        start = time.time()
        for batch, (img1, img2, labels) in enumerate(train_loader):
            img1 = img1.to(DEVICE).float()
            img2 = img2.to(DEVICE).float()
            labels = labels.to(DEVICE)

            predictions = model(img1, img2).squeeze()
            loss = loss_function(predictions, labels)

            losses.update(loss.item(), labels.size(0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (batch + 1) % Config.PRINT_FREQ == 0:
                print(f'Epoch [{epoch}][{batch + 1}/{batches}] Elapsed: {time_since(start, float(batch + 1) / batches)}'
                      f' Loss: {losses.val:.4f}({losses.avg:.4f})')

        return losses.avg, losses.list

    @staticmethod
    def __test_by_model(model, data_loader, loss_fn):
        test_loss = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        model.eval()
        with torch.no_grad():
            for img1, img2, labels in data_loader:
                img1 = img1.to(DEVICE).float()
                img2 = img2.to(DEVICE).float()
                labels = labels.to(DEVICE)

                prediction = model(img1, img2).squeeze()

                test_loss += loss_fn(prediction, labels).item()
                # acc += (torch.round(prediction) == labels).sum().item()

                tp_tn_fp_fn_batch = tp_tn_fp_fn(labels, torch.round(prediction))
                tp += tp_tn_fp_fn_batch['tp']
                tn += tp_tn_fp_fn_batch['tn']
                fp += tp_tn_fp_fn_batch['fp']
                fn += tp_tn_fp_fn_batch['fn']

        test_loss /= len(data_loader.dataset)
        # acc /= len(test_loader.dataset)

        # result = {'loss': test_loss, 'acc': acc, 'acc_l': acc_lecture, 'p': precision_value, 'recall': recall_value}
        result = {'loss': test_loss}
        return result

    @staticmethod
    def load_best_model():
        model = SignatureNet().to(DEVICE)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(OUTPUT_DIR + 'best_loss.pt')['model'])
        else:
            model.load_state_dict(torch.load(OUTPUT_DIR + 'best_loss.pt', map_location=torch.device('cpu'))['model'])
        return model

    @staticmethod
    def train_with_test():
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

        model = SignatureNet().to(DEVICE)
        print(model)

        loss_function = BCELoss()
        optim = Adamax(model.parameters())

        best_loss = np.inf
        std_losses = []
        early_stop_epoch_count = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()
            avg_train_loss, losses_list = SignatureNet.__train_epoch(train_loader, model, loss_function, optim, epoch)
            elapsed = time.time() - start_time

            std = np.std(losses_list)
            std_losses.append(std)

            val_loss, acc, val_time = validation_loop(model, validation_loader, loss_function)

            print(f'Epoch {epoch} - Train [avg_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s]; '
                  f'Val [avg_loss {val_loss:.6f}, acc {acc:.6f}, time: {val_time:.0f}s]')

            torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')

            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_epoch_count = 0
                print(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': model.state_dict()}, OUTPUT_DIR + 'best_loss.pt')
            else:
                early_stop_epoch_count += 1
                if early_stop_epoch_count == Config.EARLY_STOPPING_EPOCH:
                    print(f"Early stopping. No better loss value for last {Config.EARLY_STOPPING_EPOCH} epochs")
                    break

        test_best_model(loss_function)


def make_predictions(model, dataloader, loss_fn):
    predictions = []
    targets = []
    loss = 0
    model.eval()
    with torch.no_grad():
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


def validation_loop(model, dataloader, loss_fn):
    start_time = time.time()
    trues, predictions, val_loss = make_predictions(model, dataloader, loss_fn)
    acc = accuracy_score(trues, predictions)
    elapsed_time = time.time() - start_time
    return val_loss, acc, elapsed_time


def test_best_model(loss_fn):
    model = SignatureNet.load_best_model()
    test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS,
                             pin_memory=True, drop_last=True)

    start_time = time.time()
    trues, predictions, test_loss = make_predictions(model, test_loader, loss_fn)
    report = classification_report(trues, predictions)
    matrix = confusion_matrix(trues, predictions)
    # test_acc = report['accuracy']
    # p = report['macro avg']['precision']
    # r = report['macro avg']['recall']
    # f1 = report['macro avg']['f1 - score']
    elapsed_time = time.time() - start_time
    print(f'Test [avg_loss {test_loss:.6f}'
          # f' acc {test_acc:.6f}, precision {p:.4f}, recall {r:.4f}, f1 {f1:.4f}'
          f'] - time: {elapsed_time:.0f}s')
    print(report)
    print(matrix)
