import pandas as pd
import numpy as np

import os
import random
import time

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Linear, BatchNorm1d, Module, MaxPool2d, Sigmoid, BCELoss
from torch.optim import Adam, Adamax, RMSprop, SGD
from torch.utils.data import DataLoader, random_split

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, zero_one_loss, RocCurveDisplay, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
import seaborn as sns

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_dict = {1.0: 'Forged', 0.0: 'Original'}

# Train data
train_csv = "../sign_data/train_data.csv"
train_dir = "../../sign_data/train"
train = pd.read_csv(train_csv)
train.rename(columns={"1": "label"}, inplace=True)
train["image_real_paths"] = train["033/02_033.png"].apply(lambda x: f"../sign_data/train/{x}")
train["image_forged_paths"] = train["033_forg/03_0203033.PNG"].apply(lambda x: f"../sign_data/train/{x}")

# Test data
test_csv = "../sign_data/test_data.csv"
test_dir = "../../sign_data/test"
test = pd.read_csv(test_csv)
test.rename(columns={"1": "label"}, inplace=True)
test["image_real_paths"] = test["068/09_068.png"].apply(lambda x: f"../sign_data/test/{x}")
test["image_forged_paths"] = test["068_forg/03_0113068.PNG"].apply(lambda x: f"../sign_data/test/{x}")

OUTPUT_DIR = os.getcwd() + '/savedmodels/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class Config:
    SEED = 42
    EPOCHS = 20
    EARLY_STOPPING_EPOCH = 1
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PRINT_FREQ = 50  # 100
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


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


class SignatureNet(Module):
    def __init__(self):
        super(SignatureNet, self).__init__()

        self.conv = Sequential(
            conv_block(1, 96, 11, stride=4),
            # conv_block(1, 96, 5, stride=4),
            MaxPool2d(3, 2),

            conv_block(96, 128, 5, padding=2),
            MaxPool2d(3, 2),

            conv_block(128, 256, 3, padding=1),
            # conv_block(384, 384, 3, padding=1),
            # conv_block(384, 256, 3, padding=1),
            MaxPool2d(3, 2)
        )

        self.fc = Sequential(
            linear_block(256 * 3 * 5, 512),
            linear_block(512, 256)
        )

        self.out = output_block(256 * 2, 1)

    def forward_once(self, img):
        x = self.conv(img)
        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)

        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)

        output = torch.cat([embedding1, embedding2], dim=1)
        return self.out(output)

    @staticmethod
    def __train_epoch(train_loader, model, criterion, optimizer, epoch):
        data_time = AverageMeter()
        losses = AverageMeter()

        batches = len(train_loader)
        model.train()
        start = end = time.time()
        for batch, (img1, img2, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            img1 = img1.to(DEVICE).float()
            img2 = img2.to(DEVICE).float()
            labels = labels.to(DEVICE)

            batch_size = labels.size(0)
            predictions = model(img1, img2).squeeze()
            loss = criterion(predictions, labels)

            losses.update(loss.item(), batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            end = time.time()
            if batch % Config.PRINT_FREQ == 0 or batch == batches - 1:
                print(f'Epoch: [{epoch}][{batch}/{batches}] ', end='')
                print(f'Elapsed: {time_since(start, float(batch + 1) / batches)} ', end='')
                print(f'Loss: {losses.val:.4f}({losses.avg:.4f})')

        return losses.avg, losses.list

    @staticmethod
    def __test_by_model(model, data_loader, loss_fn):
        test_loss = 0
        # acc = 0
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

        acc_lecture = accuracy(tp, tn, fp, fn)
        precision_value = precision(tp, fp)
        recall_value = recall(tp, fn)

        # result = {'loss': test_loss, 'acc': acc, 'acc_l': acc_lecture, 'p': precision_value, 'recall': recall_value}
        result = {'loss': test_loss, 'acc_l': acc_lecture, 'p': precision_value, 'recall': recall_value}
        return result

    @staticmethod
    def load_best_model():
        model = SignatureNet().to(DEVICE)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(OUTPUT_DIR + 'best_loss.pt')['model'])
        else:
            model.load_state_dict(
                torch.load(OUTPUT_DIR + 'best_loss.pt', map_location=torch.device('cpu'))['model'])
        return model

    @staticmethod
    def train_with_test():
        seed_torch(seed=Config.SEED)

        # train_dataset = SignatureDataset(train, Config.CANVAS_SIZE, dim=(256, 256))
        # train_size = int(len(train_dataset) * 0.8)
        # validation_size = len(train_dataset) - train_size
        # train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
        #
        # train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        #                           num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
        #
        # validation_loader = DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        #                                num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
        #
        # model = SignatureNet().to(DEVICE)
        # print(model)  # prin_layers_info(model)
        #
        loss_function = BCELoss()
        # # lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        # optim = Adamax(model.parameters())
        #
        # best_loss = np.inf
        # std_losses = []
        #
        # early_stop_epoch_count = 0
        #
        # for epoch in range(Config.EPOCHS):
        #     start_time = time.time()
        #     avg_train_loss, losses_list = SignatureNet.__train_epoch(train_loader, model, loss_function, optim, epoch)
        #     elapsed = time.time() - start_time
        #
        #     std = np.std(losses_list)
        #     std_losses.append(std)
        #
        #     start_time = time.time()
        #     # val_loss, val_acc = SignatureNet.__test_by_model(model, validation_loader, loss_function)
        #     metrics = SignatureNet.__test_by_model(model, validation_loader, loss_function)
        #     val_time = time.time() - start_time
        #
        #     val_loss = metrics['loss']
        #     # val_acc = metrics['acc']
        #     acc_l = metrics['acc_l']
        #     p = metrics['p']
        #     recall_value = metrics['recall']
        #     print(f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s; '
        #           f'Val [avg_loss {val_loss:.4f},'
        #           # f' acc {val_acc:.4f},'
        #           f' acc_l {acc_l:.4f}, p {p:.4f}, recall {recall_value:.4f}, time: {val_time:.0f}s]')
        #
        #     torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')
        #
        #     if val_loss < best_loss:
        #         best_loss = val_loss
        #         early_stop_epoch_count = 0
        #         print(f'Epoch {epoch} - Save Best Loss: {best_loss:.4f} Model')
        #         torch.save({'model': model.state_dict()}, OUTPUT_DIR + 'best_loss.pt')
        #     else:
        #         early_stop_epoch_count += 1
        #         if early_stop_epoch_count == Config.EARLY_STOPPING_EPOCH:
        #             print(
        #                 f"Early stopping. No better loss value for last {Config.EARLY_STOPPING_EPOCH} epochs")
        #             break

        model = SignatureNet.load_best_model()
        test_dataset = SignatureDataset(test, Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                                 num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
        test_model(model, test_loader, loss_function)
        # start_time = time.time()

        # test_loss, test_acc = SignatureNet.__test_by_model(model, test_loader, loss_function)
        # metrics = SignatureNet.__test_by_model(model, validation_loader, loss_function)

        # make_report(model, test_loader)
        # test_time = time.time() - start_time
        # print(f'Test [time: {test_time:.0f}s]')

        # test_loss = metrics['loss']
        # test_acc = metrics['acc']
        # acc_l = metrics['acc_l']
        # p = metrics['p']
        # recall_value = metrics['recall']
        # print(f'Test [avg_loss {test_loss:.6f}, acc {test_acc:.6f},'
        #       f' acc_l {acc_l:.6f}, p {p:.4f}, recall {recall_value:.4f}, time: {test_time:.0f}s]')


def make_report(model, dataloader):
    predictions = []
    trues = []

    model.eval()
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(DEVICE).float()
            img2 = img2.to(DEVICE).float()
            labels = labels.to(DEVICE)

            trues.append(labels)
            prediction = model(img1, img2).squeeze()
            predictions.append(prediction)

    predictions = torch.round(torch.cat(predictions))
    trues = torch.cat(trues)

    print(classification_report(trues, predictions))


def make_predictions(model, dataloader, loss_fn):
    predictions = []
    trues = []
    loss = 0
    model.eval()
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(DEVICE).float()
            img2 = img2.to(DEVICE).float()
            labels = labels.to(DEVICE)

            trues.append(labels)
            prediction = model(img1, img2).squeeze()
            loss += loss_fn(prediction, labels).item()
            predictions.append(prediction)

    trues = torch.cat(trues)
    predictions = torch.round(torch.cat(predictions))
    loss /= len(dataloader)
    # return {'trues': trues, 'predictions': predictions}
    return trues, predictions, loss


def test_model(model, dataloader, loss_fn):
    start_time = time.time()

    trues, predictions, test_loss = make_predictions(model, dataloader, loss_fn)
    # test_acc = accuracy_score(trues, predictions)
    # p = precision_score(trues, predictions)
    # r = recall_score(trues, predictions)
    # f1 = f1_score(trues, predictions)

    test_time = time.time() - start_time
    print(f'Test [avg_loss {test_loss:.6f}'
          # f', acc {test_acc:.6f}, precision {p:.4f}, recall {r:.4f}, f1 {f1:.4f}'
          f'] - time: {test_time:.0f}s')

    print(classification_report(trues, predictions))
    matrix = confusion_matrix(trues, predictions)
    print(matrix)
    # disp = ConfusionMatrixDisplay.from_predictions(trues, predictions)
    # disp.plot()
    # plt.show()

    sns.heatmap(matrix, annot=True)
    plt.show()
