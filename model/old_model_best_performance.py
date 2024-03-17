import numpy as np

import os
import random
from time import time
from collections import OrderedDict
from psutil import cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import Adam, Adamax
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter, time_since
from utils.signature_dataset import SignatureDataset
from utils.preprocess_image import PreprocessImage
from utils.config import Config
from model.loss.cosine_contrastive_loss import ContrastiveLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = cpu_count(logical=False)  # - 1

OUTPUT_DIR = "./savedmodels/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
BEST_MODEL = OUTPUT_DIR + 'best_loss.pt'

criterion = ContrastiveLoss()

THRESHOLD = 0.5


def get_predictions_by_cosine_similarities(cosine_similarities: torch.Tensor):
    return torch.where(torch.gt(cosine_similarities, THRESHOLD), 0.0, 1.0)  # real = 0.0, forg = 1.0


def get_predictions_by_eucl(eucl_dist: torch.Tensor):
    return torch.where(torch.gt(eucl_dist, THRESHOLD), 1.0, 0.0)  # real = 0.0, forg = 1.0


def get_predicted_label_a1(cosine_similarities, confidence):
    confidence_rounded = torch.round(confidence, decimals=1)
    left_tensor = torch.gt(cosine_similarities, 0.8)
    right_tensor = torch.ge(confidence_rounded, THRESHOLD)
    boolean_tensor = torch.logical_and(left_tensor, right_tensor)
    return torch.where(boolean_tensor, 0.0, 1.0)  # real = 0.0, forg = 1.0


def get_predicted_label_a2(cosine_similarities, confidence):
    sim_rounded = torch.round(cosine_similarities, decimals=1)
    confidence_rounded = torch.round(confidence, decimals=1)

    first_boolean_tensor = torch.logical_and(torch.gt(sim_rounded, 0.9), torch.ge(confidence_rounded, 0.4))
    second_boolean_tensor = torch.logical_and(torch.gt(sim_rounded, 0.8), torch.ge(confidence_rounded, THRESHOLD))
    # real = 0.0, forg = 1.0
    return torch.where(first_boolean_tensor, 0.0, torch.where(second_boolean_tensor, 0.0, 1.0))


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# def conv_bn_mish(in_channels, out_channels, kernel_size, stride=1, pad=0):
#     return nn.Sequential(OrderedDict([
#         ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
#         ('bn', nn.BatchNorm2d(out_channels)),
#         ('mish', nn.Mish()),
#     ]))
#
#
# def linear_bn_mish(in_features, out_features):
#     return nn.Sequential(OrderedDict([
#         ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
#         ('bn', nn.BatchNorm1d(out_features)),
#         ('mish', nn.Mish()),
#     ]))


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features)),
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))


def conv_block(in_channels, out_channels, kernel_size, stride=1, pad=1):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def linear_block(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features)),
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))


def flatten_model(modules):
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    ret = []
    try:
        for _, n in modules:
            ret.append(flatten_model(n))
    except:
        try:
            if str(modules._modules.items()) == "odict_items([])":
                ret.append(modules)
            else:
                for _, n in modules._modules.items():
                    ret.append(flatten_model(n))
        except:
            ret.append(modules)
    return flatten_list(ret)


def init_weight_in_layers(model: nn.Module):
    module_list = [module for module in model.modules()]  # this is needed
    flatted_list = flatten_model(module_list)
    for count, value in enumerate(flatted_list):
        if isinstance(value, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(value.weight, nonlinearity='relu')


class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()

        self.feature_space_size = 2048  # 1024

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(1, 96, 11, stride=4, pad=1)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(256, 384, 3, pad=1)),
            ('conv4', conv_bn_relu(384, 384, 3, pad=1)),
            ('conv5', conv_bn_relu(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2))
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_block(256 * 3 * 5, self.feature_space_size)),  # 2048)),
            ('fc2', linear_block(self.feature_space_size, self.feature_space_size)),
        ]))

    def forward_once(self, img):
        x = self.conv_layers(img)
        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.fc_layers(x)
        return x

    def forward(self, img1, img2):
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        img1 = img1.view(-1, 1, 150, 220).div(255)
        img2 = img2.view(-1, 1, 150, 220).div(255)

        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2


class SiameseModel(nn.Module):
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """

    def __init__(self):
        super(SiameseModel, self).__init__()

        self.model = SigNet()
        init_weight_in_layers(self.model)

        if Config.projection2d:
            self.probs = nn.Linear(4, 1)
        else:
            self.probs = nn.Linear(self.model.feature_space_size * 2, 1)
        self.projection2d = nn.Linear(self.model.feature_space_size, 2)

    def forward_once(self, img):
        x = self.model.forward_once(img)
        return x

    def forward(self, img1, img2):
        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)

        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)

        if Config.projection2d:
            # print("Project embeddings into 2d space")
            embedding1 = self.projection2d(embedding1)
            embedding2 = self.projection2d(embedding2)

        # Classification
        output = torch.cat([embedding1, embedding2], dim=1)
        output = self.probs(output)
        # eucl_dist = nnf.pairwise_distance(embedding1, embedding2)  # cos sim-> pairwise dist
        # sim = nnf.cosine_similarity(embedding1, embedding2)
        confidence = nnf.sigmoid(output)
        # return sim, confidence
        return embedding1, embedding2, confidence
        # return eucl_dist, confidence

    def load_model(self, file_name: str):
        self.to(DEVICE)
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(file_name)['model'])
        else:
            self.load_state_dict(torch.load(file_name, map_location=torch.device('cpu'))['model'])

    def load_best_model(self):
        self.load_model(BEST_MODEL)

    @staticmethod
    def __get_predicted_label_a2(similarity, confidence):
        if np.round(similarity, decimals=1) > 0.9 and np.round(confidence, decimals=1) >= 0.4:
            predicted_label = 'Original'
        elif similarity > 0.8 and np.round(confidence, decimals=1) >= THRESHOLD:
            predicted_label = 'Original'
        else:
            predicted_label = 'Forged'
        return predicted_label

    @staticmethod
    def __get_predicted_label_rus(similarity, confidence) -> str:
        if np.round(similarity, decimals=1) > 0.9 and np.round(confidence, decimals=1) >= 0.4:
            predicted_label = 'Настоящая подпись'
        elif similarity > 0.8 and np.round(confidence, decimals=1) >= THRESHOLD:
            predicted_label = 'Настоящая подпись'
        else:
            predicted_label = 'Подделанная подпись'
        return predicted_label

    def __train_loop(self, train_loader, loss_function, optimizer, epoch, print_fn_callback):
        losses = AverageMeter()
        batches = len(train_loader)
        targets, predictions, confidences = [], [], []
        start = time()

        self.train()
        for batch, (img1, img2, labels) in enumerate(train_loader):
            labels = labels.to(DEVICE)
            targets.append(labels)

            # sim, _ = self(img1.to(DEVICE), img2.to(DEVICE))
            # predictions.append(sim)
            # loss = loss_function(sim, labels)

            out1, out2, confidence = self(img1.to(DEVICE), img2.to(DEVICE))  # eucl_d, _
            loss = loss_function(out1, out2, labels)  # loss = loss_function(eucl_d, labels)
            sim = nnf.cosine_similarity(out1, out2)
            predictions.append(sim)  # predictions.append(eucl_d)
            confidences.append(confidence)

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
        confidences = torch.cat(confidences).squeeze()

        predictions_a1 = get_predicted_label_a1(predictions, confidences)
        predictions = get_predictions_by_cosine_similarities(predictions)
        # predictions = get_predictions_by_eucl(predictions)

        targets, predictions, predictions_a1 = targets.to('cpu'), predictions.to('cpu'), predictions_a1.to('cpu')
        acc = 1 - accuracy_score(targets, predictions)
        acc_a1 = accuracy_score(targets, predictions)
        std = np.std(losses.list)
        return losses.avg, std, acc, acc_a1

    def make_predictions(self, dataloader, loss_fn):
        targets, predictions, confidences = [], [], []
        losses = AverageMeter()

        self.eval()
        with torch.no_grad():
            for img1, img2, label in dataloader:
                label = label.to(DEVICE)
                targets.append(label)

                # cos_sim, confidence = self(img1.to(DEVICE), img2.to(DEVICE))
                # predictions.append(cos_sim)
                # loss = loss_fn(cos_sim, label)

                out1, out2, confidence = self(img1.to(DEVICE), img2.to(DEVICE))
                sim = nnf.cosine_similarity(out1, out2)
                predictions.append(sim)
                confidences.append(confidence)

                loss = loss_fn(out1, out2, label)

                # eucl_d, _ = self(img1.to(DEVICE), img2.to(DEVICE))
                # loss = loss_fn(eucl_d, label)
                # predictions.append(eucl_d)

                losses.update(loss.item(), label.size(0))

        targets = torch.cat(targets)
        predictions = torch.cat(predictions)
        confidences = torch.cat(confidences).squeeze()

        predictions_a1 = get_predicted_label_a1(predictions, confidences)
        predictions = get_predictions_by_cosine_similarities(predictions)
        # predictions = get_predictions_by_eucl(predictions)
        std = np.std(losses.list)
        return targets, predictions, predictions_a1, losses.avg, std

    def __validation_loop(self, dataloader, loss_fn):
        trues, predictions, predictions_a1, val_loss, std = self.make_predictions(dataloader, loss_fn)
        trues, predictions, predictions_a1 = trues.to('cpu'), predictions.to('cpu'), predictions_a1.to('cpu')
        acc = accuracy_score(trues, predictions)
        acc_a1 = accuracy_score(trues, predictions_a1)
        return val_loss, std, acc, acc_a1

    def fit(self, batch_size: int, epochs_number: int, print_fn):
        seed_torch(seed=Config.SEED)

        train_dataset = SignatureDataset("train", Config.CANVAS_SIZE, dim=(256, 256))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True, drop_last=True)

        validation_dataset = SignatureDataset("val", Config.CANVAS_SIZE, dim=(256, 256))
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=NUM_WORKERS,
                                       pin_memory=True, drop_last=True)

        self.to(DEVICE)
        optim = Adam(self.parameters())

        best_loss = np.inf
        avg_train_losses, avg_val_losses, = [], []
        # std_train_losses, std_val_losses, = [], []
        acc_train_list, acc_val_list = [], []
        acc_1_train_list, acc_1_val_list = [], []
        early_stop_epoch_count = 0

        for epoch in range(epochs_number):
            start_time = time()
            avg_loss, std_loss, train_acc, train_a1 = self.__train_loop(train_loader, criterion, optim, epoch, print_fn)
            train_loop_time = time() - start_time

            avg_train_losses.append(avg_loss)
            # std_train_losses.append(std_train_loss)
            acc_train_list.append(train_acc)
            acc_1_train_list.append(train_a1)

            start_time = time()
            avg_val_loss, std_val_loss, val_acc, val_acc_1 = self.__validation_loop(validation_loader, criterion)
            val_loop_time = time() - start_time

            avg_val_losses.append(avg_val_loss)
            # std_val_losses.append(std_val_loss)
            acc_val_list.append(val_acc)
            acc_1_val_list.append(val_acc_1)

            print_fn(
                f'Эпоха {epoch} - время: {train_loop_time:.0f}s - loss: {avg_loss:.4f} - std_loss: {std_loss:.4f} - acc: {train_acc:.4f} - acc1: {train_a1:.4f}'
                f' - время: {val_loop_time:.0f}s - val_loss {avg_val_loss:.4f} - val_std_loss: {std_val_loss:.4f} - val_acc: {val_acc:.4f} - val_acc_1: {val_acc_1:.4f}')

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
        return avg_train_losses, avg_val_losses, acc_train_list, acc_val_list  # std_train_losses, std_val_losses

    def test(self, batch_size: int, print_fn):
        test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)

        start_time = time()
        trues, predictions, predictions_a1, test_loss, losses = self.make_predictions(test_loader, criterion)
        trues, predictions, predictions_a1 = trues.to('cpu'), predictions.to('cpu'), predictions_a1.to('cpu')
        report = classification_report(trues, predictions, output_dict=True)  # , output_dict=True)
        report_a1 = classification_report(trues, predictions_a1, output_dict=True)
        matrix = confusion_matrix(trues, predictions)
        matrix_a1 = confusion_matrix(trues, predictions_a1)
        elapsed_time = time() - start_time
        print_fn(
            f'Тест - время: {elapsed_time:.0f}s - avg_loss: {test_loss:.5f} - std_loss: {np.std(losses):.5f}')
        return report, matrix, report_a1, matrix_a1

    def test_model_by_name(self, model_name, batch_size: int, print_fn):
        self.load_model(model_name)
        test_dataset = SignatureDataset("test", Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, drop_last=True,
                                 pin_memory=True)

        start_time = time()
        trues, predictions, predictions_a1, test_loss, losses = self.make_predictions(test_loader, criterion)
        trues, predictions, predictions_a1 = trues.to('cpu'), predictions.to('cpu'), predictions_a1.to('cpu')
        report = classification_report(trues, predictions)
        report_a1 = classification_report(trues, predictions_a1)
        matrix = confusion_matrix(trues, predictions)
        matrix_a1 = confusion_matrix(trues, predictions_a1)
        elapsed_time = time() - start_time
        print_fn(f'Тест - время: {elapsed_time:.0f}s - avg_loss: {test_loss:.5f} - std_loss: {np.std(losses):.5f}')
        return report, matrix, report_a1, matrix_a1

    def test_best_model(self, batch_size: int, print_fn_callback):
        return self.test_model_by_name(BEST_MODEL, batch_size, print_fn_callback)

    def predict(self, image_path1, image_path2):
        img1 = torch.tensor(PreprocessImage.transform_image(image_path1, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)
        img2 = torch.tensor(PreprocessImage.transform_image(image_path2, Config.CANVAS_SIZE, (256, 256)), device=DEVICE)

        self.eval()
        with torch.no_grad():
            cos_sim, confidence = self(img1, img2).item()
        prediction = 'Подделанная' if cos_sim > THRESHOLD else 'Настоящая'
        return prediction

    @staticmethod
    def predict(model, image_path1, image_path2):
        model.eval()

        img1 = SiameseModel.__transform_image(image_path1)
        img2 = SiameseModel.__transform_image(image_path2)

        with torch.no_grad():
            op1, op2, confidence = model(img1.to('cpu'), img2.to('cpu'))

        cosine_similarity = nnf.cosine_similarity(op1, op2)
        confidence = confidence.sigmoid().detach().to('cpu')
        confidence = 1 - confidence

        predicted_label = SiameseModel.__get_predicted_label_rus(cosine_similarity, confidence)
        return cosine_similarity.item(), confidence.item(), predicted_label

    @staticmethod
    def test_by_model(model, test_loader):
        counter_simple = 0
        counter_advance = 0
        counter_advance_2 = 0

        label_dict = {1.0: 'Forged', 0.0: 'Original'}
        model.eval()
        samples_count = len(test_loader)
        for i, data in enumerate(test_loader, 0):
            img1, img2, label = data
            label = label_dict[label.item()]

            with torch.no_grad():
                op1, op2, confidence = model(img1.to('cpu'), img2.to('cpu'))
            cos_sim = nnf.cosine_similarity(op1, op2)

            # predicted_label_simple = SiameseModel.__get_predicted_label_by_similarity(cos_sim)
            # if label == predicted_label_simple:
            #     counter_simple += 1

            confidence = confidence.sigmoid().detach().to('cpu')
            confidence = 1 - confidence

            predicted_label_advance = SiameseModel.__get_predicted_label_a1(cos_sim, confidence)
            if label == predicted_label_advance:
                counter_advance += 1

            predicted_label_advance_2 = SiameseModel.__get_predicted_label_a2(cos_sim, confidence)
            if label == predicted_label_advance_2:
                counter_advance_2 += 1

        accuracy = counter_simple / samples_count
        accuracy_1 = counter_advance / samples_count
        accuracy_2 = counter_advance_2 / samples_count
        return accuracy, accuracy_1, accuracy_2
