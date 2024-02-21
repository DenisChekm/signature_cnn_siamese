import pandas as pd
import numpy as np

import os
import random
import time
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.loss.contrastive_loss import ContrastiveLoss
from utils.preprocess_image import PreprocessImage
from utils.signature_dataset import SignatureDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ====================================================
# Directory settings
# ====================================================
OUTPUT_DIR = '../'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class Config:
    SEED = 42
    projection2d = True
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    EPOCHS = 20
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PRINT_FREQ = 50  # 100
    CANVAS_SIZE = (952, 1360)
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1000
    MODEL_NAME = "siamnet"


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.list.append(val * n)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


def conv_bn_mish(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('mish', nn.Mish()),
    ]))


def linear_bn_mish(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('mish', nn.Mish()),
    ]))


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('mish', nn.ReLU()),
    ]))


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features)),
        ('bn', nn.BatchNorm1d(out_features)),
        ('mish', nn.ReLU()),
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


# def init_weights(model: nn.Module):
#     print(model.modules())
#     for module in model.modules():
#         print(module)
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
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


def prin_layers_info(model):
    target_layers = []
    module_list = [module for module in model.modules()]  # this is needed
    flatted_list = flatten_model(module_list)

    for count, value in enumerate(flatted_list):

        if isinstance(value, (nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.Linear)):
            # if isinstance(value, (nn.Conv2d)):
            print(count, value)
            target_layers.append(value)


def init_weight_in_layers(model: nn.Module):
    module_list = [module for module in model.modules()]  # this is needed
    flatted_list = flatten_model(module_list)

    for count, value in enumerate(flatted_list):

        if isinstance(value, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(value.weight, nonlinearity='relu')



class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()

        self.feature_space_size = 1024  # 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_block(1, 96, 11, stride=4)),
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
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)

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
            return embedding1, embedding2, output
        else:
            # Classification
            output = torch.cat([embedding1, embedding2], dim=1)
            print(output.shape)
            output = self.probs(output)
            return embedding1, embedding2, output

    @staticmethod
    def __train_epoch(train_loader, model, criterion, optimizer, epoch):
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()

        start = end = time.time()
        for step, (img1, img2, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            img1 = img1.to(DEVICE).float()
            img2 = img2.to(DEVICE).float()
            labels = labels.to(DEVICE)

            batch_size = labels.size(0)
            out1, out2, _ = model(img1, img2)
            loss = criterion(out1, out2, labels)

            losses.update(loss.item(), batch_size)
            if Config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            end = time.time()
            if step % Config.PRINT_FREQ == 0 or step == (len(train_loader) - 1):
                print(f'Epoch: [{epoch}][{step}/{len(train_loader)}] ', end='')
                print(f'Elapsed: {time_since(start, float(step + 1) / len(train_loader))} ', end='')
                print(f'Loss: {losses.val:.4f}({losses.avg:.4f}) ', end='')
                print(f'Grad: {grad_norm:.4f}')

        return losses.avg, losses.list

    @staticmethod
    def train_model():
        seed_torch(seed=Config.SEED)
        train_dataset = SignatureDataset(train, Config.CANVAS_SIZE, dim=(256, 256))
        train_loader = DataLoader(train_dataset,
                                  batch_size=Config.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)

        model = SiameseModel().to(DEVICE)

        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        contrastive = ContrastiveLoss()

        best_loss = np.inf

        std_losses = []

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            avg_train_loss, losses_list = SiameseModel.__train_epoch(train_loader, model, contrastive, optimizer, epoch)

            elapsed = time.time() - start_time

            std = np.std(losses_list)
            std_losses.append(std)

            print(f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s')
            torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')
            # OUTPUT_DIR + f'model_{epoch}_{datetime.now():%Y-%m-%d %H:%M:%S}.pt')

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                print(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': model.state_dict()}, OUTPUT_DIR + 'best_loss.pt')

    @staticmethod
    def train_model_by_params(batch_size, lr, epochs_count):
        seed_torch(seed=Config.SEED)
        train_dataset = SignatureDataset(train, Config.CANVAS_SIZE, dim=(256, 256))
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)

        model = SiameseModel().to(DEVICE)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
        contrastive = ContrastiveLoss()

        best_loss = np.inf

        std_losses = []

        for epoch in range(epochs_count):
            start_time = time.time()

            avg_train_loss, losses_list = SiameseModel.__train_epoch(train_loader, model, contrastive, optimizer, epoch)

            elapsed = time.time() - start_time

            std = np.std(losses_list)
            std_losses.append(std)

            print(f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s')
            torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'model_delta_{epoch}.pt')

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                print(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': model.state_dict()}, OUTPUT_DIR + 'best_loss.pt')

        return std_losses

    @staticmethod
    def __get_predicted_label_by_similarity(similarity):
        predicted_label = 'Original' if similarity > 0.5 else 'Forged'
        return predicted_label

    @staticmethod
    def __get_predicted_label_a1(similarity, confidence):
        if similarity > 0.8 and np.round(confidence, decimals=1) >= 0.5:
            predicted_label = 'Original'
        else:
            predicted_label = 'Forged'
        return predicted_label

    @staticmethod
    def __get_predicted_label_a2(similarity, confidence):
        if np.round(similarity, decimals=1) > 0.9 and np.round(confidence, decimals=1) >= 0.4:
            predicted_label = 'Original'
        elif similarity > 0.8 and np.round(confidence, decimals=1) >= 0.5:
            predicted_label = 'Original'
        else:
            predicted_label = 'Forged'
        return predicted_label

    @staticmethod
    def test_model():
        seed_torch(seed=Config.SEED)
        model = SiameseModel().to(DEVICE)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(f'{Config.MODEL_NAME}_best_loss.pt')['model'])
        else:
            model.load_state_dict(
                torch.load(f'{Config.MODEL_NAME}_best_loss.pt', map_location=torch.device('cpu'))['model'])

        test_dataset = SignatureDataset(test, Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)

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

            predicted_label_simple = SiameseModel.__get_predicted_label_by_similarity(cos_sim)
            if label == predicted_label_simple:
                counter_simple += 1

            confidence = confidence.sigmoid().detach().to('cpu')
            confidence = 1 - confidence

            predicted_label_advance = SiameseModel.__get_predicted_label_a1(cos_sim, confidence)
            if label == predicted_label_advance:
                counter_advance += 1

            predicted_label_advance_2 = SiameseModel.__get_predicted_label_a2(cos_sim, confidence)
            if label == predicted_label_advance_2:
                counter_advance_2 += 1

        print(f'Accuracy[simple]: {counter_simple / samples_count:.4f}')
        print(f'Accuracy[A1]: {counter_advance / samples_count:.4f}')
        print(f'Accuracy[A2]: {counter_advance_2 / samples_count:.4f}')

    @staticmethod
    def test_model_best_loss():
        seed_torch(seed=Config.SEED)
        model = SiameseModel().to(DEVICE)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load('../best_loss.pt')['model'])
        else:
            model.load_state_dict(
                torch.load('../best_loss.pt', map_location=torch.device('cpu'))['model'])

        test_dataset = SignatureDataset(test, Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)

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

            predicted_label_simple = SiameseModel.__get_predicted_label_by_similarity(cos_sim)
            if label == predicted_label_simple:
                counter_simple += 1

            confidence = confidence.sigmoid().detach().to('cpu')
            confidence = 1 - confidence

            predicted_label_advance = SiameseModel.__get_predicted_label_a1(cos_sim, confidence)
            if label == predicted_label_advance:
                counter_advance += 1

            predicted_label_advance_2 = SiameseModel.__get_predicted_label_a2(cos_sim, confidence)
            if label == predicted_label_advance_2:
                counter_advance_2 += 1

        print(f'Accuracy[simple]: {counter_simple / samples_count:.4f}')
        print(f'Accuracy[A1]: {counter_advance / samples_count:.4f}')
        print(f'Accuracy[A2]: {counter_advance_2 / samples_count:.4f}')

    @staticmethod
    def load_best_model():
        seed_torch(seed=Config.SEED)
        model = SiameseModel().to(DEVICE)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(f'{Config.MODEL_NAME}_best_loss.pt')['model'])
        else:
            model.load_state_dict(
                torch.load(f'{Config.MODEL_NAME}_best_loss.pt', map_location=torch.device('cpu'))['model'])
        return model

    @staticmethod
    def __get_predicted_label_rus(similarity, confidence) -> str:
        if np.round(similarity, decimals=1) > 0.9 and np.round(confidence, decimals=1) >= 0.4:
            predicted_label = 'Настоящая подпись'
        elif similarity > 0.8 and np.round(confidence, decimals=1) >= 0.5:
            predicted_label = 'Настоящая подпись'
        else:
            predicted_label = 'Подделанная подпись'
        return predicted_label

    @staticmethod
    def __transform_image(image_path):
        img1 = PreprocessImage.load_signature(image_path)
        canvas_size = Config.CANVAS_SIZE
        dim = (256, 256)

        img1 = PreprocessImage.preprocess_signature(img1, canvas_size, dim)
        return torch.tensor(img1)

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

            predicted_label_simple = SiameseModel.__get_predicted_label_by_similarity(cos_sim)
            if label == predicted_label_simple:
                counter_simple += 1

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

    @staticmethod
    def train_with_test():
        seed_torch(seed=Config.SEED)
        train_dataset = SignatureDataset(train, Config.CANVAS_SIZE, dim=(256, 256))
        train_loader = DataLoader(train_dataset,
                                  batch_size=Config.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)

        test_dataset = SignatureDataset(test, Config.CANVAS_SIZE, dim=(256, 256))
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)

        model = SiameseModel().to(DEVICE)

        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        contrastive = ContrastiveLoss()

        best_loss = np.inf
        std_losses = []

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            avg_train_loss, losses_list = SiameseModel.__train_epoch(train_loader, model, contrastive, optimizer, epoch)

            elapsed = time.time() - start_time

            std = np.std(losses_list)
            std_losses.append(std)

            start_time = time.time()
            a, a_1, a_2 = SiameseModel.test_by_model(model, test_loader)
            test_time = time.time() - start_time

            print(
                f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f} - std_loss: {std:.4f} time: {elapsed:.0f}s; '
                f'Test [A {a:.4f}, A1 {a_1:.4f}, A2 {a_2:.4f} time: {test_time:.0f}s]')

            torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'model_{epoch}.pt')

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                print(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': model.state_dict()}, OUTPUT_DIR + 'best_loss.pt')
