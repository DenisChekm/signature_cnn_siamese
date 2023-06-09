import pandas as pd
import numpy as np

import os
import random
import time
from collections import OrderedDict
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, Dataset

from skimage import img_as_ubyte
from skimage import filters, transform
from skimage.io import imread

import matplotlib.pyplot as plt
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train data
train_csv = "../sign_data/train_data.csv"
train_dir = "../sign_data/train"
train = pd.read_csv(train_csv)
train.rename(columns={"1": "label"}, inplace=True)
train["image_real_paths"] = train["033/02_033.png"].apply(lambda x: f"../sign_data/train/{x}")
train["image_forged_paths"] = train["033_forg/03_0203033.PNG"].apply(lambda x: f"../sign_data/train/{x}")

# Test data
test_csv = "../sign_data/test_data.csv"
test_dir = "../sign_data/test"
test = pd.read_csv(test_csv)
test.rename(columns={"1": "label"}, inplace=True)
test["image_real_paths"] = test["068/09_068.png"].apply(lambda x: f"../sign_data/test/{x}")
test["image_forged_paths"] = test["068_forg/03_0113068.PNG"].apply(lambda x: f"../sign_data/test/{x}")

# ====================================================
# Directory settings
# ====================================================
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class CFG:
    apex = False
    debug = False
    print_freq = 50 # 100
    size = 128
    num_workers = 4
    epochs = 10
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-3
    canvas_size = (952, 1360)
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_size = train["label"].shape[0]
    threshold = 0.69
    trn_folds = [0]
    model_name = 'delta_test.pt'  # 'vit_base_patch32_224_in21k' 'tf_efficientnetv2_b0' 'resnext50_32x4d' 'tresnet_m'
    train = True
    early_stop = True
    target_col = "label"
    projection2d = True
    fc_dim = 512
    early_stopping_steps = 5
    grad_cam = False
    seed = 42


if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_signature(path):
    return img_as_ubyte(imread(path, as_gray=True))


def preprocess_signature(img: np.ndarray,
                         canvas_size: Tuple[int, int],
                         img_size: Tuple[int, int] = (170, 242),
                         input_size: Tuple[int, int] = (150, 220)) -> np.ndarray:
    """ Pre-process a signature image, centering it in a canvas, resizing the image and cropping it.
    Parameters
    ----------
    img : np.ndarray (H x W)
        The signature image
    canvas_size : tuple (H x W)
        The size of a canvas where the signature will be centered on.
        Should be larger than the signature.
    img_size : tuple (H x W)
        The size that will be used to resize (rescale) the signature
    input_size : tuple (H x W)
        The final size of the signature, obtained by croping the center of image.
        This is necessary in cases where data-augmentation is used, and the input
        to the neural network needs to have a slightly smaller size.
    Returns
    -------
    np.narray (input_size):
        The pre-processed image
    -------
    """
    img = img.astype(np.uint8)
    centered = normalize_image(img, canvas_size)
    inverted = 255 - centered
    resized = resize_image(inverted, img_size)

    if input_size is not None and input_size != img_size:
        cropped = crop_center(resized, input_size)
    else:
        cropped = resized

    return cropped


def normalize_image(img: np.ndarray,
                    canvas_size: Tuple[int, int] = (840, 1360)) -> np.ndarray:
    """ Centers an image in a pre-defined canvas size, and remove
    noise using OTSU's method.
    Parameters
    ----------
    img : np.ndarray (H x W)
        The image to be processed
    canvas_size : tuple (H x W)
        The desired canvas size
    Returns
    -------
    np.ndarray (H x W)
        The normalized image
    """

    # 1) Crop the image before getting the center of mass

    # Apply a gaussian filter on the image to remove small components
    # Note: this is only used to define the limits to crop the image
    blur_radius = 2
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)

    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
    threshold = filters.threshold_otsu(img)

    # Find the center of mass
    binarized_image = blurred_image > threshold
    r, c = np.where(binarized_image == 0)
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    # Crop the image with a tight box
    cropped = img[r.min(): r.max(), c.min(): c.max()]

    # 2) Center the image
    img_rows, img_cols = cropped.shape
    max_rows, max_cols = canvas_size

    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center

    # Make sure the new image does not go off bounds
    # Emit a warning if the image needs to be cropped, since we don't want this
    # for most cases (may be ok for feature learning, so we don't raise an error)
    if img_rows > max_rows:
        # Case 1: image larger than required (height):  Crop.
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        r_start = 0
        difference = img_rows - max_rows
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + max_rows, :]
        img_rows = max_rows
    else:
        extra_r = (r_start + img_rows) - max_rows
        # Case 2: centering exactly would require a larger image. relax the centering of the image
        if extra_r > 0:
            r_start -= extra_r
        if r_start < 0:
            r_start = 0

    if img_cols > max_cols:
        # Case 3: image larger than required (width). Crop.
        print('Warning: cropping image. The signature should be smaller than the canvas size')
        c_start = 0
        difference = img_cols - max_cols
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + max_cols]
        img_cols = max_cols
    else:
        # Case 4: centering exactly would require a larger image. relax the centering of the image
        extra_c = (c_start + img_cols) - max_cols
        if extra_c > 0:
            c_start -= extra_c
        if c_start < 0:
            c_start = 0

    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    # Add the image to the blank canvas
    normalized_image[r_start:r_start + img_rows, c_start:c_start + img_cols] = cropped

    # Remove noise - anything higher than the threshold. Note that the image is still grayscale
    normalized_image[normalized_image > threshold] = 255

    return normalized_image


def remove_background(img: np.ndarray) -> np.ndarray:
    """ Remove noise using OTSU's method.
        Parameters
        ----------
        img : np.ndarray
            The image to be processed
        Returns
        -------
        np.ndarray
            The image with background removed
        """

    img = img.astype(np.uint8)
    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
    threshold = filters.threshold_otsu(img)

    # Remove noise - anything higher than the threshold. Note that the image is still grayscale
    img[img > threshold] = 255

    return img


def resize_image(img: np.ndarray,
                 size: Tuple[int, int]) -> np.ndarray:
    """ Crops an image to the desired size without stretching it.
    Parameters
    ----------
    img : np.ndarray (H x W)
        The image to be cropped
    size : tuple (H x W)
        The desired size
    Returns
    -------
    np.ndarray
        The cropped image
    """
    height, width = size

    # Check which dimension needs to be cropped
    # (assuming the new height-width ratio may not match the original size)
    width_ratio = float(img.shape[1]) / width
    height_ratio = float(img.shape[0]) / height
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(img.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(img.shape[0] / width_ratio))

    # Resize the image (will still be larger than new_size in one dimension)
    img = transform.resize(img, (resize_height, resize_width),
                           mode='constant', anti_aliasing=True, preserve_range=True)

    img = img.astype(np.uint8)

    # Crop to exactly the desired new_size, using the middle of the image:
    if width_ratio > height_ratio:
        start = int(round((resize_width - width) / 2.0))
        return img[:, start:start + width]
    else:
        start = int(round((resize_height - height) / 2.0))
        return img[start:start + height, :]


def crop_center(img: np.ndarray,
                size: Tuple[int, int]) -> np.ndarray:
    """ Crops the center of an image
        Parameters
        ----------
        img : np.ndarray (H x W)
            The image to be cropped
        size: tuple (H x W)
            The desired size
        Returns
        -------
        np.ndarray
            The cRecentropped image
        """
    img_shape = img.shape
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    cropped = img[start_y: start_y + size[0], start_x:start_x + size[1]]
    return cropped


def crop_center_multiple(imgs: np.ndarray,
                         size: Tuple[int, int]) -> np.ndarray:
    """ Crops the center of multiple images
        Parameters
        ----------
        imgs : np.ndarray (N x C x H_old x W_old)
            The images to be cropped
        size: tuple (H x W)
            The desired size
        Returns
        -------
        np.ndarray (N x C x H x W)
            The cropped images
        """
    img_shape = imgs.shape[2:]
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    cropped = imgs[:, :, start_y: start_y + size[0], start_x:start_x + size[1]]
    return cropped


class SignatureDataset(Dataset):

    def __init__(self, df, canvas_size, dim=(256, 256)):
        self.df = df
        self.real_file_names = df["image_real_paths"].values
        self.forged_file_names = df["image_forged_paths"].values
        self.labels = df["label"].values
        self.dim = dim
        self.canvas_size = canvas_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # getting the image path
        real_file_path = self.real_file_names[index]
        forged_file_path = self.forged_file_names[index]

        img1 = load_signature(real_file_path)
        img2 = load_signature(forged_file_path)

        img1 = preprocess_signature(img1, self.canvas_size, self.dim)
        img2 = preprocess_signature(img2, self.canvas_size, self.dim)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return torch.tensor(img1), torch.tensor(img2), label.float()


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


class SigNet(nn.Module):
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """

    def __init__(self):
        super(SigNet, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_mish(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_mish(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_mish(256, 384, 3, pad=1)),
            ('conv4', conv_bn_mish(384, 384, 3, pad=1)),
            ('conv5', conv_bn_mish(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_mish(256 * 3 * 5, 2048)),
            ('fc2', linear_bn_mish(self.feature_space_size, self.feature_space_size)),
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
        # forward pass of input 1
        output1 = self.forward_once(img1)
        # forward pass of input 2
        output2 = self.forward_once(img2)
        return output1, output2


class SiameseModel(nn.Module):
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """

    def __init__(self):
        super(SiameseModel, self).__init__()

        self.model = SigNet()

        if CFG.projection2d:
            self.probs = nn.Linear(4, 1)
        else:
            self.probs = nn.Linear(self.model.feature_space_size * 2, 1)
        self.projection2d = nn.Linear(self.model.feature_space_size, 2)

    def forward_once(self, img):
        x = self.model.forward_once(img)
        return x

    def forward(self, img1, img2):

        # Inputs need to have 4 dimensions (batch x channels x height x width), and also be between [0, 1]
        # forward pass of input 1
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)
        embedding1 = self.forward_once(img1)
        # forward pass of input 2
        embedding2 = self.forward_once(img2)

        if CFG.projection2d:
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


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.cosine_similarity(F.normalize(output1), F.normalize(output2))
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch):
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (img1, img2, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img1 = img1.to(device).float()
        img2 = img2.to(device).float()
        labels = labels.to(device)

        batch_size = labels.size(0)
        out1, out2, preds = model(img1, img2)
        loss = criterion(out1, out2, labels)

        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(f'Epoch: [{epoch}][{step}/{len(train_loader)}] ', end='')
            print(f'Elapsed: {time_since(start, float(step + 1) / len(train_loader))} ', end='')
            print(f'Loss: {losses.val:.4f}({losses.avg:.4f}) ', end='')
            print(f'Grad: {grad_norm:.4f}')

    return losses.avg


def val_fn(val_loader, model, criterion, epoch):
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.eval()
    start = end = time.time()
    global_step = 0
    with torch.no_grad():
        for step, (img1, img2, labels) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            img1 = img1.to(device).float()
            img2 = img2.to(device).float()
            labels = labels.to(device)

            batch_size = labels.size(0)
            out1, out2, preds = model(img1, img2)
            loss = criterion(out1, out2, labels)

            # record loss
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            end = time.time()
            if step % CFG.print_freq == 0 or step == (len(val_loader) - 1):
                print(f'Epoch: [{epoch}][{step}/{len(val_loader)}] ', end='')
                print(f'Elapsed: {time_since(start, float(step + 1) / len(val_loader))} ', end='')
                print(f'Loss: {losses.val:.4f}({losses.avg:.4f})')

    return losses.avg


def imshow(img, text=None, save=False):
    npimg = img.numpy()
    plt.axis('off')

    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    # plt.imshow(np.transpose(npimg, (1,2,0)), cmap='gray')
    plt.imshow(npimg[1, :, :])
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def train_valid(train_dataloader, val_dataloader):
    nn_model = SiameseModel().to(device)
    criterion = ContrastiveLoss()
    # optimizer = RMSprop(nn_model.parameters(), lr=1e-4, alpha=0.99)
    optimizer = Adam(nn_model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    train_losses = []
    val_losses = []

    num_train_batches = len(train_dataloader) - 1
    min_valid_loss = np.inf

    for epoch in range(1, CFG.epochs + 1):

        train_loss = 0.0
        nn_model.train()
        for batch_i, batch_data in enumerate(train_dataloader):
            img0, img1, labels = batch_data
            img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)

            optimizer.zero_grad()
            output1, output2 = nn_model(img0, img1)

            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_i % CFG.print_freq == 0:
                print("Epoch: [{}] [{}/{}] Loss {:.6f}"
                      .format(epoch, batch_i, num_train_batches, loss.item()))
            elif batch_i == num_train_batches:
                print("Epoch: [{}] [{}/{}] Loss {:.6f}"
                      .format(epoch, batch_i, num_train_batches, loss.item()))

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        val_loss = 0.0
        nn_model.eval()
        with torch.no_grad():
            for batch_data in val_dataloader:
                img0, img1, labels = batch_data
                img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)

                output1, output2 = nn_model(img0, img1)
                loss = criterion(output1, output2, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print("Epoch {} - avg_train_loss={:.6f} - avg_val_loss={:.6f}\n".format(epoch, avg_train_loss,  avg_val_loss))

        if min_valid_loss > avg_val_loss:
            print("Val loss decreased ({:.6f}->{:.6f})\t Saving the model\n".format(min_valid_loss, avg_val_loss))
            min_valid_loss = avg_val_loss
            torch.save(nn_model.state_dict(), CFG.model_name)


def get_test_acc(data_loader, nn_model):
    correct = 0
    nn_model.eval()

    with torch.no_grad():
        for batch_data in data_loader:
            img0, img1, label = batch_data
            img0, img1, label = img0.to('cpu'), img1.to('cpu'), label.to('cpu')

            output1, output2, preds = nn_model(img0, img1)
            cos_sim = F.cosine_similarity(output1, output2)

            is_label_orig = False
            if label.item() == 0:
                is_label_orig = True

            is_predict_orig = False
            if cos_sim > CFG.threshold:
                is_predict_orig = True

            if is_label_orig == is_predict_orig:
                correct += 1

    test_acc = 100. * correct / len(data_loader)
    return test_acc


def main():
    seed_torch(seed=CFG.seed)
    train_dataset = SignatureDataset(train, CFG.canvas_size, dim=(256, 256))
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    val_dataset = SignatureDataset(test, CFG.canvas_size, dim=(256, 256))
    val_loader = DataLoader(val_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=True,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    model = SiameseModel().to(device)

    # delete
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('./model_delta_5.pt')['model'])
    else:
        model.load_state_dict(torch.load('./model_delta_5.pt', map_location=torch.device('cpu'))['model'])

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    contrastive = ContrastiveLoss()

    best_loss = np.inf

    train_losses = []
    val_losses = []
    # train_accs = []
    # val_accs = []

    for epoch in range(CFG.epochs):
        start_time = time.time()
        epoch = epoch + 6   # delete
        avg_train_loss = train_fn(train_loader, model, contrastive, optimizer, epoch)
        avg_val_loss = val_fn(val_loader, model, contrastive, epoch)

        elapsed = time.time() - start_time

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1} - avg_train_loss: {avg_train_loss:.4f} - avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s')
        torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'model_delta_{epoch}.pt')

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            print(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict()}, OUTPUT_DIR + f'{CFG.model_name}')

        # seed_torch(seed=CFG.seed)
        # model = SiameseModel().to(device)
        # if torch.cuda.is_available():
        #     model.load_state_dict(torch.load(f'{CFG.model_name}')['model'])
        # else:
        #     model.load_state_dict(torch.load(f'{CFG.model_name}', map_location=torch.device('cpu'))['model'])
        #
        # train_loader = DataLoader(train_dataset,
        #                           batch_size=1,
        #                           shuffle=True,
        #                           num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        #
        # test_loader = DataLoader(val_dataset,
        #                          batch_size=1,
        #                          shuffle=True,
        #                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

        # start_time = time.time()
        # train_acc = get_test_acc(train_loader, model)
        # elapsed = time.time() - start_time
        # print(f'train acc elapsed {elapsed}s')
        #
        # start_time = time.time()
        # val_acc = get_test_acc(test_loader, model)
        # elapsed = time.time() - start_time
        # print(f'val acc elapsed {elapsed}s')
        # train_accs.append(train_acc)
        # val_accs.append(val_acc)
        # print(f'Epoch {epoch + 1} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}')

        # counter = 0
        # label_dict = {1.0: 'Forged', 0.0: 'Original'}
        # model.eval()
        # for i, data in enumerate(test_loader, 0):
        #     img1, img2, label = data
        #     concatenated = torch.cat((img1, img2), 0)
        #
        #     with torch.no_grad():
        #         op1, op2, confidence = model(img1.to('cpu'), img2.to('cpu'))
        #     confidence = confidence.sigmoid().detach().to('cpu')
        #     if label == 0.0:
        #         confidence = 1 - confidence
        #     cos_sim = F.cosine_similarity(op1, op2)
        #
        #     imshow(torchvision.utils.make_grid(concatenated.unsqueeze(1)),
        #            f'similarity: {cos_sim.item():.2f} Confidence: {confidence.item():.2f} Label: {label_dict[label.item()]}')
        #     counter += 1
        #     if counter == 40:
        #         break

    plt.plot(train_losses, '-o')
    plt.plot(val_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.grid()
    plt.show()

    # plt.plot(train_accs, '-o')
    # plt.plot(val_accs, '-o')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(['Train', 'Valid'])
    # plt.title('Train vs Valid Accuracy')
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    main()
