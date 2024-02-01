import torch
from torch.utils.data import Dataset

from preprocess_image import PreprocessImage


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

        img1 = PreprocessImage.load_signature(real_file_path)
        img2 = PreprocessImage.load_signature(forged_file_path)

        img1 = PreprocessImage.preprocess_signature(img1, self.canvas_size, self.dim)
        img2 = PreprocessImage.preprocess_signature(img2, self.canvas_size, self.dim)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return torch.tensor(img1), torch.tensor(img2), label.float()
