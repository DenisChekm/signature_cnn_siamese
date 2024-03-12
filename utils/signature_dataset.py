from pandas import read_csv, DataFrame

import torch
from torch.utils.data import Dataset

from utils.preprocess_image import PreprocessImage

DATA_DIR = "../sign_data/"


def get_dataframe_no_balance(csv_file_name: str, dataset_folder: str) -> DataFrame:
    df = read_csv(DATA_DIR + csv_file_name + ".csv", names=['image_real_paths', 'image_forged_paths', 'label'])
    df["image_real_paths"] = df["image_real_paths"].apply(lambda x: DATA_DIR + dataset_folder + "/" + x)
    df["image_forged_paths"] = df["image_forged_paths"].apply(lambda x: DATA_DIR + dataset_folder + "/" + x)
    return df


def get_dataframe_balance(csv_file_name: str) -> DataFrame:
    balanced_dataset_folder = "../sign_data/dataset/"
    df = read_csv(DATA_DIR + csv_file_name + ".csv", names=['image_real_paths', 'image_forged_paths', 'label'])
    df["image_real_paths"] = df["image_real_paths"].apply(lambda x: balanced_dataset_folder + x)
    df["image_forged_paths"] = df["image_forged_paths"].apply(lambda x: balanced_dataset_folder + x)
    return df


class SignatureDataset(Dataset):
    # TRAIN_DF = get_dataframe_no_balance('train_data', 'train')
    # VALIDATION_DF = get_dataframe_no_balance('val_data', 'val')
    # TEST_DF = get_dataframe_no_balance('test_data', 'test')

    TRAIN_DF = get_dataframe_balance("train_data_balanced")
    VALIDATION_DF = get_dataframe_balance("val_data_balanced")
    TEST_DF = get_dataframe_balance("test_data_balanced")

    def __init__(self, df_name: str, canvas_size, dim=(256, 256)):
        self.df = self.__get_dataframe_by_name(df_name)
        self.real_file_names = self.df["image_real_paths"].values
        self.forged_file_names = self.df["image_forged_paths"].values
        self.labels = self.df["label"].values
        self.dim = dim
        self.canvas_size = canvas_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        real_file_path = self.real_file_names[index]
        forged_file_path = self.forged_file_names[index]

        img1 = PreprocessImage.transform_image(real_file_path, self.canvas_size, self.dim)
        img2 = PreprocessImage.transform_image(forged_file_path, self.canvas_size, self.dim)

        img1 = torch.tensor(img1, dtype=torch.float)
        img2 = torch.tensor(img2, dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)  # dtype=torch.long).float()
        return img1, img2, label

    def __get_dataframe_by_name(self, dataframe_name: str):
        match dataframe_name:
            case "train":
                return self.TRAIN_DF
            case "val":
                return self.VALIDATION_DF
            case "test":
                return self.TEST_DF
