from pandas import read_csv

import torch
from torch.utils.data import Dataset

from utils.preprocess_image import PreprocessImage

DATA_DIR = "../sign_data/"
BALANCE_DIR = "../sign_data/dataset/"


def get_train_dataframe_no_balance():
    train_df = read_csv(DATA_DIR + "train_data.csv")
    train_df.rename(columns={"1": "label"}, inplace=True)
    train_df["image_real_paths"] = train_df["033/02_033.png"].apply(lambda x: DATA_DIR + f"train/{x}")
    train_df["image_forged_paths"] = train_df["033_forg/03_0203033.PNG"].apply(lambda x: DATA_DIR + f"train/{x}")
    return train_df


def get_validation_dataframe_no_balance():
    val_df = read_csv(DATA_DIR + "val_data.csv")
    val_df.rename(columns={"1": "label"}, inplace=True)
    val_df["image_real_paths"] = val_df["047/01_047.png"].apply(lambda x: DATA_DIR + f"val/{x}")
    val_df["image_forged_paths"] = val_df["047_forg/02_0113047.PNG"].apply(lambda x: DATA_DIR + f"val/{x}")
    return val_df


def get_test_dataframe_no_balance():
    test_df = read_csv(DATA_DIR + "test_data.csv")
    test_df.rename(columns={"1": "label"}, inplace=True)
    test_df["image_real_paths"] = test_df["068/09_068.png"].apply(lambda x: DATA_DIR + f"test/{x}")
    test_df["image_forged_paths"] = test_df["068_forg/03_0113068.PNG"].apply(lambda x: DATA_DIR + f"test/{x}")
    return test_df


def get_train_dataframe_balance():
    train_df = read_csv(DATA_DIR + "train_data_balanced.csv")
    train_df.rename(columns={"1": "label"}, inplace=True)
    train_df["image_real_paths"] = train_df["068/09_068.png"].apply(lambda x: BALANCE_DIR + x)
    train_df["image_forged_paths"] = train_df["068_forg/03_0113068.PNG"].apply(lambda x: BALANCE_DIR + x)
    return train_df


def get_validation_dataframe_balance():
    val_df = read_csv(DATA_DIR + "val_data_balanced.csv")
    val_df.rename(columns={"1": "label"}, inplace=True)
    val_df["image_real_paths"] = val_df["015/015_14.PNG"].apply(lambda x: BALANCE_DIR + x)
    val_df["image_forged_paths"] = val_df["015_forg/0106015_04.png"].apply(lambda x: BALANCE_DIR + x)
    return val_df


def get_test_dataframe_balance():
    test_df = read_csv(DATA_DIR + "test_data_balanced.csv")
    test_df.rename(columns={"1": "label"}, inplace=True)
    test_df["image_real_paths"] = test_df["016/016_09.PNG"].apply(lambda x: BALANCE_DIR + x)
    test_df["image_forged_paths"] = test_df["016_forg/0202016_02.png"].apply(lambda x: BALANCE_DIR + x)
    return test_df


class SignatureDataset(Dataset):
    # TRAIN_DF = get_train_dataframe_no_balance()
    # VALIDATION_DF = get_validation_dataframe_no_balance()
    # TEST_DF = get_test_dataframe_no_balance()

    TRAIN_DF = get_train_dataframe_balance()
    VALIDATION_DF = get_validation_dataframe_balance()
    TEST_DF = get_test_dataframe_balance()

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
