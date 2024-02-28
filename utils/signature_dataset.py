from pandas import read_csv

import torch
from torch.utils.data import Dataset

from utils.preprocess_image import PreprocessImage

DATA_DIR = "C:/Users/denle/PycharmProjects/sign_data/"


def get_dataframe(file):
    dataset_dir = DATA_DIR + file + "/"
    csv_file_path = DATA_DIR + file + "_data.csv"
    df = read_csv(csv_file_path)  # , names=["image_real_paths, image_forged_paths, label"])
    df.rename(columns={"1": "label"}, inplace=True)
    df["image_real_paths"] = df["033/02_033.png"].apply(lambda x: dataset_dir + x)
    df["image_forged_paths"] = df["033_forg/03_0203033.PNG"].apply(lambda x: dataset_dir + x)
    print(df.head())
    print(df.columns.values)
    df = df[["label", "image_real_paths", "image_forged_paths"]]
    print(df.head())
    print(df.columns.values)
    return df


def get_train_dataframe():
    train_df = read_csv(DATA_DIR + "train_data.csv")
    train_df.rename(columns={"1": "label"}, inplace=True)
    train_df["image_real_paths"] = train_df["033/02_033.png"].apply(lambda x: DATA_DIR + f"train/{x}")
    train_df["image_forged_paths"] = train_df["033_forg/03_0203033.PNG"].apply(lambda x: DATA_DIR + f"train/{x}")
    return train_df


def get_validation_dataframe():
    val_df = read_csv(DATA_DIR + "val_data.csv")
    val_df.rename(columns={"1": "label"}, inplace=True)
    val_df["image_real_paths"] = val_df["047/01_047.png"].apply(lambda x: DATA_DIR + f"val/{x}")
    val_df["image_forged_paths"] = val_df["047_forg/02_0113047.PNG"].apply(lambda x: DATA_DIR + f"val/{x}")
    return val_df


def get_test_dataframe():
    test_df = read_csv(DATA_DIR + "test_data.csv")
    test_df.rename(columns={"1": "label"}, inplace=True)
    test_df["image_real_paths"] = test_df["068/09_068.png"].apply(lambda x: DATA_DIR + f"test/{x}")
    test_df["image_forged_paths"] = test_df["068_forg/03_0113068.PNG"].apply(lambda x: DATA_DIR + f"test/{x}")
    return test_df


def preprocess_tensor(img):
    return img.view(1, 150, 220).float().div(255)


class SignatureDataset(Dataset):
    TRAIN_DF = get_train_dataframe()  # get_dataframe("train")
    VALIDATION_DF = get_validation_dataframe()  # get_dataframe("val")
    TEST_DF = get_test_dataframe()  # get_dataframe("test")

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

        img1 = PreprocessImage.load_signature(real_file_path)
        img2 = PreprocessImage.load_signature(forged_file_path)

        img1 = torch.tensor(PreprocessImage.preprocess_signature(img1, self.canvas_size, self.dim))
        img2 = torch.tensor(PreprocessImage.preprocess_signature(img2, self.canvas_size, self.dim))
        label = torch.tensor(self.labels[index], dtype=torch.long).float()
        return preprocess_tensor(img1), preprocess_tensor(img2), label.float()

    def __get_dataframe_by_name(self, dataframe_name: str):
        match dataframe_name:
            case "train":
                return self.TRAIN_DF
            case "val":
                return self.VALIDATION_DF
            case "test":
                return self.TEST_DF
