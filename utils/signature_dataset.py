from pandas import read_csv
import torch
from torch.utils.data import Dataset

from utils.preprocess_image import PreprocessImage


def get_train_dataframe():
    train_csv = "../sign_data/train_data.csv"
    # train_dir = "../../sign_data/train"
    train_df = read_csv(train_csv)
    train_df.rename(columns={"1": "label"}, inplace=True)
    train_df["image_real_paths"] = train_df["033/02_033.png"].apply(lambda x: f"../sign_data/train/{x}")
    train_df["image_forged_paths"] = train_df["033_forg/03_0203033.PNG"].apply(lambda x: f"../sign_data/train/{x}")
    return train_df


def get_test_dataframe():
    test_csv = "../sign_data/test_data.csv"
    # test_dir = "../../sign_data/test"
    test_df = read_csv(test_csv)
    test_df.rename(columns={"1": "label"}, inplace=True)
    test_df["image_real_paths"] = test_df["068/09_068.png"].apply(lambda x: f"../sign_data/test/{x}")
    test_df["image_forged_paths"] = test_df["068_forg/03_0113068.PNG"].apply(lambda x: f"../sign_data/test/{x}")
    return test_df


def get_validation_dataframe():
    val_csv = "../sign_data/val_data.csv"
    # val_dir = "../../sign_data/val"
    val_df = read_csv(val_csv)
    val_df.rename(columns={"1": "label"}, inplace=True)
    val_df["image_real_paths"] = val_df["047/01_047.png"].apply(lambda x: f"../sign_data/val/{x}")
    val_df["image_forged_paths"] = val_df["047_forg/02_0113047.PNG"].apply(lambda x: f"../sign_data/val/{x}")
    return val_df


TRAIN_DF = get_train_dataframe()
VALIDATION_DF = get_validation_dataframe()
TEST_DF = get_test_dataframe()


def get_dataframe_by_name(dataframe_name: str):
    match dataframe_name:
        case "train":
            return TRAIN_DF
        case "val":
            return VALIDATION_DF
        case "test":
            return TEST_DF


class SignatureDataset(Dataset):

    def __init__(self, df_name: str, canvas_size, dim=(256, 256)):
        self.df = get_dataframe_by_name(df_name)
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

        img1 = PreprocessImage.preprocess_signature(img1, self.canvas_size, self.dim)
        img2 = PreprocessImage.preprocess_signature(img2, self.canvas_size, self.dim)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return torch.tensor(img1), torch.tensor(img2), label.float()
