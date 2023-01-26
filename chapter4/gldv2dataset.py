from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def filename_to_filepath(filename: str) -> str:
    return f"gldv2_micro/images/{filename}"


class GLDv2MiniDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Any,
        is_test: bool = False,
    ) -> None:
        self.dataframe = dataframe
        self.n_classes = self.dataframe["landmark_id"].nunique()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        assert Path(row["filepath"]).exists()

        im = np.array(Image.open(row["filepath"]))
        im = self.transform(image=im)["image"]
        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(row["landmark_id"]).long()

        return im, target


def get_dataloaders(
    path_train_csv: str,
    path_val_csv: str,
    train_transform: Any,
    val_transform: Any,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    num_workers: int = 8,
) -> Dict[str, DataLoader]:
    df_trn = pd.read_csv(path_train_csv)
    df_trn["filepath"] = df_trn["filename"].apply(filename_to_filepath)
    df_val = pd.read_csv(path_val_csv)
    df_val["filepath"] = df_val["filename"].apply(filename_to_filepath)

    dataset_train = GLDv2MiniDataset(df_trn, train_transform)
    dataset_val = GLDv2MiniDataset(df_val, val_transform, is_test=True)

    dataloaders = {}
    dataloaders["train"] = DataLoader(
        dataset=dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=train_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    dataloaders["val"] = DataLoader(
        dataset=dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=val_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataloaders
