from pathlib import Path
from typing import Any

import albumentations as albu
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from gldv2dataset import get_dataloaders
from model import AngularModel


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_augmentations(input_size):
    train_transform = albu.Compose([
        albu.RandomResizedCrop(
            input_size,
            input_size,
            scale=(0.6, 1.0),
            p=1.0,
        ),
        albu.Normalize(),
    ])
    test_transform = albu.Compose([
        albu.Resize(width=input_size, height=input_size),
        albu.Normalize(),
    ])
    return train_transform, test_transform


def save_checkpoint(model: Any, epoch: int, path: Path):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }, path)


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(labels)
    acc = correct.float().mean().item()
    return acc


def train(
    input_size: int = 128,
    num_epochs: int = 10,
    batch_size: int = 128,
    num_workers: int = 12,
    backbone: str = "resnet34",
    init_lr: float = 0.001,
    device: str = "cuda"
):
    path_train_csv = "gldv2_micro/train.csv"
    path_val_csv = "gldv2_micro/val.csv"

    train_transform, val_transform = get_augmentations(input_size)
    dataloaders = get_dataloaders(
        path_train_csv,
        path_val_csv,
        train_transform,
        val_transform,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
    )

    model = AngularModel(
        n_classes=dataloaders["train"].dataset.n_classes,
        model_name=backbone,
        pretrained=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    ttl_iters = num_epochs * len(dataloaders["train"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=0.001)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(num_epochs):
        train_meters = {
            "loss": AverageMeter(),
            "acc": AverageMeter(),
        }
        model.train()
        for iter_idx, (X, y) in enumerate(dataloaders["train"]):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            X_out = model(X, y)
            loss = criterion(X_out, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

            trn_loss = loss.item()
            trn_acc = accuracy(X_out, y)

            train_meters["loss"].update(trn_loss, n=X.size(0))
            train_meters["acc"].update(trn_acc, n=X.size(0))

            if iter_idx % 100 == 0:
                print("Epoch {:.4f} / trn/loss={:.4f}, trn/acc={:.4f}".format(
                          epoch + iter_idx / len(dataloaders["train"]),
                          train_meters["loss"].avg,
                          train_meters["acc"].avg))

        val_meters = {
            "loss": AverageMeter(),
            "acc": AverageMeter(),
        }
        model.eval()
        for iter_idx, (X, y) in enumerate(dataloaders["val"]):
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X, y)
                loss = criterion(X_out, y)

                val_loss = loss.item()
                val_acc = accuracy(X_out, y)

            val_meters["loss"].update(val_loss, n=X.size(0))
            val_meters["acc"].update(val_acc, n=X.size(0))

        print("Ep {:d} / val/loss={:.4f}, val/acc={:.4f}".format(
            epoch + 1,
            val_meters["loss"].avg,
            val_meters["acc"].avg))

        save_checkpoint(model, epoch+1, Path("out/arcface_last.pth"))


def extract_vectors(
    model,
    image_files,
    input_size,
    out_dim,
    transform,
    device_str,
    bbxs=None,
    print_freq=1000
):
    dataloader = torch.utils.data.DataLoader(
        ImagesFromList(root="", images=image_files, imsize=input_size,
                       transform=transform, bbxs=bbxs),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    with torch.no_grad():
        vecs = torch.zeros(out_dim, len(image_files))
        for i, X in enumerate(dataloader):
            if i % print_freq == 0:
                print(f"Processing {i} of {len(dataloader.dataset)}")
            X = X.to(device_str)
            vecs[:, i] = model.extract_features(X)
    return vecs


def get_query_index_images(cfg):
    index_images = [cfg["im_fname"](cfg, i) for i in range(cfg["n"])]
    query_images = [cfg["qim_fname"](cfg, i) for i in range(cfg["nq"])]

    try:
        bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]
    except KeyError:
        bbxs = None

    return index_images, query_images, bbxs


def evaluate(input_size: int = 256, device: str = "cuda"):
    datasets = {
        "roxford5k": configdataset("roxford5k", "./"),
        "rparis6k": configdataset("rparis6k", "./")
    }

    backbone = "resnet34"

    model = AngularModel(
        n_classes=3103,
        model_name=backbone,
        pretrained=True,
    )
    model.load_state_dict(
        torch.load("out/arcface_last.pth")["state_dict"],
    )
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # それぞれのデータセットで大域特徴を抽出して、中間ファイルに保存
    for dataset_name, dataset_config in datasets.items():
        index_images, query_images, bbxs = get_query_index_images(
            dataset_config
        )
        print(f"Extract index vectors on {dataset_name} ...")
        index_vectors = extract_vectors(
            model, index_images, input_size, 512, test_transform, device
        )
        print(f"Extract query vectors on {dataset_name} ...")
        query_vectors = extract_vectors(
            model, query_images, input_size, 512, test_transform, device,
            bbxs=bbxs
        )
        index_vectors = index_vectors.numpy()
        query_vectors = query_vectors.numpy()

        # 時間節約のため中間ファイルに保存
        np.save(f"{dataset_name}_index.npy", index_vectors.astype(np.float32))
        np.save(f"{dataset_name}_query.npy", query_vectors.astype(np.float32))

    # 大域特徴をロードして、内積に基づいて順位付けして評価
    for dataset_name, dataset_config in datasets.items():
        # shape = (n_dims, n_images)
        index_vectors = np.load(f"{dataset_name}_index.npy")
        query_vectors = np.load(f"{dataset_name}_query.npy")

        # shape = (n_index_images, n_query_images)
        scores = np.dot(index_vectors.T, query_vectors)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset_name, ranks, dataset_config["gnd"])


def main():
    train(input_size=256, batch_size=64, num_epochs=10, device="cuda")
    evaluate(input_size=256, device="cuda")


if __name__ == "__main__":
    main()
