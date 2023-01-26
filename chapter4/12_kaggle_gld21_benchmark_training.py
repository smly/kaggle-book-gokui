from pathlib import Path
from typing import Any

from timm.utils import AverageMeter
import albumentations as albu
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from gldv2dataset import get_dataloaders
from model import AngularModel


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


def aug_scalerotate(input_size):
    train_transform = albu.Compose([
        albu.RandomResizedCrop(
            input_size,
            input_size,
            scale=(0.6, 1.0),
            p=1.0,
        ),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=15, p=0.3),
        albu.Normalize(),
    ])
    test_transform = albu.Compose([
        albu.Resize(width=input_size, height=input_size),
        albu.Normalize(),
    ])
    return train_transform, test_transform


def aug_scalerotate_colorjit(input_size):
    train_transform = albu.Compose([
        albu.RandomResizedCrop(
            input_size,
            input_size,
            scale=(0.6, 1.0),
            p=1.0,
        ),
        albu.OneOf([
            albu.RandomBrightness(0.1, p=0.5),
            albu.RandomContrast(0.1, p=0.5),
            albu.RandomGamma(p=0.5)
        ], p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=15, p=0.3),
        albu.Normalize(),
    ])
    test_transform = albu.Compose([
        albu.Resize(width=input_size, height=input_size),
        albu.Normalize(),
    ])
    return train_transform, test_transform


def aug_scalerotate_colorjit_cutoff(input_size):
    train_transform = albu.Compose([
        albu.RandomResizedCrop(
            input_size,
            input_size,
            scale=(0.6, 1.0),
            p=1.0,
        ),
        albu.OneOf([
            albu.RandomBrightness(0.1, p=0.5),
            albu.RandomContrast(0.1, p=0.5),
            albu.RandomGamma(p=0.5)
        ], p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=15, p=0.3),
        albu.Cutout(p=0.5, max_h_size=80, max_w_size=80, num_holes=2),
        albu.Normalize(),
    ])
    test_transform = albu.Compose([
        albu.Resize(width=input_size, height=input_size),
        albu.Normalize(),
    ])
    return train_transform, test_transform


def aug_scalerotate_colorjit_cutoff_blur(input_size):
    train_transform = albu.Compose([
        albu.RandomResizedCrop(
            input_size,
            input_size,
            scale=(0.6, 1.0),
            p=1.0,
        ),
        albu.OneOf([
            albu.RandomBrightness(0.1, p=0.5),
            albu.RandomContrast(0.1, p=0.5),
            albu.RandomGamma(p=0.5)
        ], p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=15, p=0.3),
        albu.Cutout(p=0.5, max_h_size=80, max_w_size=80, num_holes=2),
        albu.Blur(blur_limit=(7, 7), p=0.3),
        albu.Normalize(),
    ])
    test_transform = albu.Compose([
        albu.Resize(width=input_size, height=input_size),
        albu.Normalize(),
    ])
    return train_transform, test_transform


def train(
    exp_name,
    backbone,
    input_size,
    augmentation,
    batch_size,
    num_epochs,
):
    path_train_csv = "gldv2_micro/train.csv"
    path_val_csv = "gldv2_micro/val.csv"
    num_workers = 8
    init_lr = 1e-4
    device = "cuda"

    train_transform, val_transform = augmentation(input_size)
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
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)
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

        save_checkpoint(model, epoch+1, Path(f"out/{exp_name}_last.pth"))


def run_experiment(backbone="resnet34", input_size=128, augmentation=aug_scalerotate):
    exp_name = f"{backbone}_size{input_size}_{augmentation.__name__}"
    print(f"Experiment name: {exp_name}")

    num_epochs = 10
    batch_size = 16
    train(exp_name, backbone, input_size, augmentation, batch_size, num_epochs)


def make_submission(backbone="resnet34", input_size=128, augmentation=aug_scalerotate):
    exp_name = f"{backbone}_size{input_size}_{augmentation.__name__}"


def main():
    # 様々なデータ拡張の手法を比較する
    run_experiment(backbone="resnet34",
                   input_size=128,
                   augmentation=aug_scalerotate)
    run_experiment(backbone="resnet34",
                   input_size=128,
                   augmentation=aug_scalerotate_colorjit)
    run_experiment(backbone="resnet34",
                   input_size=128,
                   augmentation=aug_scalerotate_colorjit_cutoff)
    run_experiment(backbone="resnet34",
                   input_size=128,
                   augmentation=aug_scalerotate_colorjit_cutoff_blur)

    # 大きなモデル、大きな入力サイズでモデルを作成する
    # (Ep 10 / val/loss=5.8804, val/acc=0.3651)
    run_experiment(backbone="resnext101_64x4d",
                   input_size=384,
                   augmentation=aug_scalerotate_colorjit_cutoff)

    # 大きなモデル、大きな入力サイズでモデルを作成する
    # (Ep 10 / val/loss=6.4721, val/acc=0.2826)
    run_experiment(backbone="resnest101e",
                   input_size=384,
                   augmentation=aug_scalerotate_colorjit_cutoff)


if __name__ == "__main__":
    main()