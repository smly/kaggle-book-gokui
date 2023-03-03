import argparse
import copy
import os
import pathlib

import numpy as np
import sklearn.model_selection
import timm
import torch
import torchvision
import torchvision.transforms.functional
from torchvision import transforms
from tqdm import tqdm

CLIP_THRESHOLD = 0.0125


# コード引用あり＠5節
def setup_train_val_split(labels, dryrun=False, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, random_state=seed
    )
    train_indices, val_indices = next(splitter.split(x, y))

    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace=False)
        val_indices = np.random.choice(val_indices, 100, replace=False)

    return train_indices, val_indices


# ↓コード引用あり@8節
def setup_cv_split(labels, n_folds, fold, dryrun=False, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )
    splits = list(splitter.split(x, y))
    train_indices, val_indices = splits[fold]

    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace=False)
        val_indices = np.random.choice(val_indices, 100, replace=False)

    return train_indices, val_indices


def get_labels(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return get_labels(dataset.dataset)[dataset.indices]
    else:
        return np.array([img[1] for img in dataset.imgs])


# コード引用あり＠5節
def setup_train_val_datasets(data_dir, dryrun=False):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=setup_center_crop_transform(),
    )
    labels = get_labels(dataset)
    train_indices, val_indices = setup_train_val_split(labels, dryrun)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset


# コード引用あり＠8節
def setup_cv_datasets(data_dir, n_folds, fold, dryrun=False):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=setup_center_crop_transform(),
    )
    labels = get_labels(dataset)
    train_indices, val_indices = setup_cv_split(labels, n_folds, fold, dryrun)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset


def set_transform(dataset, transform):
    if isinstance(dataset, torch.utils.data.Subset):
        set_transform(dataset.dataset, transform)
    else:
        dataset.transform = transform


# コード引用あり＠5節
def setup_train_val_loaders(data_dir, batch_size, dryrun=False):
    train_dataset, val_dataset = setup_train_val_datasets(
        data_dir, dryrun=dryrun
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )
    return train_loader, val_loader


########################################################################################################################
# transform
########################################################################################################################

# コード引用あり＠5節
def setup_center_crop_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# コード引用あり＠7節
def setup_crop_flip_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# コード引用あり＠8-2
def setup_tta_transforms():
    return [
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                torchvision.transforms.functional.hflip,
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    ]


########################################################################################################################
# train loop 5節（lr schedulerなし）
########################################################################################################################

# コード引用あり＠5節
def train_1epoch(model, train_loader, lossfun, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = lossfun(out, y)
        _, pred = torch.max(out.detach(), 1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


# コード引用あり＠5節
def validate_1epoch(model, val_loader, lossfun, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = lossfun(out.detach(), y)
            _, pred = torch.max(out, 1)

            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss


# コード引用あり＠5節
def train(model, optimizer, train_loader, val_loader, n_epochs, device):
    lossfun = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):
        train_acc, train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)
        print(
            f"epoch={epoch}, train loss={train_loss}, train accuracy={train_acc}, val loss={val_loss}, val accuracy={val_acc}"
        )


########################################################################################################################
# train loop 6節以降（lr schedulerあり）
########################################################################################################################

# コード引用あり＠5節
def train_1epoch2(
    model, train_loader, lossfun, optimizer, lr_scheduler, device
):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        _, pred = torch.max(out.detach(), 1)
        loss = lossfun(model(x), y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def train2(
    model, optimizer, lr_scheduler, train_loader, val_loader, n_epochs, device
):
    lossfun = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):
        train_acc, train_loss = train_1epoch2(
            model, train_loader, lossfun, optimizer, lr_scheduler, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch}, train loss={train_loss}, train accuracy={train_acc}, val loss={val_loss}, val accuracy={val_acc}, lr={lr}"
        )


########################################################################################################################
# mixup
########################################################################################################################


# コード引用あり@7-3節
def train_1epoch_mixup(
    model, train_loader, lossfun, optimizer, lr_scheduler, mixup_alpha, device
):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        lmd = np.random.beta(mixup_alpha, mixup_alpha)
        perm = torch.randperm(x.shape[0]).to(device)
        x2 = x[perm, :]
        y2 = y[perm]

        optimizer.zero_grad()
        out = model(lmd * x + (1.0 - lmd) * x2)
        loss = lmd * lossfun(out, y) + (1.0 - lmd) * lossfun(out, y2)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        _, pred = torch.max(out.detach(), 1)
        total_loss += loss.item() * x.size(0)
        total_acc += lmd * torch.sum(pred == y) + (1.0 - lmd) * torch.sum(
            pred == y2
        )

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def train3_mixup(
    model,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    n_epochs,
    n_mixup_epochs,
    mixup_alpha,
    device,
):
    lossfun = torch.nn.CrossEntropyLoss()
    last_val_loss = 0.0

    for epoch in tqdm(range(n_epochs)):
        if epoch < n_mixup_epochs:
            train_acc, train_loss = train_1epoch_mixup(
                model,
                train_loader,
                lossfun,
                optimizer,
                lr_scheduler,
                mixup_alpha,
                device,
            )
        else:
            train_acc, train_loss = train_1epoch2(
                model, train_loader, lossfun, optimizer, lr_scheduler, device
            )

        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)
        last_val_loss = val_loss

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch}, train loss={train_loss}, train accuracy={train_acc}, val loss={val_loss}, val accuracy={val_acc}, lr={lr}"
        )

    return last_val_loss


########################################################################################################################
# predict部分
########################################################################################################################

# コード引用あり＠5節
def setup_test_loader(data_dir, batch_size, dryrun):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=setup_center_crop_transform()
    )
    image_ids = [
        os.path.splitext(os.path.basename(path))[0] for path, _ in dataset.imgs
    ]

    if dryrun:
        dataset = torch.utils.data.Subset(dataset, range(0, 100))
        image_ids = image_ids[:100]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8
    )
    return loader, image_ids


# コード引用あり＠5節
def predict(model, loader, device):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []
    for x, _ in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = x.to(device)
            y = pred_fun(model(x))
        y = y.cpu().numpy()
        y = y[:, 1]  # cat:0, dog: 1
        preds.append(y)
    preds = np.concatenate(preds)
    return preds


# コード引用あり＠5節
def write_prediction(image_ids, prediction, out_path):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            f.write("{},{}\n".format(i, p))


# コード引用あり＠5節
def write_prediction_with_clip(
    image_ids, prediction, clip_threshold, out_path
):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            p = np.clip(p, clip_threshold, 1.0 - clip_threshold)
            f.write("{},{}\n".format(i, p))


########################################################################################################################
# 各種実行設定
########################################################################################################################

#
# 5: First try
#

# コード引用あり＠5節
def train_subsec5(data_dir, batch_size, dryrun=False, device="cuda:0"):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, val_loader = setup_train_val_loaders(
        data_dir, batch_size, dryrun
    )
    train(
        model, optimizer, train_loader, val_loader, n_epochs=1, device=device
    )

    return model


# コード引用あり＠5節
def predict_subsec5(
    data_dir, out_dir, model, batch_size, dryrun=False, device="cuda:0"
):
    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    preds = predict(model, test_loader, device)
    write_prediction(image_ids, preds, out_dir / "out.csv")


def run_5(data_dir, out_dir, dryrun, device):
    batch_size = 32
    model = train_subsec5(data_dir, batch_size, dryrun, device)

    # clip無しの推論
    predict_subsec5(data_dir, out_dir, model, batch_size, dryrun, device)

    # clip有りの推論
    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    preds = predict(model, test_loader, device)
    write_prediction_with_clip(
        image_ids, preds, CLIP_THRESHOLD, out_dir / "out_clip.csv"
    )


#
# 6: Weight decay, cosine annealing
#


def run_6(data_dir, out_dir, dryrun, device):
    batch_size = 32
    n_epochs = 2 if dryrun else 10

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    train_dataset, val_dataset = setup_train_val_datasets(
        data_dir, dryrun=dryrun
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    )
    n_iterations = len(train_loader) * n_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_iterations
    )

    train2(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        device=device,
    )

    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    preds = predict(model, test_loader, device)
    write_prediction_with_clip(
        image_ids, preds, CLIP_THRESHOLD, out_dir / "out_clip.csv"
    )


#
# 7-1: Random crop&flip
#


def run_7_1(data_dir, out_dir, dryrun, device):
    batch_size = 32
    n_epochs = 2 if dryrun else 10

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    train_dataset, val_dataset = setup_train_val_datasets(
        data_dir, dryrun=dryrun
    )
    train_dataset = copy.deepcopy(
        train_dataset
    )  # transformを設定した際にval_datasetに影響したくない
    train_transform = setup_crop_flip_transform()
    set_transform(train_dataset, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    )
    n_iterations = len(train_loader) * n_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_iterations
    )

    train2(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        device=device,
    )

    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    preds = predict(model, test_loader, device)
    write_prediction_with_clip(
        image_ids, preds, CLIP_THRESHOLD, out_dir / "out_clip.csv"
    )


#
# 7-3: Mixup
#


def run_7_3(data_dir, out_dir, dryrun, device):
    batch_size = 32
    n_epochs = 2 if dryrun else 10
    n_mixup_epochs = 1 if dryrun else 7
    mixup_alpha = 0.4

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    train_dataset, val_dataset = setup_train_val_datasets(
        data_dir, dryrun=dryrun
    )
    train_dataset = copy.deepcopy(
        train_dataset
    )  # transformを設定した際にval_datasetに影響したくない
    train_transform = setup_crop_flip_transform()
    set_transform(train_dataset, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    )
    n_iterations = len(train_loader) * n_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_iterations
    )

    train3_mixup(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        n_mixup_epochs=n_mixup_epochs,
        mixup_alpha=mixup_alpha,
        device=device,
    )

    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    preds = predict(model, test_loader, device)
    write_prediction_with_clip(
        image_ids, preds, CLIP_THRESHOLD, out_dir / "out_clip.csv"
    )


#
# 8-1: Averaging
#

# コード引用はしないけど引用されてるコードから呼ばれてる@8-1
def train_predict_1fold(
    data_dir,
    fold,
    n_folds,
    n_epochs,
    n_mixup_epochs,
    mixup_alpha,
    test_loader,
    batch_size,
    dryrun=False,
    device="cuda:0",
):
    print(f"Fold {fold}")
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)
    train_dataset, val_dataset = setup_cv_datasets(
        data_dir, n_folds, fold, dryrun=dryrun
    )
    train_dataset = copy.deepcopy(train_dataset)
    train_transform = setup_crop_flip_transform()
    set_transform(train_dataset, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    )
    n_iterations = len(train_loader) * n_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_iterations
    )

    val_loss = train3_mixup(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        n_mixup_epochs=n_mixup_epochs,
        mixup_alpha=mixup_alpha,
        device=device,
    )

    test_pred = predict(model, test_loader, device)
    return val_loss, test_pred


# コード引用あり@8-1
def train_predict_subsec81(
    data_dir,
    out_dir,
    n_folds,
    n_epochs,
    n_mixup_epochs,
    mixup_alpha,
    batch_size,
    dryrun=False,
    device="cuda:0",
):
    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )

    val_losses = []
    test_preds = []
    for fold in range(n_folds):
        val_loss, test_pred = train_predict_1fold(
            data_dir,
            fold,
            n_folds,
            n_epochs,
            n_mixup_epochs,
            mixup_alpha,
            test_loader,
            batch_size,
            dryrun,
            device,
        )
        val_losses.append(val_loss)
        test_preds.append(test_pred)

    # 全foldでのvalidation scoreの平均を最終的なvalidation scoreとして出力
    val_loss = np.mean(val_losses)
    print(f"val loss={val_loss}")

    # 全foldでのモデルのtest dataに対する予測結果を平均したものを最終的な予測として出力
    test_pred = np.mean(test_preds, axis=0)
    write_prediction_with_clip(
        image_ids, test_pred, CLIP_THRESHOLD, out_dir / "out_clip.csv"
    )


def run_8_1(data_dir, out_dir, dryrun, device):
    batch_size = 32
    n_epochs = 2 if dryrun else 10
    n_mixup_epochs = 1 if dryrun else 7
    n_folds = 2 if dryrun else 5
    mixup_alpha = 0.4

    train_predict_subsec81(
        data_dir,
        out_dir,
        n_folds,
        n_epochs,
        n_mixup_epochs,
        mixup_alpha,
        batch_size,
        dryrun,
        device,
    )


#
# 8-2: TTA
#


# コード引用あり＠8−2
def predict_tta(model, loader, device):
    tta_transforms = setup_tta_transforms()
    preds = []
    for transform in tta_transforms:
        set_transform(loader.dataset, transform)
        val_pred = predict(model, loader, device)
        preds.append(val_pred)

    pred = np.mean(preds, axis=0)
    return pred


# TTA
def run_8_2(data_dir, out_dir, dryrun, device):
    batch_size = 32
    n_epochs = 2 if dryrun else 10
    n_mixup_epochs = 1 if dryrun else 7
    n_folds = 2 if dryrun else 5
    mixup_alpha = 0.4

    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    val_losses = []
    test_preds = []

    for fold in range(n_folds):
        print(f"Fold {fold}")
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.to(device)

        train_dataset, val_dataset = setup_cv_datasets(
            data_dir, n_folds, fold, dryrun=dryrun
        )
        train_dataset = copy.deepcopy(train_dataset)
        train_transform = setup_crop_flip_transform()
        set_transform(train_dataset, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=8
        )

        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
        )
        n_iterations = len(train_loader) * n_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, n_iterations
        )

        train3_mixup(
            model,
            optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
            n_epochs=n_epochs,
            n_mixup_epochs=n_mixup_epochs,
            mixup_alpha=mixup_alpha,
            device=device,
        )

        val_pred = predict_tta(model, val_loader, device)
        val_labels = get_labels(val_loader.dataset)
        val_loss = sklearn.metrics.log_loss(val_labels, val_pred)
        val_losses.append(val_loss)
        print(f"fold={fold}, val loss with TTA={val_loss}")

        test_pred = predict_tta(model, test_loader, device)
        test_preds.append(test_pred)

    val_loss = np.mean(val_losses)
    print(f"val loss={val_loss}")

    test_pred = np.mean(test_preds, axis=0)
    write_prediction_with_clip(
        image_ids, test_pred, CLIP_THRESHOLD, out_dir / "out_clip.csv"
    )


# TTA
def run_9_1(data_dir, out_dir, dryrun, device):
    batch_size = 32
    n_epochs = 2 if dryrun else 10
    n_mixup_epochs = 1 if dryrun else 7
    n_folds = 2 if dryrun else 5
    mixup_alpha = 0.4

    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    val_losses = []
    test_preds = []

    for fold in range(n_folds):
        print(f"Fold {fold}")
        model = timm.create_model(
            "regnety_080", pretrained=True, num_classes=2
        )
        model.to(device)

        train_dataset, val_dataset = setup_cv_datasets(
            data_dir, n_folds, fold, dryrun=dryrun
        )
        train_dataset = copy.deepcopy(train_dataset)
        train_transform = setup_crop_flip_transform()
        set_transform(train_dataset, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=8
        )

        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
        )
        n_iterations = len(train_loader) * n_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, n_iterations
        )

        train3_mixup(
            model,
            optimizer,
            lr_scheduler,
            train_loader,
            val_loader,
            n_epochs=n_epochs,
            n_mixup_epochs=n_mixup_epochs,
            mixup_alpha=mixup_alpha,
            device=device,
        )

        val_pred = predict_tta(model, val_loader, device)
        val_labels = get_labels(val_loader.dataset)
        val_loss = sklearn.metrics.log_loss(val_labels, val_pred)
        val_losses.append(val_loss)
        print(f"fold={fold}, val loss with TTA={val_loss}")

        test_pred = predict_tta(model, test_loader, device)
        test_preds.append(test_pred)

    val_loss = np.mean(val_losses)
    print(f"val loss={val_loss}")

    test_pred = np.mean(test_preds, axis=0)
    write_prediction_with_clip(
        image_ids, test_pred, CLIP_THRESHOLD, out_dir / "out_clip.csv"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    device = args.device
    dryrun = args.dryrun

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.config == "5":
        run_5(data_dir, out_dir, dryrun, device)
    elif args.config == "6":
        run_6(data_dir, out_dir, dryrun, device)
    elif args.config == "7-1":
        run_7_1(data_dir, out_dir, dryrun, device)
    elif args.config == "7-3":
        run_7_3(data_dir, out_dir, dryrun, device)
    elif args.config == "8-1":
        run_8_1(data_dir, out_dir, dryrun, device)
    elif args.config == "8-2":
        run_8_2(data_dir, out_dir, dryrun, device)
    elif args.config == "9-1":
        run_9_1(data_dir, out_dir, dryrun, device)
    else:
        raise ValueError(f"Unknown config: {args.config}")


if __name__ == "__main__":
    main()
