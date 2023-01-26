import os
import random

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def seed_torch(seed=1485):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_1epoch(model, train_loader, lossfun, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # yの予測値を算出
        out = model(x)

        # 損失を計算
        loss = lossfun(out, y)
        loss.backward()

        # 勾配を更新
        optimizer.step()

        # バッチ単位の損失を計算
        total_loss += loss.item() * x.size(0)

        # バッチ単位の正答率を計算
        _, pred = torch.max(out, 1)
        total_acc += torch.sum(pred == y.data)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def validate_1epoch(model, val_loader, lossfun, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = lossfun(out, y)
            _, pred = torch.max(out, 1)

            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y.data)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss


def predict(model, loader, device):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []
    for x, _ in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = x.to(device)
            y = pred_fun(model(x))
        y = y.cpu().numpy()
        # y = np.argmax(y, axis=1)
        preds.append(y)
    preds = np.concatenate(preds)
    return preds


if __name__ == "__main__":
    seed_torch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    digits = load_digits()
    X = digits.data
    y = digits.target

    # train と test に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )
    model = model.to(device)

    lossfun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    cv = KFold(n_splits=5)
    oof_train = np.zeros((len(X_train), 10))
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    test_preds = []

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
        X_tr, X_val = X_train[train_index], X_train[valid_index]
        y_tr, y_val = y_train[train_index], y_train[valid_index]

        X_tr = torch.tensor(X_tr, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_tr = torch.tensor(y_tr, dtype=torch.int64)
        y_val = torch.tensor(y_val, dtype=torch.int64)

        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        for ep in range(100):
            train_acc, train_loss = train_1epoch(
                model, train_loader, lossfun, optimizer, device
            )
            valid_acc, valid_loss = validate_1epoch(
                model, val_loader, lossfun, device
            )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc.item())
        valid_accs.append(valid_acc.item())
        oof_train[valid_index] = predict(model, val_loader, device)
        test_preds.append(predict(model, test_loader, device))

    print("train acc", np.mean(train_accs))
    print("valid acc (各foldの平均)", np.mean(valid_accs))
    print(
        "valid acc (再計算)",
        sum(y_train == np.argmax(oof_train, axis=1)) / len(y_train),
    )

    test_preds = np.mean(test_preds, axis=0)
    test_preds = np.argmax(test_preds, axis=1)

    print(
        "test acc",
        sum(y_test.detach().numpy().copy() == test_preds) / len(y_test),
    )
