import time

import datasets
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def setup_adamw_optimizer(model, learning_rate, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_parameters, lr=learning_rate)


def create_dataset(df, tokenizer, max_length, remove_columns):
    def preprocess(examples):
        return tokenizer(
            examples["question1"],
            examples["question2"],
            truncation="longest_first",
            max_length=max_length,
        )

    raw_dataset = datasets.Dataset.from_pandas(df)
    return raw_dataset.map(
        preprocess,
        batched=True,
        remove_columns=remove_columns,
        desc="Running tokenizer on dataset",
    )


def train_1epoch(model, optimizer, scheduler, data_loader, device):
    model.train()
    losses = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Training"):
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        targets = batch.pop("is_duplicate").view(-1, 1).float()
        weights = batch.pop("weight").view(-1, 1)
        optimizer.zero_grad()
        logits = model(**batch)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weights
        )
        loss.backward()
        losses.append(loss.detach().cpu().item())
        optimizer.step()
        scheduler.step()

    return float(np.mean(losses))


def predict_logits(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        logits = []
        losses = []
        for batch in tqdm(
            data_loader, total=len(data_loader), desc="Predicting"
        ):
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            targets = batch.pop("is_duplicate").view(-1, 1).float()
            weights = batch.pop("weight").view(-1, 1)
            batch_logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(
                batch_logits, targets, weight=weights, reduction="none"
            )
            logits.append(batch_logits.float().detach().cpu())
            losses.append(loss.float().detach().cpu())
        logits = torch.cat(logits, 0)
        losses = torch.cat(losses, 0)
        return logits, losses.mean().item()


def train_model(model, model_path, trn_loader, val_loader, device, config):
    num_epochs = config["num_epochs"]
    tolerance = config.get("tolerance", num_epochs)

    start_time = time.time()
    optimizer = setup_adamw_optimizer(
        model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    num_training_steps = len(trn_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * config["warmup_step_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_training_steps, num_warmup_steps
    )

    best_trn_loss = None
    best_val_loss = None
    best_epoch = -1
    no_improvement_count = 0

    for epoch in range(num_epochs):
        trn_loss = train_1epoch(
            model, optimizer, scheduler, trn_loader, device
        )
        _, val_loss = predict_logits(model, val_loader, device)

        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}:\ttrn-loss={trn_loss:.4f}\tval-loss={val_loss:.4f}\t"
            f"time={elapsed_time:.2f}s"
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_trn_loss = trn_loss
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), str(model_path))
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count > tolerance:
            break

    print(
        f"Finished training: best_epoch={best_epoch}\t"
        f"best_val_loss={best_val_loss}\ttotal-time={time.time() - start_time:.2f}s"
    )

    model.load_state_dict(torch.load(model_path))
    fold_result = {
        "trn_loss": best_trn_loss,
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "time": time.time() - start_time,
    }
    return model, fold_result


def train_1epoch_amp(model, optimizer, scheduler, data_loader, device):
    model.train()
    losses = []
    scaler = GradScaler()

    for batch in tqdm(data_loader, total=len(data_loader), desc="Training"):
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        targets = batch.pop("is_duplicate").view(-1, 1).float()
        weights = batch.pop("weight").view(-1, 1)
        optimizer.zero_grad()

        with autocast():
            logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.detach().float().cpu().item())

    return float(np.mean(losses))


def predict_logits_amp(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        logits = []
        losses = []
        for batch in tqdm(
            data_loader, total=len(data_loader), desc="Predicting"
        ):
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            targets = batch.pop("is_duplicate").view(-1, 1).float()
            weights = batch.pop("weight").view(-1, 1)

            with autocast():
                batch_logits = model(**batch)
                loss = F.binary_cross_entropy_with_logits(
                    batch_logits, targets, weight=weights, reduction="none"
                )
            logits.append(batch_logits.float().detach().cpu())
            losses.append(loss.float().detach().cpu())
        logits = torch.cat(logits, 0)
        losses = torch.cat(losses, 0)
        return logits, losses.mean().item()


def train_model_amp(model, model_path, trn_loader, val_loader, device, config):
    num_epochs = config["num_epochs"]
    tolerance = config.get("tolerance", num_epochs)

    start_time = time.time()
    optimizer = setup_adamw_optimizer(
        model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    num_training_steps = len(trn_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * config["warmup_step_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_training_steps, num_warmup_steps
    )

    best_trn_loss = None
    best_val_loss = None
    best_epoch = -1
    no_improvement_count = 0

    for epoch in range(num_epochs):
        trn_loss = train_1epoch_amp(
            model, optimizer, scheduler, trn_loader, device
        )
        _, val_loss = predict_logits_amp(model, val_loader, device)

        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}:\ttrn-loss={trn_loss:.4f}\tval-loss={val_loss:.4f}\t"
            f"time={elapsed_time:.2f}s"
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_trn_loss = trn_loss
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), str(model_path))
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count > tolerance:
            break

    print(
        f"Finished training: best_epoch={best_epoch}\t"
        f"best_val_loss={best_val_loss}\ttotal-time={time.time() - start_time:.2f}s"
    )

    model.load_state_dict(torch.load(model_path))
    fold_result = {
        "trn_loss": best_trn_loss,
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "time": time.time() - start_time,
    }
    return model, fold_result
