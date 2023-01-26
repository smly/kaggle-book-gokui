import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from common.constants import SPLIT_RANDOM_SEED, TEST_CSV_PATH, TRAIN_CSV_PATH
from common.utils import compute_weights, seed_everything, simple_timer
from experiments.rnn_common import get_full_features
from experiments.transformer_common import (
    create_dataset,
    predict_logits_amp,
    train_model_amp,
)
from experiments.utils import (
    default_cli,
    get_model_output_path,
    get_num_test_samples,
    get_num_train_samples,
    get_test_prediction_output_path,
    save_results,
)

transformers.logging.set_verbosity_error()


def masked_mean(hidden_state, mask):
    mask = mask[:, :, np.newaxis].float()
    hidden_state_sum = torch.sum(hidden_state * mask, 1)
    mask_sum = torch.clamp(torch.sum(mask, 1), min=1e-5)
    return hidden_state_sum / mask_sum


def masked_max(hidden_state, mask):
    mask = (1 - mask[:, :, np.newaxis]).float()
    hidden_state = hidden_state - mask * 1e3
    return torch.max(hidden_state, 1)[0]


class TransformerModelV4(nn.Module):
    def __init__(
        self, model_name_or_path, feature_dim, hidden_units=200, dropout=0.2
    ):
        super(TransformerModelV4, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.encoder.config.hidden_size * 2 + feature_dim, hidden_units
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(
        self, input_ids, features, attention_mask=None, token_type_ids=None
    ):
        bert_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return self.classifier(
            torch.cat(
                (
                    masked_mean(bert_output.last_hidden_state, attention_mask),
                    masked_max(bert_output.last_hidden_state, attention_mask),
                    features,
                ),
                1,
            )
        )


def train_1fold(output_dir, fold_id, config, num_workers, device, dryrun):
    print(
        f"Start training (fold_id = {fold_id}, config = {json.dumps(config)})"
    )

    with simple_timer("Load training dataset"):
        num_train_samples = get_num_train_samples(dryrun)
        trn_df = pd.read_csv(
            TRAIN_CSV_PATH, nrows=num_train_samples, na_filter=False
        )
        trn_df["weight"] = compute_weights(
            trn_df.is_duplicate, config["target_positive_ratio"]
        )
        trn_df = trn_df.iloc[:num_train_samples]

    with simple_timer("Load tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])

    with simple_timer("Load features"):
        features = get_full_features()[:num_train_samples].astype(np.float32)
        trn_df["features"] = features.tolist()

    skf = StratifiedKFold(
        config["n_splits"], shuffle=True, random_state=SPLIT_RANDOM_SEED
    )
    trn_idx, val_idx = list(skf.split(trn_df, trn_df.is_duplicate))[fold_id]
    seed_everything(fold_id)

    max_length = config["max_length"]
    remove_columns = ["id", "qid1", "qid2", "question1", "question2"]
    transformers.logging.set_verbosity_error()
    trn_dataset = create_dataset(
        trn_df.iloc[trn_idx].reset_index(drop=True),
        tokenizer,
        max_length,
        remove_columns,
    )
    val_dataset = create_dataset(
        trn_df.iloc[val_idx].reset_index(drop=True),
        tokenizer,
        max_length,
        remove_columns,
    )
    model = TransformerModelV4(
        config["pretrained_model"], features.shape[1]
    ).to(device)

    batch_size = config["batch_size"]
    collator_fn = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest"
    )
    trn_loader = DataLoader(
        trn_dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collator_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size,
        num_workers=num_workers,
        collate_fn=collator_fn,
    )
    model, trn_result = train_model_amp(
        model,
        model_path=get_model_output_path(output_dir, fold_id),
        trn_loader=trn_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )

    with simple_timer("Predict OOF targets"):
        oof_predictions = np.full(len(trn_df), np.nan)
        val_logits, _ = predict_logits_amp(model, val_loader, device=device)
        oof_predictions[val_idx] = torch.sigmoid(val_logits).flatten().numpy()
        oof_df = pd.DataFrame(
            {"id": trn_df.id, "is_duplicate": oof_predictions}
        )

    torch.save(
        model.state_dict(), str(get_model_output_path(output_dir, fold_id))
    )
    save_results(output_dir, config, fold_id, result=trn_result, oof_df=oof_df)


def predict_1fold(output_dir, fold_id, config, num_workers, device, dryrun):
    model_path = get_model_output_path(output_dir, fold_id)
    assert model_path.exists()

    # テストデータのサンプル数が多いため、OOMを避けるために500000件ずつ予測を行う。
    block_size = 500000
    num_test_samples = get_num_test_samples(dryrun)

    with simple_timer("Load test dataset"):
        all_tst_df = pd.read_csv(
            TEST_CSV_PATH, nrows=num_test_samples, na_filter=False
        )
        all_tst_df["weight"] = np.ones(len(all_tst_df))
        all_tst_df["is_duplicate"] = np.zeros(len(all_tst_df))

    with simple_timer("Load tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])

    with simple_timer("Load features"):
        features = get_full_features()[-num_test_samples:].astype(np.float32)

    with simple_timer("Load model"):
        model = TransformerModelV4(
            config["pretrained_model"], features.shape[1]
        ).to(device)
        model.load_state_dict(torch.load(model_path))

    seed_everything(fold_id)

    tst_ids = []
    tst_predictions = []
    for s_idx in range(0, num_test_samples, block_size):
        e_idx = min(s_idx + block_size, num_test_samples)
        print(f"Predict test targets [{s_idx}, {e_idx})")

        tst_df = all_tst_df.iloc[s_idx:e_idx].copy()
        tst_df["features"] = features[s_idx:e_idx].tolist()
        max_length = config["max_length"]
        remove_columns = ["test_id", "question1", "question2"]
        tst_dataset = create_dataset(
            tst_df, tokenizer, max_length, remove_columns
        )
        collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest", max_length=max_length
        )
        tst_loader = DataLoader(
            tst_dataset,
            config["batch_size"],
            num_workers=num_workers,
            collate_fn=collator_fn,
        )
        (
            tst_logits,
            _,
        ) = predict_logits_amp(model, tst_loader, device=device)
        tst_ids.append(tst_df["test_id"].values)
        tst_predictions.append(torch.sigmoid(tst_logits).flatten().numpy())

    with simple_timer("Predict test targets"):
        tst_df = pd.DataFrame(
            {
                "test_id": np.concatenate(tst_ids),
                "is_duplicate": np.concatenate(tst_predictions),
            }
        )

    output_file = get_test_prediction_output_path(output_dir, fold_id)
    tst_df.to_csv(
        output_file, columns=["test_id", "is_duplicate"], index=False
    )


if __name__ == "__main__":
    default_cli(
        __file__,
        train_1fold,
        predict_1fold,
        {
            "pretrained_model": "bert-base-uncased",
            "learning_rate": 3e-5,
            "batch_size": 32,
            "num_epochs": 2,
            "n_splits": 5,
            "max_length": 83,  # 99.5%-tile of the question BERT token length distribution
            "warmup_step_ratio": 0.1,
            "weight_decay": 0.01,
            "target_positive_ratio": 0.174,
        },
    )()
