import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset

from common.constants import SPLIT_RANDOM_SEED, TRAIN_CSV_PATH
from common.utils import compute_weights, seed_everything, simple_timer
from experiments.rnn_common import (
    LSTMExtractor,
    get_full_features,
    predict_logits,
    train_model,
)
from experiments.utils import (
    default_cli,
    get_model_output_path,
    get_num_test_samples,
    get_num_train_samples,
    get_test_prediction_output_path,
    save_results,
)
from texts.preprocessing import EmbeddingKey, PreprocessingKey
from texts.rnn_inputs import build_rnn_inputs


class LSTMSiameseModelV2(nn.Module):
    def __init__(self, embeddings, num_features, hidden_units, dropout=0.2):
        super(LSTMSiameseModelV2, self).__init__()
        self.extractor = LSTMExtractor(embeddings, hidden_units)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(8 * hidden_units + num_features, 2 * hidden_units),
            nn.BatchNorm1d(2 * hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_units, 1),
        )

    def forward(self, q1_token_ids, q2_token_ids, features):
        q1_features = self.extractor(q1_token_ids)
        q2_features = self.extractor(q2_token_ids)
        features = torch.cat(
            [
                torch.abs(q1_features - q2_features),
                q1_features * q2_features,
                features,
            ],
            1,
        )
        return self.classifier(features)


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
        embedding_matrix, q1_tensor, q2_tensor = build_rnn_inputs(
            dataset_key=PreprocessingKey.NLTK_TOKENIZATION,
            embedding_key=EmbeddingKey.GLOVE,
            max_length=config["max_length"],
        )
        q1_tensor = q1_tensor[:num_train_samples]
        q2_tensor = q2_tensor[:num_train_samples]
        features = get_full_features()[:num_train_samples]
        dataset = TensorDataset(
            q1_tensor,
            q2_tensor,
            torch.tensor(features).float(),
            torch.tensor(trn_df.is_duplicate).view(-1, 1).float(),
            torch.tensor(trn_df.weight).view(-1, 1),
        )

    skf = StratifiedKFold(
        config["n_splits"], shuffle=True, random_state=SPLIT_RANDOM_SEED
    )
    trn_idx, val_idx = list(skf.split(trn_df, trn_df.is_duplicate))[fold_id]
    trn_dataset = Subset(dataset, trn_idx)
    val_dataset = Subset(dataset, val_idx)

    seed_everything(fold_id)
    model = LSTMSiameseModelV2(
        embedding_matrix,
        num_features=features.shape[1],
        hidden_units=config["hidden_units"],
    ).to(device)

    batch_size = config["batch_size"]
    trn_loader = DataLoader(
        trn_dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size, num_workers=num_workers)
    model, trn_result = train_model(
        model,
        get_model_output_path(output_dir, fold_id),
        trn_loader=trn_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )

    with simple_timer("Predict OOF targets"):
        oof_predictions = np.full(len(trn_df), np.nan)
        val_logits, _ = predict_logits(model, val_loader, device=device)
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

    with simple_timer("Load test dataset"):
        num_test_samples = get_num_test_samples(dryrun)
        embedding_matrix, q1_tensor, q2_tensor = build_rnn_inputs(
            dataset_key=PreprocessingKey.NLTK_TOKENIZATION,
            embedding_key=EmbeddingKey.GLOVE,
            max_length=config["max_length"],
        )
        q1_tensor = q1_tensor[-num_test_samples:]
        q2_tensor = q2_tensor[-num_test_samples:]
        features = get_full_features()[-num_test_samples:]
        tst_dataset = TensorDataset(
            q1_tensor,
            q2_tensor,
            torch.tensor(features).float(),
            torch.zeros(len(q1_tensor)).view(-1, 1).float(),
            torch.ones(len(q1_tensor)).view(-1, 1).float(),
        )
    seed_everything(fold_id)
    model = LSTMSiameseModelV2(
        embedding_matrix,
        num_features=features.shape[1],
        hidden_units=config["hidden_units"],
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    with simple_timer("Predict test targets"):
        tst_loader = DataLoader(
            tst_dataset, config["batch_size"], num_workers=num_workers
        )
        (
            tst_logits,
            _,
        ) = predict_logits(model, tst_loader, device=device)
        tst_df = pd.DataFrame(
            {
                "test_id": np.arange(num_test_samples),
                "is_duplicate": torch.sigmoid(tst_logits).flatten().numpy(),
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
            "learning_rate": 1e-3,
            "batch_size": 512,
            "num_epochs": 10,
            "n_splits": 5,
            "hidden_units": 100,
            "max_length": 55,
            "target_positive_ratio": 0.174,
        },
    )()
