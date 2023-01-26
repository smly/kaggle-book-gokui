import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm

from features.decomposition import (
    DecompositionType,
    VectorizerType,
    build_decomposition_features,
)
from features.edit_distance import build_edit_distance_features
from features.graph import (
    build_graph_connected_component_features,
    build_graph_link_prediction_features,
    build_graph_node_features,
)
from features.length import build_length_features
from features.magic import build_magic_features
from features.match import build_match_features
from features.word_vector import (
    build_farthest_word_distance_features,
    build_wmd_features,
)
from texts.preprocessing import EmbeddingKey, PreprocessingKey, StopwordsKey


class LSTMExtractor(nn.Module):
    def __init__(
        self, embedding_matrix, hidden_units, num_layers=2, dropout=0.2
    ):
        super(LSTMExtractor, self).__init__()
        self.embeddings = nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1]
        )
        self.embeddings.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            requires_grad=False,
        )
        self.bilstm = nn.LSTM(
            hidden_size=hidden_units,
            input_size=embedding_matrix.shape[1],
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, xs):
        hidden_units = self.bilstm(self.embeddings(xs))[0]
        return torch.cat(
            [torch.mean(hidden_units, 1), torch.max(hidden_units, 1)[0]], 1
        )


def train_1epoch(model, optimizer, data_loader, device):
    model.train()
    losses = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Training"):
        # batchの最後の要素が重みで、最後から2番目の要素がターゲット、残りの要素がモデルへの入力という仮定をしている。
        *inputs, targets, weights = [b.to(device) for b in batch]
        optimizer.zero_grad()
        logits = model(*inputs)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weights
        )
        loss.backward()
        losses.append(loss.detach().cpu().item())
        optimizer.step()

    return np.mean(losses)


def predict_logits(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        logits = []
        losses = []
        for batch in tqdm(
            data_loader, total=len(data_loader), desc="Predicting"
        ):
            # batchの最後の要素が重みで、最後から2番目の要素がターゲット、残りの要素がモデルへの入力という仮定をしている。
            *inputs, targets, weights = [b.to(device) for b in batch]
            batch_logits = model(*inputs)
            loss = F.binary_cross_entropy_with_logits(
                batch_logits, targets, weight=weights, reduction="none"
            )
            logits.append(batch_logits.detach().cpu())
            losses.append(loss.detach().cpu())
        logits = torch.cat(logits, 0)
        losses = torch.cat(losses, 0)
        return logits, losses.mean().item()


def train_model(model, model_path, trn_loader, val_loader, device, config):
    num_epochs = config["num_epochs"]
    tolerance = config.get("tolerance", num_epochs)

    start_time = time.time()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    best_trn_loss = None
    best_val_loss = None
    best_epoch = -1
    no_improvement_count = 0

    for epoch in range(num_epochs):
        trn_loss = train_1epoch(model, optimizer, trn_loader, device)
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


def get_graph_features():
    features = pd.concat(
        (
            build_magic_features(),
            build_graph_link_prediction_features(),
            build_graph_node_features(),
            build_graph_connected_component_features(),
        ),
        axis=1,
    )

    print(
        "Remove features: ",
        features.columns[
            ((features == np.inf).sum() > 0) | (features.isnull().sum() > 0)
        ],
    )
    features = features[
        features.columns[
            ((features == np.inf).sum() == 0) & (features.isnull().sum() == 0)
        ]
    ]
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def get_full_features():
    features = pd.concat(
        [
            build_match_features(
                PreprocessingKey.NLTK_STEMMING, StopwordsKey.NLTK_STEMMED, 1
            ),
            build_match_features(
                PreprocessingKey.NLTK_STEMMING, StopwordsKey.NLTK_STEMMED, 2
            ),
            build_match_features(
                PreprocessingKey.SPACE_TOKENIZATION, StopwordsKey.NLTK, 1
            ),
            build_match_features(
                PreprocessingKey.SPACE_TOKENIZATION, StopwordsKey.NLTK, 2
            ),
            build_length_features(
                PreprocessingKey.NLTK_STEMMING, StopwordsKey.NLTK_STEMMED, 1
            ),
            build_length_features(
                PreprocessingKey.NLTK_STEMMING, StopwordsKey.NLTK_STEMMED, 2
            ),
            build_length_features(
                PreprocessingKey.SPACE_TOKENIZATION, StopwordsKey.NLTK, 1
            ),
            build_length_features(
                PreprocessingKey.SPACE_TOKENIZATION, StopwordsKey.NLTK, 2
            ),
            build_edit_distance_features(
                PreprocessingKey.NLTK_STEMMING, StopwordsKey.NONE
            ),
            build_edit_distance_features(
                PreprocessingKey.SPACE_TOKENIZATION, StopwordsKey.NONE
            ),
            build_wmd_features(
                PreprocessingKey.NLTK_TOKENIZATION,
                StopwordsKey.NLTK,
                EmbeddingKey.GLOVE,
                normalize=True,
            ),
            build_wmd_features(
                PreprocessingKey.NLTK_TOKENIZATION,
                StopwordsKey.NLTK,
                EmbeddingKey.GLOVE,
                normalize=False,
            ),
            build_farthest_word_distance_features(
                PreprocessingKey.NLTK_TOKENIZATION,
                StopwordsKey.NLTK,
                EmbeddingKey.GLOVE,
                metric="cosine",
                normalize=True,
            ),
            build_farthest_word_distance_features(
                PreprocessingKey.NLTK_TOKENIZATION,
                StopwordsKey.NLTK,
                EmbeddingKey.GLOVE,
                metric="euclidean",
                normalize=True,
            ),
            build_farthest_word_distance_features(
                PreprocessingKey.NLTK_TOKENIZATION,
                StopwordsKey.NLTK,
                EmbeddingKey.GLOVE,
                metric="euclidean",
                normalize=False,
            ),
            build_decomposition_features(
                PreprocessingKey.NLTK_STEMMING,
                StopwordsKey.NLTK_STEMMED,
                VectorizerType.COUNT,
                DecompositionType.SVD,
                n_components=30,
                ngram_range=(1, 2),
            ),
            build_decomposition_features(
                PreprocessingKey.NLTK_STEMMING,
                StopwordsKey.NLTK_STEMMED,
                VectorizerType.TFIDF_NONE,
                DecompositionType.SVD,
                n_components=30,
                ngram_range=(1, 2),
            ),
            build_magic_features(),
            build_graph_link_prediction_features(),
            build_graph_node_features(),
            build_graph_connected_component_features(),
        ],
        axis=1,
    )

    print(
        "Remove features: ",
        features.columns[
            ((features == np.inf).sum() > 0) | (features.isnull().sum() > 0)
        ],
    )
    features = features[
        features.columns[
            ((features == np.inf).sum() == 0) & (features.isnull().sum() == 0)
        ]
    ]
    scaler = StandardScaler()
    return scaler.fit_transform(features)
