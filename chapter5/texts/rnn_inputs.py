import itertools

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from common.constants import FEATURE_MEMORY
from texts.preprocessing import get_dataset, get_embeddings

PADDING_TOKEN_ID = 0  # パディング用のトークンのIDは0として扱う
UNKNOWN_TOKEN_ID = 1  # 訓練済みベクトルに含まれていないトークンのIDは1として扱う
FIRST_TOKEN_ID = 2


def create_embedding_matrix(df, model, max_length):
    print("Start creating embedding matrix")

    # 1. 各質問をトークン化し、トークンの出現順にトークン番号を割り当てていく。
    q1_tokenized = df["question1"].map(lambda q: q.split())
    q2_tokenized = df["question2"].map(lambda q: q.split())
    token_id = FIRST_TOKEN_ID
    token_id_map = {}
    for token in itertools.chain.from_iterable(q1_tokenized + q2_tokenized):
        if token in model and token not in token_id_map:
            token_id_map[token] = token_id
            token_id += 1

    # 2. i行目が番号iのトークンの単語ベクトルに対応する行列を構築する。
    #    UNKNOWN_TOKEN_IDとPADDING_TOKEN_IDにはゼロベクトルを割り当てる。
    embedding_matrix = np.zeros((token_id, model.vector_size))
    for token, i in token_id_map.items():
        embedding_matrix[i, :] = model[token]

    # 3. 質問1と質問2に対応するトークンのリストのリストを、それぞれトークンの番号のリストのリストに変換する。
    #    その後、トークン番号のリストのリストをトークン番号のテンソルに変換する。
    def convert_token_to_id(tokenized_texts):
        def ids_to_tensor(ids):
            return torch.tensor(ids[:max_length], dtype=torch.long)

        return [
            ids_to_tensor([token_id_map.get(t, UNKNOWN_TOKEN_ID) for t in q])
            for q in tokenized_texts
        ]

    q1_tensor = pad_sequence(
        convert_token_to_id(q1_tokenized),
        batch_first=True,
        padding_value=PADDING_TOKEN_ID,
    )
    q2_tensor = pad_sequence(
        convert_token_to_id(q2_tokenized),
        batch_first=True,
        padding_value=PADDING_TOKEN_ID,
    )
    return embedding_matrix, q1_tensor, q2_tensor


@FEATURE_MEMORY.cache
def build_rnn_inputs(dataset_key, embedding_key, max_length):
    df = get_dataset(dataset_key)
    model = get_embeddings(embedding_key)
    embedding_matrix, q1_tensor, q2_tensor = create_embedding_matrix(
        df, model, max_length
    )
    return embedding_matrix, q1_tensor, q2_tensor
