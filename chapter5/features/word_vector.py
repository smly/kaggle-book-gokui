import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

from common.constants import FEATURE_MEMORY
from texts.preprocessing import (
    get_dataset,
    get_embeddings,
    get_stopwords,
    remove_stopwords,
)


def farthest_word_distance_features(
    df, model, stopwords, metric, normalize, suffix
):
    distance_q1 = np.full(len(df), np.inf)
    distance_q2 = np.full(len(df), np.inf)
    desc = f"Computing farthest word distance ({metric})"
    for i, (q1, q2) in enumerate(
        tqdm(zip(df.question1, df.question2), total=len(df), desc=desc)
    ):
        q1_words = set(remove_stopwords(q1.split(), stopwords))
        q1_vecs = np.array(
            [
                model.get_vector(word, norm=normalize)
                for word in q1_words
                if word in model
            ]
        )
        q2_words = set(remove_stopwords(q2.split(), stopwords))
        q2_vecs = np.array(
            [
                model.get_vector(word, norm=normalize)
                for word in q2_words
                if word in model
            ]
        )
        if len(q1_vecs) > 0 and len(q2_vecs) > 0:
            distance_matrix = cdist(q1_vecs, q2_vecs, metric)
            distance_q1[i] = distance_matrix.min(axis=1).max()
            distance_q2[i] = distance_matrix.min(axis=0).max()
    features = pd.DataFrame(index=df.index)
    features[f"fwd_min_{suffix}"] = np.minimum(distance_q1, distance_q2)
    features[f"fwd_max_{suffix}"] = np.maximum(distance_q1, distance_q2)
    return features


@FEATURE_MEMORY.cache
def build_farthest_word_distance_features(
    dataset_key, stopwords_key, embedding_key, metric, normalize
):
    df = get_dataset(dataset_key)
    stopwords = get_stopwords(stopwords_key)
    model = get_embeddings(embedding_key)

    suffix = f"{dataset_key.name}_{stopwords_key.name}_{embedding_key.name}_{metric}_{normalize}"
    return farthest_word_distance_features(
        df, model, stopwords, metric, normalize, suffix
    )


def wmd_features(df, model, stopwords, normalize):
    values = []
    for _, (q1, q2) in tqdm(
        enumerate(zip(df.question1, df.question2)),
        total=len(df),
        desc=f"Computing WMD",
    ):
        q1_tokens = list(set(remove_stopwords(q1.split(), stopwords)))
        q2_tokens = list(set(remove_stopwords(q2.split(), stopwords)))
        values.append(model.wmdistance(q1_tokens, q2_tokens, norm=normalize))
    return values


@FEATURE_MEMORY.cache
def build_wmd_features(dataset_key, stopwords_key, embedding_key, normalize):
    df = get_dataset(dataset_key)
    stopwords = get_stopwords(stopwords_key)
    model = get_embeddings(embedding_key)

    suffix = f"{dataset_key.name}_{stopwords_key.name}_{embedding_key.name}_{normalize}"
    features = pd.DataFrame(index=df.index)
    features[f"wmd_{suffix}"] = wmd_features(df, model, stopwords, normalize)
    return features
