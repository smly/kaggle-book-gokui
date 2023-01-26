import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from common.constants import FEATURE_MEMORY
from texts.preprocessing import get_dataset, get_stopwords


def length_features(df, vectorizer, transformer, suffix):
    qs_counts = vectorizer.fit_transform(
        df["question1"].to_list() + df["question2"].to_list()
    )
    qs_tfidfs = transformer.fit_transform(qs_counts)
    q1_counts, q2_counts = qs_counts[: len(df)], qs_counts[len(df) :]
    q1_tfidfs, q2_tfidfs = qs_tfidfs[: len(df)], qs_tfidfs[len(df) :]
    features = pd.DataFrame(index=df.index)

    vector_pairs = [
        ("tfidf", q1_tfidfs, q2_tfidfs),
        ("count", q1_counts, q2_counts),
    ]
    for name, q1_vectors, q2_vectors in vector_pairs:
        features[f"length_min_{name}_{suffix}"] = np.minimum(
            q1_vectors.sum(axis=1), q2_vectors.sum(axis=1)
        ).A1
        features[f"length_max_{name}_{suffix}"] = np.maximum(
            q1_vectors.sum(axis=1), q2_vectors.sum(axis=1)
        ).A1
        features[f"length_abs_diff_{name}_{suffix}"] = (
            features[f"length_max_{name}_{suffix}"]
            - features[f"length_min_{name}_{suffix}"]
        )
        features[f"length_rel_diff_{name}_{suffix}"] = (
            features[f"length_abs_diff_{name}_{suffix}"]
            / features[f"length_max_{name}_{suffix}"]
        )
    return features


@FEATURE_MEMORY.cache
def build_length_features(dataset_key, stopwords_key, n=1):
    df = get_dataset(dataset_key)
    stopwords = get_stopwords(stopwords_key)

    vectorizer = CountVectorizer(
        tokenizer=lambda s: s.split(),
        stop_words=stopwords,
        ngram_range=(n, n),
        binary=True,
    )
    transformer = TfidfTransformer(norm=None)

    suffix = f"{dataset_key.name}_{stopwords_key.name}_{n}gram"
    return length_features(df, vectorizer, transformer, suffix)


@FEATURE_MEMORY.cache
def build_normalized_length_features(preprocessing_key, stopwords_key, n=1):
    df = get_dataset(preprocessing_key)
    stopwords = get_stopwords(stopwords_key)

    vectorizer = CountVectorizer(
        tokenizer=lambda s: s.split(),
        stop_words=stopwords,
        ngram_range=(n, n),
        binary=True,
    )
    transformer = TfidfTransformer()

    suffix = f"{preprocessing_key.name}_{stopwords_key.name}_{n}gram"
    return length_features(df, vectorizer, transformer, suffix)
