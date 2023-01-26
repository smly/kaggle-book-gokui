from enum import Enum

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline

from common.constants import FEATURE_MEMORY
from texts.preprocessing import get_dataset, get_stopwords


class VectorizerType(Enum):
    COUNT = 0
    TFIDF_L2 = 1
    TFIDF_NONE = 2


class DecompositionType(Enum):
    SVD = 0
    NMF = 1


def get_vectorizer(vectorizer_type, ngram_range, stopwords):
    return {
        VectorizerType.COUNT: CountVectorizer(
            tokenizer=lambda x: x.split(),
            stop_words=stopwords,
            ngram_range=ngram_range,
            min_df=5,
            binary=True,
        ),
        VectorizerType.TFIDF_L2: TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            stop_words=stopwords,
            ngram_range=ngram_range,
            min_df=5,
            binary=True,
        ),
        VectorizerType.TFIDF_NONE: TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            stop_words=stopwords,
            ngram_range=ngram_range,
            min_df=5,
            norm=None,
            binary=True,
        ),
    }[vectorizer_type]


def get_decomposition(decomposition_type, n_components):
    return {
        DecompositionType.SVD: TruncatedSVD(n_components, random_state=1),
        DecompositionType.NMF: NMF(n_components, random_state=1),
    }[decomposition_type]


def decomposition_features(df, vectorizer, decomposition, suffix):
    all_questions = df["question1"].to_list() + df["question2"].to_list()
    pipeline = make_pipeline(vectorizer, decomposition)
    vectors = pipeline.fit_transform(all_questions).astype(np.float32)
    q1_vectors = vectors[: len(df)]
    q2_vectors = vectors[len(df) :]

    features = pd.DataFrame()
    n_components = q1_vectors.shape[1]
    for i in range(n_components):
        features[f"decomp_q1_{i}_{suffix}"] = q1_vectors[:, i]
        features[f"decomp_q2_{i}_{suffix}"] = q2_vectors[:, i]
    return features


@FEATURE_MEMORY.cache
def build_decomposition_features(
    dataset_key,
    stopwords_key,
    vectorizer_type,
    decomposition_type,
    n_components,
    ngram_range,
):
    df = get_dataset(dataset_key)
    stopwords = get_stopwords(stopwords_key)

    suffix = (
        f"{dataset_key.name}_{stopwords_key.name}_{vectorizer_type.name}_{decomposition_type.name}"
        f"_{n_components}_{ngram_range[0]}-{ngram_range[1]}"
    )
    return decomposition_features(
        df,
        get_vectorizer(vectorizer_type, ngram_range, stopwords),
        get_decomposition(decomposition_type, n_components),
        suffix,
    )
