import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from common.constants import EPS, FEATURE_MEMORY
from texts.preprocessing import get_dataset, get_stopwords


def jaccard(q1_vectors, q2_vectors):
    num = q1_vectors.minimum(q2_vectors).sum(axis=1)
    den = q1_vectors.maximum(q2_vectors).sum(axis=1)
    return num / (den + EPS)  # ゼロ除算を避けるために分母に小さい数EPS (1e-10) を足している


def dice(q1_vectors, q2_vectors):
    num = 2 * q1_vectors.minimum(q2_vectors).sum(axis=1)
    den = q1_vectors.sum(axis=1) + q2_vectors.sum(axis=1)
    return num / (den + EPS)  # ゼロ除算を避けるために分母に小さい数EPS (1e-10) を足している


def cosine(q1_vectors, q2_vectors):
    q1_vectors = normalize(q1_vectors)
    q2_vectors = normalize(q2_vectors)
    return q1_vectors.multiply(q2_vectors).sum(axis=1)


def match_features(df, count_vectorizer, tfidf_transformer, suffix):
    qs_counts = count_vectorizer.fit_transform(
        df["question1"].to_list() + df["question2"].to_list()
    )
    qs_tfidfs = tfidf_transformer.fit_transform(qs_counts)
    q1_counts, q2_counts = qs_counts[: len(df)], qs_counts[len(df) :]
    q1_tfidfs, q2_tfidfs = qs_tfidfs[: len(df)], qs_tfidfs[len(df) :]

    features = pd.DataFrame(index=df.index)
    features[f"jaccard_count_{suffix}"] = jaccard(q1_counts, q2_counts)
    features[f"dice_count_{suffix}"] = dice(q1_counts, q2_counts)
    features[f"cosine_count_{suffix}"] = cosine(q1_counts, q2_counts)
    features[f"jaccard_tfidf_{suffix}"] = jaccard(q1_tfidfs, q2_tfidfs)
    features[f"dice_tfidf_{suffix}"] = dice(q1_tfidfs, q2_tfidfs)
    features[f"cosine_tfidf_{suffix}"] = cosine(q1_tfidfs, q2_tfidfs)
    return features


@FEATURE_MEMORY.cache  # FEATURE_MEMORY = joblib.Memory(DATA_DIR / "cache")
def build_match_features(preprocessing_key, stopwords_key, n=1):
    df = get_dataset(preprocessing_key)
    stopwords = get_stopwords(stopwords_key)

    count_vectorizer = CountVectorizer(
        tokenizer=lambda s: s.split(),
        stop_words=stopwords,
        ngram_range=(n, n),
        binary=True,
    )
    tfidf_transformer = TfidfTransformer(norm=None)

    suffix = f"{preprocessing_key.name}_{stopwords_key.name}_{n}gram"
    return match_features(df, count_vectorizer, tfidf_transformer, suffix)
