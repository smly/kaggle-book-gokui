import functools
from multiprocessing import Pool

import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

from common.constants import FEATURE_MEMORY, NUM_PROCESSES
from texts.preprocessing import get_dataset, get_stopwords, remove_stopwords


def compute_edit_distance(pair, method, stopwords):
    q1 = " ".join(remove_stopwords(pair[0].split(), stopwords))
    q2 = " ".join(remove_stopwords(pair[1].split(), stopwords))
    return method(q1, q2)


def edit_distance_features(df, stopwords, suffix):
    methods = [
        fuzz.WRatio,
        fuzz.QRatio,
        fuzz.partial_ratio,
        fuzz.partial_token_set_ratio,
        fuzz.partial_token_sort_ratio,
        fuzz.token_set_ratio,
        fuzz.token_sort_ratio,
    ]
    features = pd.DataFrame(index=df.index)
    question_pairs = list(zip(df.question1, df.question2))
    with Pool(processes=NUM_PROCESSES) as pool:
        for method in methods:
            distance_name = method.__name__.lower()
            desc = f"Computing edit distance '{distance_name}' ({suffix})"
            func = functools.partial(
                compute_edit_distance, method=method, stopwords=stopwords
            )
            distances = list(
                tqdm(
                    pool.imap(func, question_pairs, chunksize=100),
                    desc=desc,
                    total=len(df),
                )
            )
            features[f"{distance_name}_{suffix}"] = distances
    return features


@FEATURE_MEMORY.cache
def build_edit_distance_features(dataset_key, stopwords_key):
    df = get_dataset(dataset_key)
    stopwords = get_stopwords(stopwords_key)

    suffix = f"{dataset_key.name}_{stopwords_key.name}"
    return edit_distance_features(df, stopwords, suffix)
