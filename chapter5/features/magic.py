from collections import defaultdict

import pandas as pd

from common.constants import FEATURE_MEMORY
from texts.preprocessing import PreprocessingKey, get_dataset


def magic_features(df):
    question_count = (
        pd.Series(df.question1.to_list() + df.question2.to_list())
        .value_counts()
        .to_dict()
    )
    features = pd.DataFrame(index=df.index)
    features["q1_freq"] = df.question1.map(question_count)
    features["q2_freq"] = df.question2.map(question_count)
    adj = defaultdict(set)
    for q1, q2 in zip(df.question1, df.question2):
        adj[q1].add(q2)
        adj[q2].add(q1)
    features["q1q2_inter"] = [
        len(adj[q1].intersection(adj[q2]))
        for q1, q2 in zip(df.question1, df.question2)
    ]
    return features


@FEATURE_MEMORY.cache
def build_magic_features():
    df = get_dataset(PreprocessingKey.SPACE_TOKENIZATION)
    return magic_features(df)
