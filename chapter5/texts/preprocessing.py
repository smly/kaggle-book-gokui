import re
from enum import Enum
from multiprocessing import Pool

import nltk
import pandas as pd
from gensim.models import KeyedVectors
from nltk import SnowballStemmer, word_tokenize
from tqdm import tqdm

from common.constants import (
    GLOVE_PATH,
    INPUT_DIR,
    NUM_PROCESSES,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
)

NLTK_STEMMER = SnowballStemmer("english")
NLTK_STOPWORDS = set(nltk.corpus.stopwords.words("english"))
NLTK_STEMMED_STOPWORDS = set(NLTK_STEMMER.stem(w) for w in NLTK_STOPWORDS)


class PreprocessingKey(Enum):
    SPACE_TOKENIZATION = 0
    NLTK_TOKENIZATION = 1
    NLTK_STEMMING = 2


def space_tokenize(s):
    return re.sub(r"\s+", " ", s).strip()


def nltk_tokenize(s):
    return " ".join(word_tokenize(space_tokenize(s)))


def nltk_stemming(s):
    return " ".join(map(NLTK_STEMMER.stem, word_tokenize(space_tokenize(s))))


def tokenize_df(df, func, processes=1, chunksize=1000):
    new_df = df.copy()
    # トークン化は結構時間がかかる処理なので並列化
    with Pool(processes=processes) as pool:
        new_df["question1"] = list(
            tqdm(
                pool.imap(func, df["question1"], chunksize=chunksize),
                total=len(df),
                desc="Tokenizing",
            )
        )
        new_df["question2"] = list(
            tqdm(
                pool.imap(func, df["question2"], chunksize=chunksize),
                total=len(df),
                desc="Tokenizing",
            )
        )
    return new_df


def get_dataset(preprocessing_key):
    dataset_path = INPUT_DIR / f"{preprocessing_key.name.lower()}.csv"
    if dataset_path.exists():
        return pd.read_csv(dataset_path, na_filter=False)

    trn_df = pd.read_csv(TRAIN_CSV_PATH, na_filter=False)
    tst_df = pd.read_csv(TEST_CSV_PATH, na_filter=False)
    tst_df["is_duplicate"] = 0
    columns = ["question1", "question2", "is_duplicate"]
    df = pd.concat((trn_df, tst_df), axis=0).reset_index(drop=True)[columns]

    func = {
        PreprocessingKey.SPACE_TOKENIZATION: space_tokenize,
        PreprocessingKey.NLTK_TOKENIZATION: nltk_tokenize,
        PreprocessingKey.NLTK_STEMMING: nltk_stemming,
    }[preprocessing_key]

    df = tokenize_df(df, func, processes=NUM_PROCESSES)
    df.to_csv(dataset_path, index=False)
    return df


class StopwordsKey(Enum):
    NONE = 0
    NLTK = 1
    NLTK_STEMMED = 2


def get_stopwords(stopwords_key):
    return {
        StopwordsKey.NONE: set(),
        StopwordsKey.NLTK: set(nltk.corpus.stopwords.words("english")),
        StopwordsKey.NLTK_STEMMED: set(
            map(NLTK_STEMMER.stem, nltk.corpus.stopwords.words("english"))
        ),
    }[stopwords_key]


def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if token.lower() not in stopwords]


class EmbeddingKey(Enum):
    GLOVE = 0


def get_embeddings(embedding_key):
    embedding_path = {
        EmbeddingKey.GLOVE: GLOVE_PATH,
    }[embedding_key]
    return KeyedVectors.load_word2vec_format(embedding_path, binary=True)
