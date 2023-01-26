import os
from pathlib import Path

import joblib

DATA_DIR = Path(os.getenv("QQP_DATA_DIR", "/data"))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TRAIN_CSV_PATH = INPUT_DIR / "train.csv"
TEST_CSV_PATH = INPUT_DIR / "test.csv"
EMBEDDING_DIR = DATA_DIR / "embeddings"
GLOVE_PATH = EMBEDDING_DIR / "glove.840B.300d.bin"

FEATURE_MEMORY = joblib.Memory(DATA_DIR / "cache")

SPLIT_RANDOM_SEED = 1
EPS = 1e-10
NUM_PROCESSES = int(os.getenv("NUM_PROCESSES", 1))
NUM_TRAIN_SAMPLES = 404290
NUM_TEST_SAMPLES = 2345796
NUM_DRYRUN_SAMPLES = 1000
