import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def simple_timer(message, logger=None):
    start_time = time.time()
    yield
    timer_message = f"{message}: {time.time() - start_time:.3f} [s]"
    if logger is None:
        print(timer_message)
    else:
        logger.info(timer_message)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_weights(y, target_positive_ratio):
    positive_ratio = y.mean()
    positive_weight = target_positive_ratio / positive_ratio
    negative_weight = (1 - target_positive_ratio) / (1 - positive_ratio)
    weights = np.full(len(y), negative_weight)
    weights[y == 1] = positive_weight
    return weights
