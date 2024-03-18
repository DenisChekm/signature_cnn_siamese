import torch

import numpy as np

from math import sqrt
import os
import random


class Config:
    SEED = 42
    CANVAS_SIZE = (952, 1360)
    EARLY_STOPPING_EPOCH = 10  # 7
    PRINT_FREQ = 66  # для balanced dataset
    # 54 для no_balanced dataset
    THRESHOLD = 0.5

    # Trash:
    projection2d = True
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1000
    MODEL_NAME = "siamnet"

    @staticmethod
    def divisor_generator(n):
        large_divisors = []
        for i in range(1, int(sqrt(n) + 1)):
            if n % i == 0:
                yield i
                if i * i != n:
                    large_divisors.append(n / i)
        for divisor in reversed(large_divisors):
            yield divisor

    @staticmethod
    def seed_torch():
        seed = Config.SEED
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
