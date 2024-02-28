from math import floor
from time import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.list.append(val * n)


def as_minutes(s):
    m = floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(start, percent):
    now = time()
    since = now - start
    es = since / percent
    remain = es - since
    return '%s (remain %s)' % (as_minutes(since), as_minutes(remain))
