import torch
import numpy as np
import random
import os

"""
Utility functions
"""

def set_seed(seed):
    """set the random seed everywhere relevant"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if \
                          torch.backends.mps.is_available() else "cpu")
    return device

# logging/writing things
def make_logfile(filepath):
    """opens a file (that will need to be closed)"""
    logfile = open(filepath, 'a')
    return logfile


def print_and_write(message, *files):
    """files should be open"""
    print(message)
    for file in files:
        if file is not None:
            file.write(message+'\n')
    return

def close_files(*files):
    """closes logs"""
    for file in files:
        if file is not None:
            file.close()
    return

def nice_interval(n: int):
    if n == 0:
        return True
    else:
        pwr = 10 ** int(np.log10(n))
        nice = True if (n == 0 or (n % pwr == 0 and n // pwr in (1, 2, 5))) else False
        return nice


class AverageMeter(object):
    def __init__(self, name=None, format=':.2f'):
        self.reset()
        self.name = name
        self.format = format
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.format + '} ({avg' + self.format + '})'
        return fmtstr.format(**self.__dict__)


def check_if_null(named_param_or_var, alternate):
    """ checks if an object exists and returns the given alternate if not
    """
    return named_param_or_var if named_param_or_var is not None else alternate

