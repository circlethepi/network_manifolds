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


def move_output_tensors(output_dict:dict, device:torch.device):
    """moves output tensors from a model to a different device
    
    TODO add description
    """

    possible_keys = ["sequences", 
                     "attentions", 
                     "hidden_states", 
                     "past_key_values"]
    # get the applicable keys
    keys_to_use = list(set(output_dict.keys()) & set(possible_keys))
    
    new_dict = {}

    for key in keys_to_use:
        value = output_dict[key]

        if key == "sequences":
            new_dict[key] = value.to(device)
        
        else:
            # `max_seq_len` entries, each a tuple with `n_layers` tensors
            new_value = []
            for entry in value:
                new_value.append([x.to(device) for x in entry])
            new_dict[key] = tuple(new_value)

    return new_dict

# def move_devices(obj, device:torch.device):
    
#     if isinstance(obj, torch.Tensor):
#         return obj.to(device)
#     elif isinstance(obj, dict):
#         return {k: move_devices(v, device) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [move_devices(x, device) for x in obj]
#     elif isinstance(obj, tuple):
#         return tuple(move_devices(x, device) for x in obj)
#     else:
#         print(f'object is {type(obj)}\nreturning object...')
#         return obj

    
    







    return