import torch
import numpy as np
import random
import datasets

"""
Processing Datasets
"""

def make_small_train_val_test(train_data, test_data, size:int=500, 
                              prop_val:float=0.2, prop_test:float=0.2, 
                              map=None,
                              seed:int=0):
    """ Makes smaller datasets from full ones
    :param size: number of samples for the training dataset
    :param prop_val: size of the validation set with respect to the size of the 
                     training dataset
    :param prop_test: size of the test set with respect to the size of the 
                      training dataset
    :param map: (optional) mapping function to apply to data
    :param seed: random seed for selecting the subset of the data (default: 0)
    """

    print(f"Creating train set of {size} samples\nCreating eval  set of {int(size*prop_val)} samples\nCreating test  set of {int(size*prop_test)} samples")

    small_train = train_data.shuffle(seed=seed).select(range(size))
    small_eval = train_data.shuffle(seed=seed).select(range(size,size+int(size*prop_val)))
    small_test = test_data.shuffle(seed=seed).select(range(int(size*prop_test)))

    if map is not None:
        small_train = small_train.map(map)
        small_eval = small_eval.map(map)
        small_test = small_test.map(map)

    return small_train, small_eval, small_test



def group_examples(examples, block_size = 128):
    """pre-processing function for causal LM
    (taken from docs on HuggingFace)
    TODO add link from docs
    
    """

    # concatenate all
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # split by chunks
    result = {}
    for k, t in concatenated.items():
        blocks = []
        for i in range(0, total_length, block_size):
            blocks.append(t[i : i + block_size])

        result[k] = blocks
    
    result['labels'] = result['input_ids'].copy()
    
    return result