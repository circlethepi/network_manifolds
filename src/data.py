#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#               Processing Datasets for Causal Language Models   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
  description of the file here
"""

import torch
import numpy as np
import random
import datasets

from src.utils import display_message


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                          Generic Processing Functions           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

    :return: small_train, small_eval, small_test datasets
    """

    print(f"New train set size: {size}" \
    f"\nNew eval  set size: {int(size*prop_val)}" \
    f"\nNew test  set size: {int(size*prop_test)}")

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#            Processing for Yahoo Dataset        #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def concat_yahoo(example):
    """
    Concatenates the question title, content, and best answer into a 
    single string.
    """
    return {"example": example['question_title'] + " " + 
            example['question_content'] + " " + example['best_answer']}


def topic_split_yahoo(*yahoo_datasets, topics="all"):
    """
    Splits the Yahoo dataset into separate datasets for each topic.
    
    :param yahoo_datasets: dataset(s) containing Yahoo questions
    :param topics: A list of topics to filter by, or "all"
                   If a list is provided, only those topics will be included.
                   List should contain integers representing topic IDs.
                   Default is "all".

    :return out: tuple:  for each input yahoo dataset, dictionary where keys 
                         are topics and values are datasets.
    """

    out = []

    for yahoo_dataset in yahoo_datasets:
        topic_datasets = {}

        if topics == "all":
            # If "all" is specified, use all topics in the dataset
            topics = yahoo_dataset.unique('topic')
            topics.sort()  
        else: # otherwise, check that the topics are valid
            assert isinstance(topics, list), \
                "Topics must be a list of integers or 'all'."
            assert all(isinstance(topic, int) for topic in topics) 

        for topic in topics:
            topic_datasets[topic] = yahoo_dataset.filter(lambda e: e['topic'] == topic)
        
        out.append(topic_datasets)
    
    return tuple(out)


                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                #        Joys of Baking: Dataset Recipes         #
#~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~#

def create_hull_grid(n:int, incs:int=1):
    # WISHLIST: create this function
    """
    Creates a "grid" of points in the convex hull of the standard basis vectors
    in R^n, with a specified increment.

    :param n: dimension of the grid (number of basis vectors)
    :param incs: increments for the grid points (default: 1)
    :return: list of points in the grid

    The increment is used to define the spacing between points in the grid and 
    corresponds to the number of nonzero steps in each dimension. For example, 
    if `incs=2`, then each dimension will have possible values of {0, 0.5, 1}
    """


    return


def get_sample_counts(recipe, n_samples):
    """
    Get the number of samples for each class based on the recipe.

    :param recipe: List of probabilities for each class/dataset
    :param n_samples: Total number of samples to draw.
    :return count_dict: [class_idx] -> number of samples
    """
    #TODO: add option to use a seed
    sample = np.random.choice(len(recipe), size=n_samples, p=recipe)
    unique, counts = np.unique(sample, return_counts=True)

    count_dict = {int(k): int(v) for k, v in zip(unique, counts)} 
            # [dataset_idx] -> number of samples

    return count_dict


def sample_from_datasets(count_dict:dict, *dataset_list, seed:int=0):
    """
    Sample from the datasets based on the count_dict

    :param count_dict: [dataset_idx] -> number of samples
    :param datasets: huggingface datasets to sample from
    :param seed: random seed for reproducibility

    :return sampled_datasets: datasets.arrow_dataset.Dataset
            concatenated dataset sampled from input datasets
    """

    assert len(dataset_list) >= max(count_dict.keys()) + 1, \
        "Not enough datasets provided in the input."
        # make sure there are enough datasets
    
    sampled_datasets = []
    for dataset_idx, count in count_dict.items():
        dataset = dataset_list[dataset_idx]
        sampled_dataset = dataset.shuffle(seed=seed).select(range(count))
        sampled_datasets.append(sampled_dataset)
    
    # concatenate the datasets
    sampled_datasets = datasets.concatenate_datasets(sampled_datasets)

    return sampled_datasets



def sample_from_recipe(recipe, n, seed:int=0, *dataset_list):
    """
    Samples 1 instance of n samples from a given recipe/sampling vector 

    :param recipe: sampling vector
    :param n: number of samples in the resulting dataset
    :param dataset_list: datasets to sample from

    :return: sampled_dataset: datasets.arrow_dataset.Dataset
    The dimension of the recipe should match the number of datasets provided.
    """

    # get the counts
    count_dict = get_sample_counts(recipe, n)
    # sample from the datasets
    sampled_datasets = sample_from_datasets(count_dict, 
                                            *dataset_list, 
                                            seed=seed)

    return sampled_datasets


def sample_from_yahoo_recipe(recipe, n:int, split="test"):
    """
    samples 1 instance of n samples from the Yahoo Answers Dataset from a given
    recipe/sampling vector

    :param recipe: sampling vector. Should be 10 entries, one for each of the 
                                    topics in the Yahoo Answers Dataset
    :param n: number of samples in the resulting dataset
    :param split: the data split to use

    :return sampled_dataset: datasets.arrow_dataset.Dataset
    """

    # check that the split is valid
    if split not in ("test", "train"):
        message = """
        split must be either 'test' or 'train' for the Yahoo Answers Dataset.
        """
        raise ValueError(display_message(message))
    if not len(recipe) == 10:
        message = """
        Recipe must have 10 entries for the Yahoo Answers Dataset
        """
        raise ValueError(display_message(message)) 

    # get the split yahoo dataset
    yahoo_set = datasets.load_dataset("yahoo_answers_topics", split=split)
    yahoo_split = topic_split_yahoo(yahoo_set, topics="all")[0]
    
    # get the sampled data
    sampled = sample_from_recipe(recipe, n, *yahoo_split.values())

    return sampled
