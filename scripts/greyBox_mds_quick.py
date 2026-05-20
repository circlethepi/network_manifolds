import transformers
import datasets

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import textwrap
import os

import wandb

from safetensors import safe_open

# custom imports
import src.model_analysis as analysis
import src.data as data
import src.dkps as dkps
from src.utils import *

base_model = "meta-llama/Llama-3.2-1B-Instruct"

# get the result files
exp_name = [f"{c}_{k}" for c in ("jedi", "wizard") for k in range(8)]
base_name = base_model
reps = 5

result_files = dkps.find_result_files(exp_name, base_name, 
                                      custom_name="grey_box_0", 
                                      narrate=True)

for q in (5, 10, 20, 50, 100):
    results = dkps.results_from_data_ids(result_files, queries=q,
                                         replicates=reps)
    name = f"greySpace_r{reps}_q{q}"
    coords = dkps.induce_dkps(results, name, get_coordinates=True)

    print(f"Found coordinates for q={q}")
    print(LOG_SEPARATOR)