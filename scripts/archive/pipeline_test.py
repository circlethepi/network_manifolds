############################################                   
#              Import Packages             #
###############################################################################        
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True

import wandb
import pickle as pickle
from tqdm import tqdm
import argparse
import time

# huggingface
import datasets
from transformers import AutoTokenizer
from transformers import pipeline
from accelerate import Accelerator

# custom imports
from src.utils import *
import src.data as data
import src.model_analysis as analysis

from safetensors.torch import save_file

device = get_device()

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"

llama_pipe = pipeline(model=base_model_id)

# data
yahoo = datasets.load_dataset("yahoo_answers_topics", split="test")
sampled = data.sample_datasets_uniform(100, 0, yahoo)
sampled = sampled.map(lambda x: data.concat_yahoo(x, concat_method="question"))

# testing the pipeline
outputs = llama_pipe(
    sampled["example"][:3],
    num_return_sequences=3,
    return_full_text=False
)

# save for examination later
savepath = os.path.join(GLOBAL_PROJECT_DIR, "results", "llama_test.pkl")

with open(savepath, "wb") as file:
    pickle.dump(outputs, file)



