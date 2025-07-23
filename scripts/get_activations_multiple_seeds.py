"""
Script to get activations from a model for specific recipes of inputs


Run 2025-07-18 
- seeds 1-10
- Responses saved on DSAI machine
"""

############################################                   
#              Import Packages             #
###############################################################################                  

import torch
import wandb
import os
import pickle as pickle
from tqdm import tqdm

# huggingface import
import datasets
from transformers import AutoTokenizer

# custom imports
from src.utils import *
import src.data as data
import src.model_analysis as analysis
from safetensors.torch import save_file


        ############################################                   
        #              Set Parameters              #
###############################################################################

count = 1  # ID
name = f"multi_recipe_activations{count}"

description =   """
                various recipes with various numbers of replicates and queries
                """

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
peft_model_id = "../results/test_finetune_llama_yahoo9_4/checkpoint-77885"

# recipe sampling parameters
recipes = [ # 
    [0.5, 0.5],  # 50% from topic 8, 50% from topic 9
    [0.2, 0.8],  # 20% from topic 8, 80% from topic 9
    [0.8, 0.2],  # 80% from topic 8, 20% from topic 9
    [1.0, 0.0],  # all from topic 8
    [0.0, 1.0],  # all from topic 9
]
n_replicates = 10
n_queries = 40

# seeds = list(range(1, 11))
# print(seeds)
# 0 already stored!
seed_start = 0  
n_runs = 10 # number of runs to do for each recipe

print(f"Running inference with {n_queries} with {n_replicates} each on {n_runs} seeds")


                ############################################                   
                #          Actually Doing Things           #
###############################################################################

## make the analysis object
tuned = analysis.LoraAnalysis(base_model_path=base_model_id, 
                              peft_path=peft_model_id)

print("Model loading complete. Collecting layer names...")

# get the layers
layer_list = list(tuned.get_layer_names(rules=["B", "_o-proj"], attributes=True, 
                                        weights=False).values())
# print(layer_list)

print("Layer names collected. Creating data split...")

# load in the dataset
yahoo_test = datasets.load_dataset("yahoo_answers_topics", split="train")
# - filter for topics [8, 9] only
split_test = data.topic_split_yahoo(yahoo_test, topics=[8, 9])[0]
print(split_test)

seed = seed_start  # start with the first seed
for rec in tqdm(recipes):
    recipe_name = f"t8={rec[0]}_t9={rec[1]}"
    ## make the query datasets
    for k in range(n_runs):
    # - sample according to recipe
        print(seed)
        sampled_data = data.sample_from_recipe(rec, n_queries, seed,
                                                *split_test.values())

        # -- process the data into the inputs
        sampled_data = sampled_data.map(data.concat_yahoo)\

        inputs = tuned.tokenizer(sampled_data['example'], 
                                padding="max_length", truncation=True, 
                                max_length=256, return_tensors="pt")

        input_name = f"r{n_replicates}_q{n_queries}_yahoo_{recipe_name}_{count}_seed{seed}"
        ## Do inference with activations
        outputs = tuned.inference_with_activations(inputs, layer_list,
                                        num_return_sequences=n_replicates,
                                        max_length=512,
                                        states=False,
                                        attention=False,
                                        input_name=input_name,
                                        return_outputs=True,
                                        output_device=torch.device('cpu'))

        seed += 1

print("\n\nFinished all seeds")