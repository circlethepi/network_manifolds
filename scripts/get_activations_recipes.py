"""
Script to get activations from a model for specific recipes of inputs

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

count = 5
name = f"recipe_activations{count}"

description =   """
                various recipes with various numbers of replicates and queries
                """

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
peft_model_id = "../results/test_finetune_llama_yahoo9_4/checkpoint-77885"

# savefile = f'../results/{name}.pkl'

# specify which parameters to get activations from
# layer_list = [
#     f"model.model.layers[0].self_attn.{p}_proj.lora_{M}.default" for p in "q" \
#                                                                  for M in "AB"
# ]

# recipe sampling parameters
recipes = [ # 
    [0.5, 0.5],  # 50% from topic 8, 50% from topic 9
    [0.2, 0.8],  # 20% from topic 8, 80% from topic 9
    [0.8, 0.2],  # 80% from topic 8, 20% from topic 9
    [1.0, 0.0],  # all from topic 8
    [0.0, 1.0],  # all from topic 9
]
n_replicates = 100
n_queries = 100

seed = 0


                ############################################                   
                #          Actually Doing Things           #
###############################################################################

## make the analysis object
tuned = analysis.LoraAnalysis(base_model_path=base_model_id, 
                              peft_path=peft_model_id)

# get the layers
layer_list = list(tuned.get_layer_names(rules=["B"], attributes=True).values())

# load in the dataset
yahoo_test = datasets.load_dataset("yahoo_answers_topics", split="train")
# - filter for topics [8, 9] only
split_test = data.topic_split_yahoo(yahoo_test, topics=[8, 9])[0]
print(split_test)

for rec in recipes:
    recipe_name = f"t8={rec[0]}_t9={rec[1]}"
    ## make the query datasets

    # - sample according to recipe
    sampled_data = data.sample_from_recipe(rec, n_queries,
                                            *split_test.values(),
                                            seed=seed)

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
                                    input_name=input_name)

    # change device to cpu
    dev = torch.device('cpu')
    outputs = analysis.move_output_tensors(outputs, dev)

    # # description = f""


    # # package for saving
    # outputs['description'] = description
    # # print(description)

    # with open(savefile, 'wb') as file:
    #     pickle.dump(outputs, file)

print("\n\nFinished")