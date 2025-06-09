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

count = 2
name = f"recipe_activations{count}"

description = \
            "test getting activations for specific query recipes" \
            "and saving to file"

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
peft_model_id = "../results/test_finetune_llama_yahoo9_4/checkpoint-77885"

savefile = f'../results/{name}.pkl'

# specify which parameters to get activations from
layer_list = [
    f"model.model.layers[0].self_attn.{p}_proj" for p in "q"
]

# recipe sampling parameters
recipes = [ # 
    [0.5, 0.5],  # 50% from topic 8, 50% from topic 9
    # [0.2, 0.8],  # 20% from topic 8, 80% from topic 9
    # [0.8, 0.2],  # 80% from topic 8, 20% from topic 9
    # [1.0, 0.0],  # all from topic 8
    # [0.0, 1.0],  # all from topic 9
]
n_replicates = 10
n_queries = 10


                ############################################                   
                #          Actually Doing Things           #
###############################################################################

## make the analysis object
tuned = analysis.LoraAnalysis(base_model_path=base_model_id, 
                              peft_path=peft_model_id)

## make the query datasets
yahoo_test = datasets.load_dataset("yahoo_answers_topics", split="test")

# - filter for topics [8, 9] only
split_test = data.topic_split_yahoo(yahoo_test, topics=[8, 9])[0]
print(split_test)

# - sample according to recipe
sampled_data = data.sample_from_recipe(recipes[0], n_queries,
                                           *split_test.values())

# - concatenate answers and tokenize
# -- make tokenizing function
# def tokenize_elements(element):
#     return tuned.tokenizer(element['example'], padding="max_length", 
#                            truncation=True, max_length=1024, 
#                            return_tensors="pt")

# -- process the data into the inputs
sampled_data = sampled_data.map(data.concat_yahoo)\
                        #    .map(tokenize_elements, batched=True, num_proc=4,
                        #         remove_columns=sampled_data.column_names)
inputs = tuned.tokenizer(sampled_data['example'], 
                         padding="max_length", truncation=True, 
                         max_length=256, return_tensors="pt")

## Do inference with activations
outputs = tuned.inference_with_activations(inputs, layer_list,
                                 num_return_sequences=n_replicates,
                                 max_length=512,
                                 states=False,
                                 attention=False)

# change device to cpu
dev = torch.device('cpu')
outputs = analysis.move_output_tensors(outputs, dev)


# package for saving
outputs['description'] = description

with open(savefile, 'wb') as file:
    pickle.dump(outputs, file)

print("\n\nFinished")