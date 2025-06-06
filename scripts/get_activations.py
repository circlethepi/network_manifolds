"""
script to get activations (hopefully) and responses from an LLM

"""
############################################                   
#              Import Packages             #
###############################################################################                  

import torch
import wandb
import os
import pickle as pickle

# custom imports
from src.utils import *
import src.model_analysis as analysis
from safetensors.torch import save_file


        ############################################                   
        #              Set Parameters              #
###############################################################################

count = "5c"

description = \
            "testing with multiple outputs for each input and batch size 2"


base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
peft_model_id = "../results/test_finetune_llama_yahoo9_4/checkpoint-77885"

savefile = f'../results/test_inference{count}.pkl'

                ############################################                   
                #          Actually Doing Things           #
###############################################################################

# make the analysis object
tuned = analysis.LoraAnalysis(base_model_path=base_model_id, 
                              peft_path=peft_model_id)

input_strings = [
    "preheat the oven to 350 degrees and place the cookie dough",
    "what is the capital of France?",
]

# make test input for getting outputs
inputs = tuned.tokenizer(input_strings, 
                         return_tensors="pt",
                         padding=True)
# TODO: use src.data group_examples to group the inputs
# future will not need padding

# do inference 
# outputs = tuned.inference(inputs)

# specify which parameters to get activations from
layer_list = [
    f"model.model.layers[0].self_attn.{p}_proj" for p in "q" 
    # this is the output of the fine-tuning layer. I think
]
# TODO: src.data for query recipes

# do inference with activations
outputs = tuned.get_activations(inputs, layer_list,
                                num_return_sequences=2)

# change device to cpu
# (otherwise I can't open it afterwards)
dev = torch.device('cpu')
outputs = analysis.move_output_tensors(outputs, dev)

# checking the outputs
# print(outputs["activations"].keys())
# layer_name = list(outputs["activations"].keys())[0] # layer name
# print(len(outputs["activations"][layer_name])) # number of activations
# print(outputs['activations'][layer_name][0].shape)

print(outputs.keys())

# set experiment description
outputs['description'] = description

# save the outputs
# with open(savefile, 'a') as file:
#     torch.save(outputs, file)

with open(savefile, 'wb') as file:
    pickle.dump(outputs, file)

# save_file(outputs, savefile+".safetensors")

print("finished")