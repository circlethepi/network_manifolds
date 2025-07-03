"""
Script to calculate similarities and corresponding MDS coordinates from 
existing activation matrices and their (uncentered) covariances.
"""

############################################
#              Import Packages             #
###############################################################################  
import torch
import os
import pickle
from tqdm import tqdm

# huggingface
import datasets
from safetensors.torch import save_file, safe_open

# custom
from src.utils import *
import src.data as data
import src.model_analysis as analysis
import src.matrix as matrix
import src.dkps as dkps

    ############################################
    #              Set Parameters              #
###############################################################################  

count = 0
name = f"mds_coords{count}"

description = """Getting similarity matrices and MDS coordinates from pre-computed activations"""

coord_savefile = os.path.join(GLOBAL_PROJECT_DIR, "results/coordinates", name)

recipes = [ # 
    [1.0, 0.0],  # all from topic 8
    [0.8, 0.2],  # 80% from topic 8, 20% from topic 9
    [0.5, 0.5],  # 50% from topic 8, 50% from topic 9
    [0.2, 0.8],  # 20% from topic 8, 80% from topic 9
    [0.0, 1.0],  # all from topic 9
]

proj_type = "o"  # the type of proj/activations we look at

layer = 0

similarity = "fro"

        ############################################
        #           Actually Doing Things          #
###############################################################################  

## Pre-Processing
layer = f'{int(layer):0{2}d}'   # pad the layer
layer_name = f'{layer}_{proj_type}-proj_B'

# collect the filepaths
filepaths = []
# - First, get the cache location
base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
peft_model_id = "results/test_finetune_llama_yahoo9_4/checkpoint-77885" 
save_path = analysis.make_cache_dir_name(base_model_path=base_model_id,
                                         peft_path=peft_model_id)

# - iterate over the recipes to make the tensor names
# -- Parameters from running the inference
act_count = 5
n_replicates = 10
n_queries = 10
dataset_name = "yahoo"
data_name = f"r{n_replicates}_q{n_queries}_{dataset_name}"

# -- Doing the loop
for rec in recipes:
    recipe_name = f"t8={rec[0]}_t9={rec[1]}"   # see get_activations_recipes.py
    tensorname = layer_name + "___" + data_name + "_" + recipe_name + \
        f"_{act_count}_seed0.safetensors"
    filepath = os.path.join(save_path, "activations", tensorname)
    
    print(filepath)

# Load in the tensors
print("Loading activation tensors")
activations = []
for filename in filepaths:
    act = {}
    with safe_open(filename, framework="pt") as f:
        for k in f.keys():
            act[k] = f.get_tensor(k)
    
    activations.append(matrix.Matrix(act["activations"]).flatten()) 
        # save as Matrix and flatten


# Calculate the similarity
print(f"Calculating {similarity} similarity")
if similarity == "bw":
    covariances = [m.T.matrix @ m.matrix for m in activations]

    sim_out = matrix.matrix_similarity_matrix(*covariances, sim_type="bw", 
                                              aligned=False)
else:
    sim_out = matrix.matrix_similarity_matrix(*activations, sim_type="fro")


# get the MDS coordinates
print("Calculating MDS coordinates")
coordinates = dkps.compute_MDS(sim_out, n_components=2, align_coords=True)
coordinates = {"coordinates" : torch.from_numpy(coordinates)}

# save the coordinates
print(f"Saving coordinates to {coord_savefile}")
save_file(coordinates, coord_savefile)

print("SUCCESS!")