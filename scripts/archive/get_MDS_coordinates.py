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
description = """Getting similarity matrices and MDS coordinates from pre-computed activations"""

recipes = [ # 
    [1.0, 0.0],  # all from topic 8
    [0.8, 0.2],  # 80% from topic 8, 20% from topic 9
    [0.5, 0.5],  # 50% from topic 8, 50% from topic 9
    [0.2, 0.8],  # 20% from topic 8, 80% from topic 9
    [0.0, 1.0],  # all from topic 9
]
proj_type = "o"  # the type of proj/activations we look at

similarity = "bw"

print("PWD:  ", os.getcwd())

        ############################################
        #           Actually Doing Things          #
###############################################################################  

for layer in tqdm(range(1, 16), desc=description):

    layer = f'{int(layer):0{2}d}'   # pad the layer

    ## Pre-Processing for looking at saved matrices
    layer_name = f'{layer}_{proj_type}-proj_B'

    # collect the filepaths
    filepaths = []
    # - First, get the cache location
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    peft_model_id = "results/test_finetune_llama_yahoo9_4/checkpoint-77885" 
    save_path = analysis.make_cache_dir_name(base_model_path=base_model_id,
                                            peft_path=peft_model_id)
    # cache_path = cache_path.replace("home/mohata1/scratch", "scratch/") # for running on GPU node
    # save_path = os.path.join(cache_path, "activations")
    # save_path = '/weka/home/mohata1/scratchcpriebe1/MO/network_manifolds/results/cache/meta-llama__Llama-3.2-1B-Instruct___test_finetune_llama_yahoo9_4__checkpoint-77885/activations/'
    print(save_path)

    # - iterate over the recipes to make the tensor names
    # -- Parameters from running the inference
    act_count = 5
    n_replicates = 10
    n_queries = 10
    dataset_name = "yahoo"
    data_name = f"r{n_replicates}_q{n_queries}_{dataset_name}"

    # -- Doing the loop
    for rec in recipes:
        recipe_name = f"t8={str(rec[0])}_t9={str(rec[1])}"   # see get_activations_recipes.py
        tensorname = layer_name + "___" + data_name + "_" + recipe_name + f"_{act_count}_seed0.safetensors"

        filepath = os.path.join(save_path, "activations", tensorname)
        filepaths.append(filepath)

        print(filepath)

    # Load in the tensors
    print("Loading activation tensors")
    activations = []
    for filename in filepaths:
        act = {}
        with safe_open(filename, framework="pt") as f:
            print("keys: ", f.keys())
            for k in f.keys():
                act[k] = f.get_tensor(k)
                # print(act[k])
        
        activations.append(matrix.Matrix(act["activations"]).flatten()) 
            # save as Matrix and flatten

    print("Number of Activation Matrices: ", len(activations))

    for similarity in "bw", "fro":
        # Make the save file name
        count = f"{layer}_{similarity}"
        name = f"mds_coords{count}"
        coord_savefile = os.path.join(GLOBAL_PROJECT_DIR, "scripts/sync/coordinates", name)

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
        to_save = {"coordinates" : torch.from_numpy(coordinates),
                "similarity_matrix" : torch.from_numpy(sim_out)}

        # save the coordinates
        print(f"Saving coordinates to {coord_savefile}")
        save_file(to_save, coord_savefile)

print("SUCCESS!")