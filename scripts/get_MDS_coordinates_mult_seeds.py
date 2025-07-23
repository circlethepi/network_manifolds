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
import argparse
import re
import time

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

def build_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0,
                        help="layer to look at (default: 0)")
    parser.add_argument("--act-count", "--act-id", type=int, default=0, nargs=1,
            help="ID or count of the activation run to look at (default: 0)")
    parser.add_argument("--dir", type=str, default="results",
                        help="destination directory to save the results to {relative to the project dir} (default: results)")
    parser.add_argument("--logfile", type=str, default="sim_MDS.log",
                        help="log file to save the results to (default: sim_MDS.log)")
    
    return parser

def get_args(*args_to_parse):
    """Get the arguments from the command line"""
    parser = build_parser(*args_to_parse)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    do_code(args)


recipes = [ # 
    [1.0, 0.0],  # all from topic 8
    [0.8, 0.2],  # 80% from topic 8, 20% from topic 9
    [0.5, 0.5],  # 50% from topic 8, 50% from topic 9
    [0.2, 0.8],  # 20% from topic 8, 80% from topic 9
    [0.0, 1.0],  # all from topic 9
]
proj_type = "o"  # the type of proj/activations we look at

# -- Parameters from running the inference
# act_count = 0
n_replicates = 10
n_queries = 40

SEPARATOR = "\n" + "="*80 + "\n"
TIMESEP = "\n" + "-  -"*20 + "\n"

def time_elapsed_str(start_time):
    """returns the time elapsed since start_time in seconds"""
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    fraction = elapsed - int(elapsed)
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}.{int(fraction * 100):02d}]\t"

print("PWD:  ", os.getcwd())

        ############################################
        #           Actually Doing Things          #
###############################################################################  

def do_code(args):
    start_time = time.time()
# for layer in tqdm(range(0, 16), desc=description):
    layer = args.layer
    act_count = args.act_count  # get the activation count from the args
    savedir = args.dir

    # logging and config
    logfilename = args.logfile
    logfile = make_logfile(os.path.join(GLOBAL_PROJECT_DIR, savedir, logfilename))
    save_config(os.path.join(GLOBAL_PROJECT_DIR, savedir, "config.json"), vars(args))


    layer_str = f'{int(layer):0{2}d}'   # pad the layer
    print_and_write(f"{time_elapsed_str(start_time)}Layer: {layer_str}\n\n", logfile)

    ## Pre-Processing for looking at saved matrices
    layer_name = f'{layer_str}_{proj_type}-proj_B'

    # collect the filepaths
    filepaths = []
    # - First, get the cache location
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    peft_model_id = "results/test_finetune_llama_yahoo9_4/checkpoint-77885" 
    save_path = analysis.make_cache_dir_name(base_model_path=base_model_id,
                                            peft_path=peft_model_id)
    
    # print(save_path)
    all_files = sorted(os.listdir(os.path.join(save_path, "activations")))

    # - iterate over the recipes to make the tensor names
    dataset_name = "yahoo"
    data_name = f"r{n_replicates}_q{n_queries}_{dataset_name}"

    # -- Doing the loop. ### UNCOMMENT HERE TO CHECK RECIPE NAME FILE COLLECTION
    for rec in recipes:
        recipe_name = f"t8={str(rec[0])}_t9={str(rec[1])}"   # see get_activations_recipes.py
        tensorname = layer_name + "___" + data_name + "_" + recipe_name + f"_{act_count}"
        pattern = re.compile(rf"^{re.escape(tensorname)}_.+\.safetensors$") # ignore seed (for now)

        found_files = [os.path.join(save_path, "activations", f) for f in \
                      all_files if pattern.match(f)]
        filepaths += found_files
        print_and_write(f"{time_elapsed_str(start_time)}Found {len(found_files)} files for recipe {rec}", logfile)

    # Load in the tensors
    print(f"\n{SEPARATOR}{time_elapsed_str(start_time)}Loading activation tensors")
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
        
        print_and_write(filename, logfile)

    print_and_write(f"\n{time_elapsed_str(start_time)}Total Number of Activation Matrices: " + len(activations), logfile)
    

    for similarity in ("bw", "fro"):
        # Make the save file name
        count = f"{int(layer_str):0{2}d}_{similarity}"
        name = f"mds_coords{count}"
        coord_savefile = os.path.join(GLOBAL_PROJECT_DIR, savedir, name)

        # Calculate the similarity
        print_and_write(f"\n{SEPARATOR}{time_elapsed_str(start_time)}Calculating {similarity} similarity", logfile)
        if similarity == "bw":
            covariances = [m.T.matrix @ m.matrix for m in activations]
            sim_out = matrix.matrix_similarity_matrix(*covariances, sim_type="bw", 
                                                    aligned=False)
        else:
            sim_out = matrix.matrix_similarity_matrix(*activations, sim_type="fro")

        # get the MDS coordinates
        print_and_write(f"{time_elapsed_str(start_time)}Calculating MDS coordinates", logfile)
        coordinates = dkps.compute_MDS(sim_out, n_components=2, align_coords=True)
        to_save = {"coordinates" : torch.from_numpy(coordinates),
                "similarity_matrix" : torch.from_numpy(sim_out)}

        # save the coordinates
        print_and_write(f"{time_elapsed_str(start_time)}Saving sims and coords to {coord_savefile}", logfile)

        save_file(to_save, coord_savefile)
    
    print_and_write(f"{SEPARATOR}{time_elapsed_str(start_time)}Finished all seeds", logfile)
    close_files(logfile)
    

main()
print("SUCCESS!")