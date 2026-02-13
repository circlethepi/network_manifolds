"""
Script to collect activations from a model

Hopefully becomes framework for general experiments

collects generated output sequences for a model

Merrick Ohata
JHU AMS 2026
"""
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

# custom imports
from src.utils import *
import src.data as data
import src.model_analysis as analysis

from safetensors.torch import save_file


    ############################################                   
    #              Set Parameters              #
###############################################################################

def build_parser():
    parser = argparse.ArgumentParser()

    ### META
    parser.add_argument("--id", type=str, default=None, 
                        help="id for experiment org")
    
    ### ### TASK
    parser.add_argument("--do_inference", action="store_true", 
                        help="whether to run inference", default=False)
    parser.add_argument("--do_MDS", action="store_true",
                        help="whether to run MDS analysis", default=False)

    ### INFERENCE
    ### ### MODEL
    parser.add_argument("--base_model", type=str,
                        default="meta-llama/Llama-3.2-1B-Instruct",
                        help="base model to analyze")
    parser.add_argument("--peft_model", type=str, default=None,
                        help="path to peft model to analyze. Should be a path \
                            to a directory containing the model files")
    parser.add_argument("--model_name", type=str, default=None,
                        help="custom name for the model (for logging and cache\
                            purposes)")
    parser.add_argument("--context_file", type=str, default=None,
                        help="path to context file for context window content \
                            (default: None)")
    parser.add_argument("--context_line", type=int, default=0,
                        help="line number in context file to use for context \
                            window (default: 0)")
    
    ### ### DATASET
    parser.add_argument("--dataset", type=str, default="yahoo_answers_topics",
                        help="dataset to use for inference. Should be a name \
                            from the HuggingFace datasets library")
    parser.add_argument("--dataset_split", type=str, default="test",
                        choices=["train", "test"], help="split of the dataset \
                            to use for inference (default: test)")
    parser.add_argument("--data_seed", type=int, default=0,
                        help="random seed for data sampling")
    parser.add_argument("--concat_method", type=str, default="all", 
                        choices=["all", "question"], help="method for \
                            concatenating the question and answer")
    
    ### ### QUERIES / SAMPLING
    parser.add_argument("--n_query", type=int, default=40,
                        help="number of queries for each recipe \
                            (default: 40)")
    parser.add_argument("--n_replicate", type=int, default=10, 
                        help="number of replicates for each query \
                            (default: 10)")
    # TODO add sampling recipe parameter(s)

    ### ### TOKENIZATION / INPUT FORMAT
    parser.add_argument("--max_length", type=int, default=512,
                        help="maximum sequence length for tokenization \
                            (default: 512)")
    # parser.add_argument("--")

    ### MDS 
    ### ### Output files to access
    parser.add_argument("--outputs_dir", type=str, default=None,
                        help="directory containing output files from \
                            inference step (relative to project dir)")

    ### RESULTS
    parser.add_argument("--dir", type=str, default="results",
                        help="destination directory for results {relative to \
                            the project dir} (default: results)")

    ### LOGGING
    parser.add_argument("--logfile", type=str, default="log.log")

    return parser
    


def get_args(*args_to_parse):
    """Get the arguments from the command line"""
    parser = build_parser(*args_to_parse)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    do_code(args)


def do_code(args):
    ### Logging Setup
    start_time = time.time()
    logfilename = args.id + "___" + args.logfile
    savedir = args.dir # WISHLIST add validation for savedir (utils file checking probably)
    logfile = make_logfile(os.path.join(GLOBAL_PROJECT_DIR, savedir, 
                                        logfilename))
    save_config(os.path.join(GLOBAL_PROJECT_DIR, savedir, f"config_{args.id}.json"), 
                vars(args))
    to_log = f'{time_elapsed_str(start_time)}Beginning experiment {args.id}'
    print_and_write(display_message(to_log), logfile)

    ### Inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.do_inference:
        ### Make the analysis object
        model = analysis.LoraAnalysis(base_model_path=args.base_model, 
                                    peft_path=args.peft_model,
                                    custom_name=args.model_name)
        device = model.device
        
        to_log = f'{time_elapsed_str(start_time)}Built analyzer for \
            {args.base_model} with PEFT {args.peft_model}'
        print_and_write(display_message(to_log), logfile)
        

        ### Load and process the dataset from HuggingFace
        dataset = datasets.load_dataset(args.dataset, split=args.dataset_split)
        # sampling 
        # TODO add sampling recipe options here
        dataset = data.sample_datasets_uniform(args.n_query, args.data_seed, 
                                            dataset)
        to_log = f'{time_elapsed_str(start_time)}Loaded dataset {args.dataset} \
            split {args.dataset_split} and sampled {args.n_query} examples'
        print_and_write(display_message(to_log), logfile)

        # process into format we want
        # WISHLIST concatenation options for other datasets
        dataset = dataset.map(lambda x: data.concat_yahoo(x, concat_method=
                                                        args.concat_method))
        # add context if context file provided
        if args.context_file is not None:
            message ="""If providing a context file, must also provide a line 
            number for which context to use"""
            assert args.context_line is not None, display_message(message)
            context_path = os.path.join(GLOBAL_PROJECT_DIR, args.context_file)
            context = data.get_context(context_path, args.context_line)
            dataset = dataset.map(lambda x: data.add_context(x, context=context))

        # generate inputs
        # TODO add options for this
        inputs = model.tokenizer(dataset["example"],
                                padding="max_length", # pad to max length 
                                truncation=True, # truncate to max length 
                                return_tensors="pt", # return PyTorch tensors
                                max_length=args.max_length,
                                ) # TODO option for max input length
        
        # WISHLIST additional information here
        to_log = f'{time_elapsed_str(start_time)}Processed and tokenized dataset'
        print_and_write(to_log, logfile)

        ### Run inference (no activations right now) and save outputs
        # TODO add options for collecting activations
        # TODO turn this process into another method somewhere else
        outputs = model.inference(inputs, n_replicates=args.n_replicate,)

        to_log = f'{time_elapsed_str(start_time)}Ran inference on {args.n_query} \
            query with {args.n_replicate} replicates'
        print_and_write(display_message(to_log), logfile)

        # save output sequences as safetensors
        # outputs = outputs["sequences"] # we ONLY want the sequences here
            # not necessary because this is outputting a tensor!
        filename = f"{args.id}_r{args.n_replicate}_q{args.n_query}_seed{args.data_seed}.safetensors" # file name and filepath for saving
        filepath = os.path.join(GLOBAL_PROJECT_DIR, savedir, filename)

        to_save = {"outputs": outputs}
        save_file(to_save, filepath)

        to_log = f'{time_elapsed_str(start_time)}Saved outputs to {filepath}'
        print_and_write(display_message(to_log), logfile)
    
    ### MDS Analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.do_MDS:
        pass # TODO implement MDS analysis

    close_files(logfile)


main()
print("SUCCESS")
    

    
    