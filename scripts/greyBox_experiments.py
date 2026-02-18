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
    parser.add_argument('--batch', type=int, default=10,
                        help="inference batch size for memory purposes")
    # WISHLIST add sampling recipe parameter(s)

    ### ### TOKENIZATION / INPUT FORMAT
    parser.add_argument("--max_length", type=int, default=512,
                        help="maximum sequence length for tokenization \
                            (default: 512)")
    # parser.add_argument("--")

    ### ### INFERENCE METHOD
    parser.add_argument('--pipe_inference', action='store_true', default=False,
                        help="Whether to use pipeline for inference")

    ### ### EMBEDDING / OUTPUT FORMAT
    parser.add_argument("--decode_output", action="store_true", default=False,
                        help="whether to decode outputs to string")
    parser.add_argument("--embed_output", action="store_true", default=False,
                        help="whether to embed inference ouputs")
    

    ### MDS 
    ### ### Output files to access
    parser.add_argument("--outputs_dir", type=str, default=None,
                        help="directory containing output files from \
                            inference step (relative to project dir)")

    ### RESULTS
    parser.add_argument("--dir", type=str, default="results/00_logs",
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


def validate_args(args, logfile):
    # checking specific dependencies
    to_log = f"""\nValidating arguments for {args.id}..."""
    print_and_write(display_message(to_log), logfile)

    # decode and embed

    if args.embed_output and not args.decode_output:
        msg = """To embed outputs, they must first be decoded, but flag 
              --decode-output was not included. Forcing decode-output to true
              """
        print_and_write(display_message(msg))
        args.decode_output = True
    
    if args.pipe_inference and not args.decode_output:
        msg = """Pipe inference automatically decodes output but flag 
                --decode-output was not included. Forcing decode_output=True"""
        print_and_write(display_message(msg))
        args.decode_output = True

    return args


def do_code(args):
    ### Logging Setup
    start_time = time.time()
    logfilename = args.id + "___" + args.logfile
    savedir = args.dir # WISHLIST validation for savedir (utils file checking probably)
    logfile = make_logfile(os.path.join(GLOBAL_PROJECT_DIR, savedir, 
                                        logfilename))
    # WISHLIST outsource pipeline logging (probably utils)
    # WISHLIST better config fitting so that it can be used to find where files were saved
        # (just means generating result savefile names earlier, 
        # updating the log name, etc)
    save_config(os.path.join(GLOBAL_PROJECT_DIR, savedir, f"config_{args.id}.json"), 
                vars(args))
    to_log = f"""{time_elapsed_str(start_time)}Beginning experiment 
            {args.id}"""
    print_and_write(display_message(to_log), logfile)

    args = validate_args(args, logfile)

    ### Inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.do_inference:
        ### Make the analysis object
        model = analysis.LoraAnalysis(base_model_path=args.base_model, 
                                    peft_path=args.peft_model,
                                    custom_name=args.model_name)
        device = model.device
        
        to_log = f"""{time_elapsed_str(start_time)}Built analyzer for
                     {args.base_model} with PEFT {args.peft_model}"""
        print_and_write(display_message(to_log), logfile)
        

        ### Load and process the dataset from HuggingFace
        dataset = datasets.load_dataset(args.dataset, split=args.dataset_split)
        # nbatch = int(args.n_query // args.batch)
        # sampling 
        # WISHLIST add sampling recipe options here
        dataset = data.sample_datasets_uniform(args.n_query, args.data_seed, 
                                            dataset)
        to_log = f"""{time_elapsed_str(start_time)}Loaded dataset 
                 {args.dataset} split {args.dataset_split} and sampled 
                 {args.n_query} examples"""
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
            context = data.get_context_by_line(context_path, args.context_line)
            dataset = dataset.map(lambda x: data.add_context(x, context=context))
            to_log = f"""{time_elapsed_str(start_time)}using context 
                     \"{context}\""""
            print_and_write(display_message(to_log), logfile)

        # generate inputs
        # WISHLIST add options for tokenizing
        if not args.pipe_inference:
            inputs = model.tokenizer(dataset["example"],
                                    padding="max_length", # pad to max length 
                                    truncation=True, # truncate to max length 
                                    return_tensors="pt", # return PyTorch tensors
                                    max_length=args.max_length,
                                    ) # WISHLIST option for max input length
        else:
            inputs = dataset["example"] # pipe takes untokenized
        input_keys = dataset["id"]

        # WISHLIST additional information here
        to_log = f"""{time_elapsed_str(start_time)}Processed and tokenized 
                 dataset"""
        print_and_write(display_message(to_log), logfile)

        ### Run inference (no activations right now) and save outputs
        # WISHLIST add options for collecting activations
        # WISHLIST turn this process into another method somewhere else?
        if args.pipe_inference:
            pipe = model.get_pipe()
            def do_inference(ex):
                out = pipe(ex, 
                           num_return_sequences=args.n_replicate,
                           max_new_tokens=args.max_length,
                           return_full_text=False) # WISHLIST settings
                out = model.postprocess_pipe(out, embed=args.embed_output)
                return out
        else:
            def do_inference(ex):
                out = model.inference(ex, n_replicates=args.n_replicate, 
                                    decode_outputs=args.decode_output,
                                    max_new_tokens=args.max_length)
                if args.embed_output: # WISHLIST additional parameters
                    out = analysis.embed_strings(out) 
                return out
        
        to_save = {} # id -> all replicates 
        descstr = f"Inference with {args.n_replicate} replicates"
        for i, ex in tqdm(enumerate(inputs), desc=descstr):
            outputs = do_inference(ex)
            to_save[f"id_{input_keys[i]}"] = outputs
                    # if decoded but not embedded: list of strings
                    # if decoded and embedded: tensor (n_replicate, embed_dim)
                    # if not decoded: tensor of tokens (n_replicate, 2* max_length)
                
        # PIPELINE
        # PIPELINE DOESNT WORK HERE YET
        # if args.pipe_inference:
        #     outputs = model.inference_pipe(inputs, n_replicates=args.n_replicate)
        #     outputs = model.postprocess_pipe(outputs, embed=args.embed_output)

        # print(outputs)
        # if args.embed_output: # WISHLIST additional parameters
        #     outputs = analysis.embed_strings(outputs)
            # ( n_query x n_replicates, embed_dim ) <- 768 for Nomic embed 

        to_log = f"""{time_elapsed_str(start_time)}Ran inference on 
                {args.n_query} queries with {args.n_replicate} replicates"""
        to_log += f" and sent to embedding dim {outputs.shape[1]}" if \
            args.embed_output else ""
        print_and_write(display_message(to_log), logfile)

        # save output sequences as safetensors
        # outputs = outputs["sequences"] # we ONLY want the sequences here
            # not necessary because this is outputting a tensor!
            # TODO adjust for various formats of outputs? might need to be in model_analysis
        outpathname = "embeds" if args.embed_output else "outputs"
        outdir = os.path.join(GLOBAL_PROJECT_DIR, model.cache_file_path, 
                                outpathname)
        filename = f"{args.id}_r{args.n_replicate}_q{args.n_query}_seed{args.data_seed}"
        
        if args.decode_output and (not args.embed_output):
            filename += "_strings.pkl"
            def save_outputs(outs, filepath):
                with open(filepath, "wb") as file:
                    pickle.dump(outs, file)
                return
        else:
            filename += "_tokens" if not args.embed_output else ""
            filename += ".safetensors"
            def save_outputs(outs, filepath):
                save_file(outs, filepath)
                return
                      
        filepath = os.path.join(GLOBAL_PROJECT_DIR, outdir)
        if not os.path.exists(filepath):
            to_log = f"""{time_elapsed_str(start_time)}Creating output 
                    directory {filepath}"""
            os.makedirs(filepath, exist_ok=True)
            print_and_write(display_message(to_log), logfile)
        filepath = os.path.join(filepath, filename)

        save_outputs(to_save, filepath)

        to_log = f"""{time_elapsed_str(start_time)}Saved outputs to 
                {filepath}"""
        print_and_write(display_message(to_log), logfile)
    

    ### MDS Analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.do_MDS:
        pass # TODO implement MDS analysis


    close_files(logfile)


main()
print("SUCCESS")
    

    
    