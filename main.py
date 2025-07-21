#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#               Main Script to work from the Command Line   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
Main script to run various portions of the code
"""

# imports
from src import data, dkps, matrix, plot, model_analysis as analysis
from src.utils import *

import argparse
import re
import os
import sys

import numpy as np
import torch

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                          Argument Parser for the Script         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
parser_desc = """Main script to run various portions of the code"""


def build_parser():
    """builds the argument parser for the script"""
    parser = argparse.ArgumentParser(description=parser_desc)
    
    # Arguments
    ## MODES / functionalities
    mode_choices = ["fine_tune", "inference", "sims", "mds_coords"]
    # parser.add_argument("--mode", type=str, required=True, # nargs="+", 
    #                     choices=mode_choices, help="mode(s) to run")
    

    ### FINE TUNING OPTIONS
    parser.add_argument("--fine_tune", action="store_true", 
                        help="run fine-tuning of the model")
    
    training_time = parser.add_mutually_exclusive_group(required=False)
    training_time.add_argument("--epochs", type=int, 
                               help="number of epochs to run for fine-tuning")
    training_time.add_argument("--steps", type=int,
                               help="number of steps to run for fine-tuning")
    

    ### INFERENCE OPTIONS



    ### INFERENCE & SIMILARITY OPTIONS
    parser.add_argument("--proj_type", type=str, default="o",
                        help="type of projection to use for activations")
    parser.add_argument("--act_id", type=int, default=0,
                        help=display_message("activation id in case of " \
                        "multiple runs with the same config"))
    parser.add_argument("--n_replicates", type=int, default=10,
                        help=display_message("number of replicates to use for " \
                        "calculating similarities"))
    parser.add_argument("--n_queries", type=int, default=40,
                        help=display_message("number of queries to use for " \
                        "calculating similarities"))
    parser.add_argument("--recipes", nargs="+", type=str, 
                        help="recipes as\"(p1, p2,...) (q1, q2,...)\"")


    ### SIMLARITY CALCULATION OPTIONS
    parser.add_argument("--sims", action="store_true",
                        help="calculate similarities from activations")
    # parser.add_argument("--layer_selection_rules", "--layer_rules", type=str, 
    #                     default="B", nargs="*", 
    #                     help="rules for selecting layers names")


    return parser

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               arg Pre-Processing           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #                            during build           
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #                            post build          
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

parser = build_parser() # get the args


def parse_recipes(recipes):
    """parses the recipes from the command line"""

    if check_if_null(recipes, True, False): # no values
        return None
    
    def single_recipe(recipes):
        match = re.fullmatch(r"\((.*?)\)", recipes.replace(" ", ""))
        if not match:
            raise ValueError(f"Invalid recipe format: {recipes}." \
                            "Use format '(x, y)'.")
        try:
            parts = [float(a) for a in match.group(1).split(",")]
            return tuple(parts)
        except ValueError:
            message = f"Invalid recipe values: {recipes}. " \
                  "Ensure all recipe values are numeric."
            raise argparse.ArgumentTypeError(display_message(message))

    parsed_recipes = []
    if len(recipes) == 1:  # single recipe
        all_tuples = recipes[0]
        tuples = re.findall(r"\s*\((.*?)\)\s*", all_tuples)
        for t in tuples:
            parsed_recipes.append(tuple([float(x.strip()) for \
                                         x in t.split(",")]))
    else:  # multiple recipes
        for r in recipes:
            parsed_recipes.append(single_recipe(r)) 
    return tuple(parsed_recipes)


def get_args(*args_to_parse):
    """gets the arguments from the command line + preprocessing"""
    args = parser.parse_args(*args_to_parse)

    # Recipe handling
    args.recipes = parse_recipes(args.recipes)

    return args



def main():
    args = get_args()
    
    do_code(args)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              Main Functionalities          
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def do_code(args):


    return


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                      Run           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
if __name__ == '__main__':
    main()
