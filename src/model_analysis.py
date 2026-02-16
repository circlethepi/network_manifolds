#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#               Accessing HuggingFace models and performing analysis   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
Structure to access HuggingFace models, perform inference, and store 
relevant values for inducing various DKPS representations. 
"""

import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from sentence_transformers import SentenceTransformer

from src.utils import check_if_null, get_device, is_int_or_int_string
import textwrap
import pickle


GLOBAL_ANALYSIS_DIR = "/weka/home/mohata1/scratchcpriebe1/MO/network_manifolds/results/cache"


# Memory stuff
def memoize(savepath, compute:callable, recompute=False, device="cuda"):
    """
    Saving and checking disc for computed quantity

    :param savepath :   the path to save the tensors to
    :param compute : callable   callable with no arguments to compute
    :param recompute : bool     whether to force recomputation 
    
    """
    device = torch.device(device)

    
    if not isinstance(savepath, Path):
        savepath = Path(savepath)

    if (not recompute) and savepath.exists():
        # load the cached result
        print(f"Loading cached result from {savepath}")

        tensors = {}
        with safe_open(savepath, framework='.pt') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        # WISHLIST only get part of the tensor ?
        # with f.get_slice("")

        # return torch.load(savepath, map_location=device)
        return tensors
    else:
        # compute the result
        print(f"Computing result and saving to {savepath}")
        result = compute()
        # torch.save(result, savepath)
        # save with huggingface safetensors
        save_file(result, savepath)

        return result


def query_cache(savename:str, cache_dict:dict, compute:callable, recompute=False):
    """
    Saving and checking cache for computed quantity
    """
    if recompute or (savename not in cache_dict):
        cache_dict[savename] = compute()
    return cache_dict[savename]



class LoraAnalysis:
    """
    Class to wrap a fine-tuned model for analysis
    """

    def __init__(self, base_model_path:str, peft_path:str, 
                #  peft_local:bool=True, 
                 build_model:bool=True, 
                 n_layers:Optional[int]=None, 
                 custom_name:Optional[str]=None,
                 optimize_for_inference:bool=True):
        """
        param: base_model_path : str    path/name/id of the base model
        param: peft_path : str          path/name/id of the fine-tuned model
        param: build_model : bool       whether to build the fine-tuned model.
                                        This is necessary for most analyses.
                                        (default: True)
        param custom_name : str|None    custom name to append to the cache path
        param optimize_for_inference : bool whether to optimize the model for inference
        """

        self.device = get_device()
        self.base_model_path = base_model_path
        
        # if peft_local: 
        #     peft_path = os.path.abspath(peft_path)
        self.peft_path = peft_path

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, 
                                                       use_fast=True)
                                                # WISHLIST add option to not
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if build_model:
            self.model = self.get_model().to(self.device)
        
        # set default weight naming function 
        self.layer_name_function = get_layer_name_function(base_model_path)
        self.layer_name_dict = None
        self.layer_name_dict_weights = None

        # caching
        # WISHLIST better caching functionality
        # WISHLIST add disc caching (eventually)
        cache_path = make_cache_dir_name(base_model_path, peft_path,
                                         custom_name=custom_name)        
        ## check if path exists
        if not os.path.exists(cache_path):
            print(f"Creating cache directory at {cache_path}")
            os.makedirs(cache_path)
        else:
            print(f"Using existing cache directory at {cache_path}")
        self.cache_file_path = cache_path

        self.cache = {} # cache key -> cached value

        ## optimization for inference
        if optimize_for_inference:
            # self.model.set_attention_implementation("flash_attention_2") 
            self.model.generation_config.cache_implementation = "static"
            # per huggingface documentation https://huggingface.co/docs/transformers/main/llm_optims?static-kv=1.+cache_implementation#optimizing-inference

    
    def clear_cache(self):
        """clear cache"""
        self.cache.clear()

    def to(self, device):
        self.model.to(device)


    def memoize(self, savename:Union[str, Path], compute:callable, 
                recompute=False, save_to_disc=True, save_to_cache=True,
                device=None):
        """ Checks for a computation to disc and/or cache """
        # FIXME this doesn't work currently

        device = check_if_null(device, self.device)

        if save_to_disc:
            savepath = os.path.join(self.cache_file_path, 
                                    savename + ".safetensors")
            compute = lambda compute_in=compute: memoize(savepath, 
                                                         compute=compute_in, 
                                                         recompute=recompute, 
                                                         device=device)
        if save_to_cache:
            compute = lambda compute_in=compute: query_cache(savename, 
                                                            compute=compute_in, 
                                                        recompute=recompute)
            
        return compute()



    def get_model(self):
        """
        Loads the LoRA model from the specified path. Usually used on build

        :return model : PeftModel
        """
        base_model = AutoModelForCausalLM.\
                        from_pretrained(self.base_model_path, 
                                        torch_dtype="auto", device_map="auto")
        # if no peft / LoRA, return the base model only
        if self.peft_path is None:
            return base_model

        return PeftModel.from_pretrained(base_model, self.peft_path)

    
    def get_lora_weights(self, layer_name_function:Optional[callable],
                         alternate_tensor_name:Optional[str]=None,
                         cache=True):
        """
        Load the LoRA weight matrices from file 

        :param layer_name_function : function|None
                                      f(key) -> {layer}_{type}proj_{lora A,B}
                                      function that renames keys in the tensor
                                      file to keys of standard format
        :param alternate_tensor_name : str|None    name of the file containing
                                       the weight tensors. defaults to 
                                       huggingface naming convention 
                                       "adapter_model.safetensors" inline

        :return tensor_dict : dict[matrix name] -> LoRA weights (A, B separate)
        """
        cache_key = "lora_weights"
        if cache and cache_key in self.cache.keys():
            return self.cache[cache_key]

        elnf = self.find_layer_name_func(custom_elnf=layer_name_function)
        
        # get path to weight tensors
        if alternate_tensor_name is not None:
            tensor_filename = alternate_tensor_name
        else:
            tensor_filename = "adapter_model.safetensors"
        tensor_path = os.path.join(self.peft_path, tensor_filename)

        # load the weight tensors and make dict
        # keys are renamed with the weight name function
        tensor_dict = {}
        with safe_open(tensor_path, framework="pt") as file: 
            # TODO add option to change device
            for k in file.keys():
                key = elnf(k)
                tensor_dict[key] = file.get_tensors(k) 
        
        if cache:
            self.cache[cache_key] = tensor_dict

        return tensor_dict
    

    def get_lora_layer_attribute_names(self, 
                                       alternate_tensor_name:Optional[str]=None,
                                       cache:bool=True,
                                       weights:bool=True):
        """get the attribute names for each of the LoRA matrices
        useful for registering forward hooks
        """
        cache_key = "lora_attribute_names"
        cache_key += "_weights" if weights else ""

        def weights_to_acts(name):
            # convert names for activation hooks vs accessing weights (default)
            return name.replace("weight", "default")

        if cache and cache_key in self.cache.keys():
            return self.cache[cache_key]
        
        elif not weights and cache and self.layer_name_dict is not None:
            layer_names = list(self.layer_name_dict.values())
        
        elif weights and cache and self.layer_name_dict_weights is not None:
            layer_names = list(self.layer_name_dict.values())
        
        elif cache and "lora_weights" in self.cache.keys():
            # if we have the weights, we can get the layer names from there
            layer_names = list(self.cache["lora_weights"].keys())

        # otherwise, load the tensor file and get the layer names
        else:
            if alternate_tensor_name is not None:
                tensor_filename = alternate_tensor_name
            else:
                tensor_filename = "adapter_model.safetensors"
            tensor_path = os.path.join(check_if_null(self.peft_path, ""), tensor_filename)

            # TODO get layer names for base models (no LoRA layers!!!) 


            with safe_open(tensor_path, framework="pt") as file: 
                layer_names = file.keys()
                
        # convert if necessary
        if not weights:
            layer_names = [weights_to_acts(n) for n in layer_names]

        if cache:
            self.cache[cache_key] = layer_names

        return layer_names


    def find_layer_name_func(self, custom_elnf:Optional[callable]=None):
        """determine weight name function"""
        if custom_elnf is not None:
            elnf = custom_elnf
        elif self.layer_name_function is not None:
            elnf = self.layer_name_function
        else:
            elnf = get_layer_name_function("DEFAULT")
        
        return elnf


    @property
    def lora_weights(self, layer_name_function:Optional[callable]=None, 
                     alternate_tensor_name:Optional[str]=None, cache:bool=True):
        """property to get lora weights"""
        return self.get_lora_weights(layer_name_function=layer_name_function,
                                alternate_tensor_name=alternate_tensor_name,
                                cache=cache)
    
    @property
    def lora_layer_attributes_weights(self, alternate_tensor_name:Optional[str]=None, 
                              cache:bool=True):
        """property to get lora weight attribute names"""
        return self.get_lora_layer_attribute_names(
                                alternate_tensor_name=alternate_tensor_name,
                                cache=cache, weights=True)
    
    @property
    def lora_layer_attributes(self, alternate_tensor_name:Optional[str]=None, 
                              cache:bool=True):
        """property to get lora weight attribute names"""
        return self.get_lora_layer_attribute_names(
                                alternate_tensor_name=alternate_tensor_name,
                                cache=cache, weights=False)
    
    def get_lora_layer_names(self, layer_name_function:Optional[callable]=None,
                         alternate_tensor_name:Optional[str]=None, 
                         cache:bool=True, weights=True):
        """property to get the names of the LoRA layers according to the
        model weight name function (architecture specific)"""

        cache_key = "lora_layer_names"
        cache_key += "_weights" if weights else ""

        if cache and cache_key in self.cache.keys():
            return self.cache[cache_key]
        
        elif weights and (self.layer_name_dict_weights is not None):
            return list(self.layer_name_dict_weights.keys())
        
        elif (self.layer_name_dict is not None) and not weights:
            return list(self.layer_name_dict.keys())

        # get the weight name function
        elnf = self.find_layer_name_func(custom_elnf=layer_name_function)

        atr_names = self.get_lora_layer_attribute_names(
                                alternate_tensor_name=alternate_tensor_name, 
                                cache=cache, weights=weights)
        layer_name_dict = dict(zip(
            [elnf(a) for a in atr_names],
            atr_names,
        ))

        names = list(layer_name_dict.keys())
        if weights:
            self.layer_name_dict_weights = layer_name_dict
        else:
            self.layer_name_dict = layer_name_dict

        names.sort()
        
        if cache:
            self.cache[cache_key] = names

        return names
    
    @property
    def lora_layer_names_weights(self, layer_name_function:Optional[callable]=None,
                         alternate_tensor_name:Optional[str]=None, 
                         cache:bool=True):
        return self.get_lora_layer_names(
                                layer_name_function=layer_name_function,
                                alternate_tensor_name=alternate_tensor_name,
                                cache=cache, weights=True)
    
    @property
    def lora_layer_names(self, layer_name_function:Optional[callable]=None,
                         alternate_tensor_name:Optional[str]=None, 
                         cache:bool=True):
        return self.get_lora_layer_names(
                                layer_name_function=layer_name_function,
                                alternate_tensor_name=alternate_tensor_name,
                                cache=cache, weights=False)


    def get_layer_names(self, rules:Optional[Union[list, str, int]],
                        int_pad:int=2,
                        attributes:bool=False, weights=True):
        """
        returns the layer name (renamed from layer name function) or attribute
        names of the given functions
        """
        all_strings = self.lora_layer_names_weights if weights \
                        else self.lora_layer_names
        
        # sets dict 
        all_names = filter_by_rules(all_strings=all_strings, 
                                    int_pad=int_pad,
                                    rules=rules)
        
        layer_dict = self.layer_name_dict_weights if weights else \
                        self.layer_name_dict

        # check if also returning attribute names
        if attributes:
            return {x: layer_dict[x] for x in all_names}
        else:
            return all_names
                


    def inference(self, inputs:dict, n_replicates:int=1, 
                  decode_outputs:bool=True, states:bool=False, 
                  attention:bool=False, max_new_tokens:int=512,
                  generate_kwargs:Optional[dict]=None,
                  decode_kwargs:Optional[dict]=None):
        """
        Do inference (does not collect activations). Inputs should be
        tokenized already.

        TODO add disc saving functionality
        TODO add support for max_length option
        TODO add documentation for this function
        WISHLIST collapse into inference_with_activations?
        """
        model = self.model

        model.eval()

        with torch.no_grad():
            outputs = model.generate(inputs=inputs['input_ids'].to(self.device),
                                     attention_mask=inputs['attention_mask']\
                                        .to(self.device), 
                                    num_return_sequences=n_replicates,
                                    max_new_tokens=max_new_tokens,
                                    return_dict_in_generate=False,
                                    output_hidden_states=states,
                                    output_attentions=attention,
                                    **generate_kwargs)
            if decode_outputs:
                outputs = self.tokenizer.batch_decode(outputs, 
                                                      skip_special_tokens=True,
                                                      **decode_kwargs)

        return outputs
    

    def inference_with_activations(self, inputs:dict, layers:list[str], 
                        return_outputs:bool=True, states:bool=True, 
                        attention:bool=True,
                        num_return_sequences:int=1,
                        max_new_tokens:int=512,

                        input_name:Optional[str]=None,
                        output_device=None):
        """
        Do inference and collect activations. 
        Inputs should be a tokenized dictionary.
        Layers should be in the form of a path. 

        WISHLIST add option/way to specify which layers by number, projection, etc
        TODO add option to save generated outputs
        WISHLIST add option to save hidden states, attentions
        """

        # saving activations and outputs to disc
        act_save_path = os.path.join(self.cache_file_path, "activations")
        out_save_path = os.path.join(self.cache_file_path, "outputs")

        if not os.path.exists(act_save_path):
            print(f"Creating activation directory\n{act_save_path}")
            os.makedirs(act_save_path, exist_ok=True)
        
        outputs = inference_with_activations(
            input=inputs,
            layers=layers,
            model=self.model,
            return_outputs=return_outputs,
            states=states,
            attention=attention,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,

            # saving
            save_dir=act_save_path,
            save_to_disc=True,
            input_name=input_name,
            layer_name_func=self.layer_name_function,

            output_dir=out_save_path,
            output_device=output_device,
        )

        return outputs

    # def pipe_inference(self, inputs:dict, s)

    # def generate_with_activations(self, inputs:dict, layers:list[str], 
    #                     return_outputs:bool=True, states:bool=True, 
    #                     attention:bool=True,
    #                     num_return_sequences:int=1,
    #                     max_length:int=512,

    #                     input_name:Optional[str]=None,
                        
    #                     recompute:bool=False,
    #                     save_to_disc:bool=True,
    #                     save_to_cache:bool=False):
    #     """
    #     Do inference for generation and collect activations
    #     Inputs should be a tokenized dictionary and layers in the form of an
    #     attribute path. 

    #     """
    #     # WISHLIST a way to get the desired layers from numbers and proj type
    #     # TODO documentation for this method




    #     return


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                           Helpers for LoraAnalysis            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# functions that parse the names of the .safetensor weight tensors
def llama_elnf(k:str):
    """
    makes weight matrix keys for llama architectures 
        f(key) -> {layer}_{type}proj_{lora A,B}
    
    :param k : str  
    """

    # get each of the identifiers we want
    layer_number = re.search(r'\b\d{1,2}\b', k).group()
    proj_type = re.search(r'\b([qkvo])_proj', k).group(1)
    lora_id = re.search(r'\blora_([AB])', k).group(1)

    # pad layer number to have same number of leading 0s
    padded_number = f'{int(layer_number):0{2}d}'

    name = f"{padded_number}_{proj_type}-proj_{lora_id}"
    return name

def default_elnf(k:str):
    """
    default weight name function. Minimal changes; only removes some strings
    that might appear in a layer name 
    """
    return


layer_name_func_dict = {
    "DEFAULT" : llama_elnf,      # TODO create/add default weight name func
    "meta-llama" : llama_elnf,
}


def get_layer_name_function(base_model_path:str):
    """
    finds weight name parsing function for loading model LoRA weights from
    huggingface .safetensors files
    """

    # check if default 
    if base_model_path == "DEFAULT":
        return layer_name_func_dict["DEFAULT"]

    matches = [word for word in layer_name_func_dict.keys() \
               if word in base_model_path]
    
    assert len(matches) <= 1, "multiple model names matched"

    if len(matches) > 0:
        m = matches[0]
        print(f"using predefined weight name parsing function for {m}")
        return layer_name_func_dict[m]
    else:
        print("No predefined function exists."\
              "DEFAULT will be used. Check that DEFAULT is suitable for the " \
              "architecture of the model. Enter custom weight name parsing " \
              "function if necessary where applicable.")
        return
    

def pad_number(num, pad):
    return f"{int(num):0{pad}d}"


def filter_by_rules(all_strings:list, rules:Optional[Union[list, str, int]],
                    int_pad:int=2):
    """
    Filters all_strings according to the substring rules
    - if rule is int or a string of int: keep strings containing ints
    padded to length of int_pad
    - if rule is str: keep strings containing the rule as a substring
    - if rule is list: apply listed rules as as "OR"s

    Each rule is generally an "AND" filtering rule. Rules that are given as 
    lists are "OR"s 
    """
    filtered = all_strings.copy()

    for k in rules:
        # print(k)
        if isinstance(k, list):
            sub_rules = [pad_number(j, int_pad) for j in k if is_int_or_int_string(j)]
            sub_rules += [j for j in k if not is_int_or_int_string(j)]
            filtered = [s for s in filtered if any(sub in s \
                                                   for sub in sub_rules)]
        elif is_int_or_int_string(k):
            substr = pad_number(k, int_pad)
            filtered = [s for s in filtered if substr in s]
        elif isinstance(k, str):
            filtered = [s for s in filtered if k in s]
        else:
            # Ignore unsupported types
            continue
    
    return list(filtered)


def make_cache_dir_name(base_model_path:str, peft_path:Optional[str]=None, 
                        custom_name:Optional[str]=None):
    """
    returns the path to the analysis cache directory
    :param base_model_path : str   path to the base model
    :param peft_path : str         path to the fine-tuned model
    :param custom_name : str|None  custom name to append to the cache path
                                    (default: None)
    :return pathname : str         path to the cache directory

    This function will create a path in the form of
    {GLOBAL_ANALYSIS_DIR}/{base_model_path}/{peft_path}/{custom_name}
    """

    # WISHLIST add option to add custom rules
        # may be useful for models other than llama

    rules = [ # (compile/pattern, replacement)
        (re.compile(r'^.*?results/'), ""),   # remove path to results dir
        (re.compile(r'/'), "__")           # replace "/" for names
    ]
    paths = [base_model_path]
    if check_if_null(peft_path): 
        paths.append(peft_path)
    # apply the replacement rules
    names = []
    for s in paths:
        for pattern, replacement in rules:
            s = pattern.sub(replacement, s)
        names.append(s)
    names = "___".join(names)

    # check if we have a custom name
    custom_name = check_if_null(custom_name, "")
    if custom_name != "":
        names += f"/{custom_name}"

    # make the path
    pathname = os.path.join(GLOBAL_ANALYSIS_DIR, names)

    return pathname


def get_nested_attr(obj:object, path:str):
    """accesses an attribute from a path that is a string. Helper for accessing
    attributes from namestrings iteratively.

    :param obj : object     object should have the desired attribute
    :param path : str       string of the way we would access the attribute 
                            directly inline
    
    :return object.path 

    example. Suppose `B` is an object with attribute `A`. Attribute `A` is also 
             an object with attribute `Z`, which we want to access.

             We can either do this with `B.A.Z` inline, or with 
             `get_nested_attr(B, "A.Z")`. 
    """
    # Split the path by the dot notation; make indices into splits
    parts = path.replace('[', '.').replace(']', '').split('.')
    
    for part in parts:
        # If the part is an index, access the list by index
        if part.isdigit():
            obj = obj[int(part)]
        else:
            # Otherwise, use getattr to access the attribute
            obj = getattr(obj, part)
    return obj


# getting activations
# useful to also get the output information (expensive otherwise)
def inference_with_activations(input:dict, layers:list[str], model,
                               return_outputs:bool=True, 
                               states:bool=False,
                               attention:bool=False, 
                               num_return_sequences:int=1,
                               max_length:int=512,

                               # saving things
                               concat_activations:bool=True,
                               save_to_disc:bool=True,
                               save_dir:Optional[str]=None,
                               input_name:Optional[str]=None,
                               layer_name_func:Optional[callable]=None,
                               output_dir:Optional[Union[str, Path]]=None,
                               output_device=None):
    """ function to get model activations at the specified layers for the given
    input data/queries
    
    :param input : dict of tokens   input['input_ids'] : tensor with 
                                    shape (batch_size, {max}_seq_len)
    :param layers : list[str]   should be a list of layer names/ 
                                attribute paths
                                each should be a torch.nn.Module
    :param model :   the model to do inference on
    :param outputs : bool   whether to return the outputs of the model 
    :param states : bool    whether to return the hidden states (output 
                            from embedding layer + all attention layers)
    :param attention : bool whether to return the attention scores
    :param num_return_sequences : int   number of sequences to return 
                                        for each input sequence. ie 
                                        replicates. Default is 1
    :param max_length : int     maximum length of the output sequence. 
                                Default is 512

    :param concat_activations : bool    whether to concatenate the input 
                                        and generation activations in 
                                        one tensor. Default is True
    :param save_to_disc : bool      whether to save the activation 
                                    tensors. Default is True
    :param save_path : str|None     (optional) path to save activation 
                                    tensors 
    :param input_name : str|None    (optional) an identifier for the 
                                    input queries/strings for use when 
                                    constructing the filenames for saved 
                                    activations
    :param layer_name_func : callable:None      (optional) function to 
                                                rename layers when 
                                                writing the files
    :param output_dir : str|Path|None   (optional) path to save output 
                                        tokens and description


    :return: dict from model.generate 
    """
    # check if saving 
    if save_to_disc:
        if save_dir is None:
            # WISHLIST format like other errors
            raise ValueError(textwrap.fill(textwrap.dedent("""save_path must be 
                                        specified if save_to_disc is True""")))
        else:
            save_dir = Path(save_dir)
            if not save_dir.exists():
                print(f"Creating directory {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
        
        if return_outputs and output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                print(f"Creating directory {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
    model.eval()

    saved_activations = dict()
        
    def get_activations(name):
        def hook(self, inputs, output):
            # saved_activations[name].append(inputs[0].detach())
            saved_activations[name].append(output.detach())
        return hook

    # make the hooks
    for ell in layers:
        saved_activations[ell] = []
        lay = get_nested_attr(model, ell)
        lay.register_forward_hook(get_activations(ell))

    # run inference
    with torch.no_grad():
        outputs = model.generate(inputs=input['input_ids'].to(model.device),
                        attention_mask=input['attention_mask'].to(model.device),
                        return_dict_in_generate=True,
                        max_length=max_length,
                        output_hidden_states=states,
                        output_attentions=attention,
                        num_return_sequences=num_return_sequences)

    # clear the hooks
    for ell in layers:
        ell = get_nested_attr(model, ell)
        ell._forward_hooks.clear()

    # print(saved_activations)

    # process the activations
    # TODO add option to not
    saved_activations = restructure_activation_dict(saved_activations, 
                                            n_replicates=num_return_sequences,
                                            concat_all=concat_activations)
    
    # saving
    if save_to_disc:
        save_activations_to_disc(activations=saved_activations,
                                 save_dir=save_dir,
                                 input_name=input_name,
                                 layer_name_function=layer_name_func)
        
        if return_outputs and not check_if_null(output_dir, False):
            filename = check_if_null(input_name, "OUTPUTS") + ".pkl"
            output_filename = os.path.join(output_dir, filename)

            # move outputs if necessary
            if check_if_null(output_device, False, True):
                outputs = move_output_tensors(outputs, output_device)

            # WISHLIST save hidden states and attentions to a different place
            # (will require restructuring them somehow, want safetensors also)

            print(f"saving output sequences to {output_filename}")
            with open(output_filename, "wb") as file:
                pickle.dump(outputs, file)
                
    # configure output
    # outdict = {
    #     "layers" : layers,
    #     "activations" : saved_activations,
    # }
    # if return_outputs:
    #     outdict.update(outputs)
    
    # print(outputs)
    if return_outputs:
        return outputs
    else:
        return


def restructure_activation_dict(saved_activations:dict, n_replicates:int,
                                concat_all:bool=True
                                # max_length:int
                                ):
    """
    Averages and restructures the outputs of a saved activation dictionary for
    further analysis (and packages them nicely to be saved as safetensors)

    :param saved_activations : dict [ layer_name -> list(activations) ]
    :param n_replicates : int   number of replicates for inputs
    :param max_length : int   maximum length of the input sequence
    :param concat_all : bool   whether to concatenate all activations into one
                                tensor. If False, the activations are returned
                                as a tuple of tensors corrensponding to the 
                                input and the generated sequence.

    activations are a list of max_length specified at generation
        1st entry: activations for the input string
            ( n_replicates x batch_size, input_length , hidden_size)
        2nd - (max_length - 1)th entries: activations during generation
            ( n_replicates x batch_size , 1 , hidden_size)


    :return restructured: dict [ layer_name -> tuple(activations) ]

    where each activation is now a tensor of shape
        1st entry : activations for the input string
            (batch_size, input_length, hidden_size)
        2nd entry : activations for generation
            (batch_size, max_length - input_length - 1, hidden_size)

    If concat_all, then the activation is a tensor of shape
            (batch_size, max_length - 1, hidden_size)
        
    """

    restructured = {}

    for layer_name, activations in saved_activations.items():
        
        nrb, inlen, hidden_size = activations[0].shape 
        batch_size = nrb // n_replicates # TODO do I need this?

        # create stacks for activation and generation
        input_activations = activations[0]  
            # ( n_replicates x batch_size, input_length , hidden_size)

        generation_activations = torch.cat(activations[1:], dim=1)
            # ( n_replicates x batch_size, generated_length, hidden_size)

        # average each over the replicates
        idx_list = list(range(0, nrb, n_replicates))

        def average_over_replicates(acts:torch.Tensor):
            avged = torch.stack([torch.mean(acts[k:k+n_replicates], dim=0) \
                                for k in idx_list])
            return avged
        
        # get the averages
        input_avg = average_over_replicates(input_activations)
        gener_avg = average_over_replicates(generation_activations)

        # set the new value
        # input_restructured[layer_name] = input_avg
        # generation_restructured[layer_name] = gener_avg
        if concat_all:
            # concatenate the input and generation activations
            restructured[layer_name] = torch.cat((input_avg, gener_avg), dim=1)
                                        # (batch_size, max_length, hidden_size)
        else:
            restructured[layer_name] = (input_avg, gener_avg)
                        # (batch_size, input_length, hidden_size),
                        # (batch_size, max_length - input_length, hidden_size)

    return restructured


def save_activations_to_disc(activations:dict, save_dir:str,
                            input_name:Optional[str]=None,
                            layer_name_function:Optional[callable]=None):
    """
    Saves the activations to disc in a structured way. Should be formatted as
    { layer_name -> activations } where the activations are either a tensor
    or a tuple of tensors (input_activations, generation_activations).


    :param activations : dict [ layer_name -> activations ]
    :param save_path : str   path to the directory where the activations should
                            be saved
    :param input_name : str|None   name of the input data/dataset
    :param layer_name_function : function|None
                                function to rename the layer names to a 
                                standard format. If None, the layers will not
                                be renamed. Default is None.

    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # check if the activations are in the correct format
    if not isinstance(activations, dict):
        raise ValueError(textwrap.fill(textwrap.dedent("""activations must be a 
                                dictionary of layer names to activations""")))
    if not all(isinstance(v, (torch.Tensor, tuple)) for v in activations.values()):
        raise ValueError(textwrap.fill(textwrap.dedent("""
                        activations must be a dictionary of layer names to 
                        activations, where each activation is a tensor or a 
                        tuple of tensors
                    """)))
    
    # check if input_name is provided
    input_name = check_if_null(input_name, "")

    # save the activations
    for layer_name, act in activations.items():

        # rename the layer name if a function is provided
        if layer_name_function is not None:
            layer_name = layer_name_function(layer_name)

        # get the filename and filepath
        filename = f"{layer_name}___{input_name}.safetensors" if \
                           input_name != "" else f"{layer_name}.safetensors"
        filepath = os.path.join(save_dir, filename)

        # if in format of a tuple, save a dictionary of tensors
        if isinstance(act, tuple):
            assert len(act) == 2, "activations must be a tuple of length 2"
            to_save = {
                "input_activations": act[0], 
                "generation_activations": act[1]
            }
        else:

            if not isinstance(act, torch.Tensor):
                raise TypeError(textwrap.fill(textwrap.dedent("""
                    Single activation should be a torch.Tensor.
                    You might be trying to save the wrong thing?
                    """)))
            
            to_save = {
                "activations": act
            }
            
        # save the tensor(s) to disc
        save_file(to_save, filepath)

    return


def move_output_tensors(output_dict:dict, device:torch.device):
# TODO add description
    """moves output tensors from a model to a different device
    
    """

    possible_keys = ["sequences", 
                     "attentions", 
                     "hidden_states", 
                     "past_key_values",

                     # for use with inference_with_activations
                     "activations", 
                     "layers"
                     ]
    # get the applicable keys
    keys_to_use = list(set(output_dict.keys()) & set(possible_keys))
    
    new_dict = {}

    for key in keys_to_use:
        value = output_dict[key]

        if key == "sequences":
            new_dict[key] = value.to(device)
        
        elif key == "layers": # each of these is a string
            new_dict[key] = value
        
        elif key == 'activations': # dictionary layer_name -> list(tensor) 
            act_dict = {}
            for k, v in value.items():
                act_dict[k] = [x.to(device) for x in v]

            new_dict[key] = act_dict

        else:
            # `max_seq_len` entries, each a tuple with `n_layers` tensors
            new_value = []
            for entry in value:
                new_value.append([x.to(device) for x in entry])
            new_dict[key] = tuple(new_value)

    print(new_dict.keys())
    return new_dict


def move_to_device(x, device:Union[str, torch.device]):
    """
    Moves a tensor or a list of tensors to the specified device.
    
    :param x: object with tensors to move
    :param device: Device to move the tensor(s) to.
    
    :return: Tensor or list of tensors on the specified device.
    """
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [t.to(device) for t in x]
    elif isinstance(x, tuple):
        return tuple(t.to(device) for t in x)
    else:
        return x.to(device)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                        Outputs of Inference Processing            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
DEFAULT_EMBED = "nomic-ai/nomic-embed-text-v2-moe"
def embed_strings(strings:list, embed_model:str=DEFAULT_EMBED, 
                  prompt_name:str="passage", truncate:Optional[int]=None):
    """
    embeds strings using sentence_transformers
    
    :param strings: list of strings to embed
    :type strings: list
    :param embed_model: the model to use for embedding
    :type embed_model: str
    :param prompt_name: either "query" or "passage" if using Nomic. 
                        Informs the embedding
    :type prompt_name: str
    :param truncate: truncate embedding to the given dimension. If None, 
                     does not truncate (default: None)
    :type truncate: Union[bool, int]
    """
    embedder = SentenceTransformer(embed_model, trust_remote_code=True,
                                   truncate_dim=truncate)
    embeddings = embedder(strings, prompt_name=prompt_name)

    return embeddings # ( len(strings), min{768, truncate} )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                           Looking at Activations            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class Activations:

    """
    Class to handle activations loading and some other things
    """