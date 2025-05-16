import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file

from src.utils import check_if_null, get_device

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#               Accessing HuggingFace models and performing analysis   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 
#   description of the file here
#

GLOBAL_ANALYSIS_DIR = "~/scratchcpriebe1/MO/network_manifolds/results/cache"


class LoraAnalysis:

    """
    Class to wrap a fine-tuned model for analysis
    """

    def __init__(self, base_model_path:str, peft_path:str, 
                #  peft_local:bool=True, 
                 build_model:bool=True, 
                 n_layers:Optional[int]=None, 
                 custom_name:Optional[str]=None):
        """
        param: base_model_path : str    path/name/id of the base model
        param: peft_path : str          path/name/id of the fine-tuned model
        param: build_model : bool       whether to build the fine-tuned model.
                                        This is necessary for most analyses.
                                        (default: True)

        """

        self.device = get_device()
        self.base_model_path = base_model_path
        
        # if peft_local: 
        #     peft_path = os.path.abspath(peft_path)
        self.peft_path = peft_path

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, 
                                                       use_fast=True)
                                                       # TODO add option to not
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if build_model:
            self.model = self.get_lora_model().to(self.device)
        
        # set default weight naming function 
        self.weight_name_function = get_weight_name_function(base_model_path)

        
        # caching
        # TODO better caching functionality
        # TODO add disc caching (eventually)
        # cache_path = make_cache_dir_name(base_model_path, peft_path,
        #                                  custom_name=custom_name)
        
        # ## check if path exists
        # if not os.path.exists(cache_path):
        #     os.mkdir(cache_path)
        # self.cache_file_path = cache_path

        self.cache = {} # cache key -> cached value

    
    def clear_cache(self):
        """clear cache"""
        self.cache.clear()

    def to(self, device):
        self.model.to(device)



    def get_lora_model(self):
        base_model = AutoModelForCausalLM.\
                        from_pretrained(self.base_model_path, 
                                        torch_dtype="auto", device_map="auto")
        return PeftModel.from_pretrained(base_model, self.peft_path)

    
    def get_lora_weights(self, weight_name_function:Optional[callable],
                         alternate_tensor_name:Optional[str]=None,
                         cache=True):
        """
        Load the LoRA weight matrices from file 

        :param weight_name_function : function|None
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

        if weight_name_function is not None:
            wnf = weight_name_function
        elif self.weight_name_function is not None:
            wnf = self.weight_name_function
        else:
            wnf = get_weight_name_function("DEFAULT")
        
        # get path to weight tensors
        if alternate_tensor_name is not None:
            tensor_filename = alternate_tensor_name
        else:
            tensor_filename = "adapter_model.safetensors"
        tensor_path = os.path.join(self.peft_path, tensor_filename)

        # load the weight tensors and make dict
        tensor_dict = {}
        with safe_open(tensor_path, framework="pt") as file: 
            # TODO add option to change device
            for k in file.keys():
                key = wnf(k)
                tensor_dict[key] = file.get_tensors(k) 
        
        if cache:
            self.cache[cache_key] = tensor_dict

        return tensor_dict
    

    def get_lora_layer_attribute_names(self, 
                                       alternate_tensor_name:Optional[str]=None,
                                       cache:bool=True):
        """get the attribute names for each of the LoRA matrices
        useful for registering forward hooks
        """
        cache_key = "lora_names"

        if cache and cache_key in self.cache.keys():
            return self.cache[cache_key]

        if alternate_tensor_name is not None:
            tensor_filename = alternate_tensor_name
        else:
            tensor_filename = "adapter_model.safetensors"
        tensor_path = os.path.join(self.peft_path, tensor_filename)

        with safe_open(tensor_path, framework="pt") as file: 
            layer_names = file.keys()

        if cache:
            self.cache[cache_key] = layer_names

        return layer_names


    @property
    def lora_weights(self, weight_name_function:Optional[callable]=None, 
                     alternate_tensor_name:Optional[str]=None, cache:bool=True):
        """property to get lora weights"""
        return self.get_lora_weights(weight_name_function=weight_name_function,
                                alternate_tensor_name=alternate_tensor_name,
                                cache=cache)
    
    @property
    def lora_layer_attributes(self, alternate_tensor_name:Optional[str]=None, 
                              cache:bool=True):
        """property to get lora weight attribute names"""
        return self.get_lora_layer_attribute_names(
                                alternate_tensor_name=alternate_tensor_name,
                                cache=cache)


    def inference(self, inputs:dict, states:bool=False, attention:bool=False):
        """
        Do inference (does not collect activations). Inputs should be
        tokenized already.

        TODO add disc saving functionality
        TODO add support for max_length option
        """
        model = self.model

        model.eval()

        with torch.no_grad():
            outputs = model.generate(inputs=inputs['input_ids'].to(self.device),
                                     attention_mask=inputs['attention_mask']\
                                        .to(self.device), 
                                    max_length=512,
                                    return_dict_in_generate=True,
                                    output_hidden_states=states,
                                    output_attentions=attention)

        return outputs
    

    def get_activations(self, inputs:dict, layers:list[str], 
                        return_outputs:bool=True, states:bool=True, 
                        attention:bool=True):
        """
        Do inference and collect activations. 
        Inputs should be a tokenized dictionary.
        Layers should be in the form of a path. 

        TODO add option/way to specify which layers by number, projection, etc
        TODO add max_length option
        """

        outputs = inference_with_activations(
            input=inputs,
            layers=layers,
            model=self.model,
            return_outputs=return_outputs,
            states=states,
            attention=attention
        )

        return outputs


    # def pipe_inference(self, inputs:dict, s)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                           Helpers for LoraAnalysis            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# functions that parse the names of the .safetensor weight tensors
def llama_wnf(k:str):
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

    name = f"{padded_number}_{proj_type}-prj_{lora_id}"
    return name


weight_name_func_dict = {
    "DEFAULT" : llama_wnf,      # TODO create/add default weight name func
    "meta-llama" : llama_wnf,
}


def get_weight_name_function(base_model_path:str):
    """
    finds weight name parsing function for loading model LoRA weights from
    huggingface .safetensors files
    """

    # check if default 
    if base_model_path == "DEFAULT":
        return weight_name_func_dict["DEFAULT"]

    matches = [word for word in weight_name_func_dict.keys() \
               if word in base_model_path]
    
    assert len(matches) <= 1, "multiple model names matched"

    if len(matches) > 0:
        m = matches[0]
        print(f"using predefined weight name parsing function for {m}")
        return weight_name_func_dict[m]
    else:
        print("No predefined function exists."\
              "DEFAULT will be used. Check that DEFAULT is suitable for the " \
              "architecture of the model. Enter custom weight name parsing " \
              "function if necessary where applicable.")
        return
    

def make_cache_dir_name(base_model_path:str, peft_path:str, 
                        custom_name:Optional[str]=None):
    """
    returns the path to the analysis cache directory
    """

    rules = [ # (compile/pattern, replacement)
        (re.compile(r'^.*?results/'), ""),   # remove path to results dir
        (re.compile(r'/'), "__")           # replace "/" for names
    ]

    names = []
    for s in [base_model_path, peft_path]:
        print(s)
        for pat, rep in rules:
            s = pat.sub(rep, s)
            print(s)
        names.append(s)
    names = "___".join(names)

    custom_name = check_if_null(custom_name, "")
    pathname = os.path.join(GLOBAL_ANALYSIS_DIR, names, custom_name)

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
                               return_outputs:bool=True, states:bool=True,
                               attention:bool=True):
    """ function to get model activations at the specified layers for the given
    input data/queries
    
    :param input : dict of tokens   input['input_ids'] : tensor with shape
                                    (batch_size, {max}_seq_len)
    :param layers : list[str]   should be a list of layer names/attribute paths
                                each should be a torch.nn.Module
    :param model :   the model to do inference on

    :param outputs : bool   whether to return the outputs of the model also
    :param states : bool    whether to return the hidden states (output from
                            embedding layer + all attention layers)
    :param attention : bool whether to return the attention scores

    :return: dict
    """
    model.eval()

    saved_activations = dict()
        
    def get_activations(name):
        def hook(self, inputs, output):
            saved_activations[name].append(inputs[0].detach())
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
                        max_length=512,
                        output_hidden_states=states,
                        output_attentions=attention)

    # clear the hooks
    for ell in layers:
        ell = get_nested_attr(model, ell)
        ell._forward_hooks.clear()
    
    # configure output
    outdict = {
        "layers" : layers,
        "activations" : saved_activations,
    }
    if return_outputs:
        outdict.update(outputs)
    
    print(outdict.keys())

    return outdict




def move_output_tensors(output_dict:dict, device:torch.device):
    """moves output tensors from a model to a different device
    
    TODO add description
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