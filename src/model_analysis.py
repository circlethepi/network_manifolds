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


    def inference(self, inputs:dict, states:bool=True, attention:bool=True):
        """
        Do inference/access activations in response to inputs. Inputs should be
        tokenized already.

        TODO add disc saving functionality
        """
        model = self.model

        model.eval()

        with torch.no_grad():
            outputs = model.generate(inputs=inputs['input_ids'].to(self.device),
                                     attention_mask=inputs['attention_mask']\
                                        .to(self.device), 
                                    max_new_tokens=512,
                                    return_dict_in_generate=True,
                                    output_hidden_states=states,
                                    output_attentions=attention)

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
    padded_number = f'{int(layer_number):0{1}d}'

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