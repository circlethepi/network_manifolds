#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                          Orthogonality Investigations   
#                          Merrick Ohata 2026, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
Orthog 
#Todo flesh out
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import torch




class SynthModel():

    def __init__(self, rank, gen="bernoulli"):
        self.r = rank
        
        # generate latent type
        if gen == "bernoulli":
            self.latent = torch.bernoulli(torch.ones(self.r) * 0.5)
        elif gen == "uniform":
            self.latent = torch.rand.uniform(0, 1, (self.r,))
        else:
            raise ValueError("Unsupported generation type")