import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union

from src.utils import check_if_null, display_message
import src.matrices as linalg

from sklearn import manifold

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                    Working in the Data Kernel Perspective Space 
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 
#   Code related to inducing the DKPS and creating visualizations of models
#   within that space


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                           Similarity to Coordinates           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def compute_MDS(sim_matrix, n_components:int=2, zero_index:Optional[int]=None, 
                align_coords:bool=True, seed=0):
    """
    Computes MDS projection using a similarity matrix into k dimensions.

    :param sim_matrix: array-like   similarity matrix to use
    :param n_components: int        number of dimensions to reduce to. 
                                    Default 2
    :param zero_index: int|None     index of element to cetner the coordinates
                                    about. Default None
    :param align_corods: bool       whether to orient the coordinates to align
                                    the primary axes of the point cloud with 
                                    the standard axes. Default True
    :param seed: int                seed for MDS algorithm reproducibility. 
                                    Default 0. It is not recommended to change
                                    this

    """
    # convert to np array if needed (to use sklearn MDS, this is necessary)
    if not isinstance(sim_matrix, np.ndarray):
        sim_matrix = np.array(sim_matrix)

    # determine whether to do metric MDS 
    metric = True
    message = """Warning: non-metric MDS calculations are happening. This 
                 does technically work, but can often lead to some wonky plots.
                 Proceed with caution!"""    

    if np.any(np.isnan(sim_matrix)):
        metric = False
        display_message(message)

        sim_matrix = np.nan_to_num(sim_matrix, nan=1)
        # replace diagonal with small value since 0 is same as missing
        np.fill_diagonal(sim_matrix, 1+np.finfo(sim_matrix.dtype).eps)
    
    # convert to dissimilarity matrix
    dissim_matrix = np.ones(sim_matrix.shape) - sim_matrix

    # do MDS
    mds = manifold.MDS(n_components=n_components,
                       dissimilarity='precomputed',
                       eps=1e-16, max_iter=1000, n_init=100, random_state=seed,
                       metric=metric, normalized_stress=False)
    
    coords = mds.fit_transform(dissim_matrix)

    # align to cartesian axes
    if align_coords:
        coords = align_to_axes(coords, 
                               uncenter=check_if_null(zero_index, True, False))

    # center zero index
    if zero_index:
        coords -= coords[zero_index]
    
    return coords


def align_to_axes(coordinates:np.ndarray, uncenter=False):
    """Aligns coordinates to the cartesian axes"""
    mean = coordinates.mean(axis=0)
    centered = coordinates - mean

    U, S, Vh = np.linalg.svd(centered, full_matrices=False)
    axis_centered = centered @ Vh.T

    # restore original data center
    if uncenter:
        axis_centered += mean

    return axis_centered


def align_signs(*coordinates, ref_ind=0, debug=False):
    """Aligns the signs of multiple sets of coordinates.

    Coordinates should be of shape (N, K), where K is the number of plotting 
    dimensions (typically 2)
    """

    shape = coordinates[0].shape

    # assert all shapes are the same
    for coords in coordinates:
        assert coords.shape == shape, "All coordinates must be the same shape"
    # WISHLIST let them have different shapes

    coords_ref = coordinates[ref_ind]

    axes = []

    for coords_unr in coordinates:
        for i in range(shape[1]):
            # if the correlation of these columns is negative, flip the other one
            if np.corrcoef(coords_unr[:, i], coords_ref[:, i])[0, 1] < 0:
                coords_unr[:, i] *= -1
                axes.append(i)
    
    if debug:
        print(f"Flipped axes {axes}")

    return coordinates


class SimilaritySpace:
    """
    Class for dealing with a similarity space
    """

    def __init__(self, similarity_matrix, seed:int=0):
        """
        
        :param similarity_matrix: array-like  The similarity matrix that 
                                                generates the space we work in.
                                                Should be (N, N), symmetric,
                                                and have 1s on the diagonal
        :param seed: int        seed for MDS calculation

        """

        self.similarity_matrix = similarity_matrix
        self.seed = seed 
        self.n = similarity_matrix.shape[-1]

        # set later
        self.coordinates = None
    

    def compute_coords(self, align_coords:bool=True,
                       n_components:int=2,
                       zero_index:Optional[int]=None,
                       seed:Optional[int]=None, 
                       recompute:bool=False,
                       cache:bool=True):
        """Computes MDS projection using the given similarity matrix"""
        
        seed = check_if_null(seed, self.seed)

        if recompute or check_if_null(self.coordinates, True, False):
            coords = compute_MDS(self.similarity_matrix,
                                 n_components=n_components,
                                 zero_index=zero_index,
                                 align_coords=align_coords,
                                 seed=seed)
        
        if cache:
            self.coordinates = coords
        
        return coords
    
    
    

    
    