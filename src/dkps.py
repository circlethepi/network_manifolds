#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                    Working in the Data Kernel Perspective Space 
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
  Code related to inducing the DKPS and creating visualizations of models
  within that space
"""

import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union

from safetensors import safe_open

from src.utils import check_if_null, display_message
import src.matrix as matrix
import src.plot as plot

from sklearn import manifold

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                           Similarity to Coordinates           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region


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

    def __init__(self, 
                 result_file:Optional[str]=None,
                 similarity_matrix=None, seed:int=0,
                 coordinates=None, name:Optional[str]=None,
                 sub_names=None, splits=None):
        """
        
        :param result_file: str|None    safetensors filepath with at least one 
                                        of the similarity matrix or MDS
                                        coordinates under the corresponding
                                        keys
        :param similarity_matrix: array-like  The similarity matrix that 
                                                generates the space we work in.
                                                Should be (N, N), symmetric,
                                                and have 1s on the diagonal
        :param seed: int        seed for MDS calculation
        :param coordinates np.array|None    pre-computed MDS coordinates

        
        If a quantity (similarity matrix or coordinates) and a results file is
        provided also containing that quantity, the provided value will be used 
        instead. 
        """

        self.file = result_file
        
        if check_if_null(result_file):
            store = {}
            with safe_open(result_file, framework='pt') as f:
                for k in f.keys():
                    store[k] = f.get_tensor(k)
            
            if 'coordinates' in store:
                coordinates = check_if_null(coordinates, store['coordinates'])
            if 'similarity_matrix' in store:
                similarity_matrix = check_if_null(similarity_matrix, 
                                                  store['similarity_matrix'])

        self.similarity_matrix = similarity_matrix
        self.seed = seed 
        self.splits = splits

        # names and stuff
        self.name = name
        self.sub_names = sub_names

        # coordinates
        self.coordinates = coordinates
        
        # number of elements
        if self.similarity_matrix is not None:
            self.n = self.similarity_matrix.shape[-1]
        elif self.coordinates is not None:
            self.n = self.coordinates.shape[0]
        else:
            self.n = None
    

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
    

    def set_name(self, name:str, override=False):
        """set name"""
        do = True
        if check_if_null(self.name, False, True):
            if not override:
                message = f"""Warning: current name is set to {self.name}. To 
                override this warning and change the name to {name}, re-run 
                this method with `override=True`"""

                do = False
            else:
                message = f"""Overriding current name {self.name}"""

                do = True
            print(display_message(message))
        
        if do:
            self.name = name

        return self   


    def set_sub_names(self, sub_names, override=False):
        """set names for each object represented"""
        do = True

        if check_if_null(self.sub_names, False, True):
            if not override:
                message = f"""Warning: current sub_names are set to 
                {self.sub_names}. To override this warning and change them to 
                {sub_names}, re-run this method with `override=True`"""

                do = False
            else:
                message = f"""Overriding current sub_names {self.sub_names}"""

                do = True
            print(display_message(message))
        
        if do:
            # TODO better error handling (allow more than n)
            message = """Number of names must match the number of objects"""
            assert len(sub_names) == self.n, display_message(message)

            self.sub_names = sub_names
        return self      
    

    def plot_similarity(self, title=None,
                        axis_label=None, ticks=None, ticklabs=None,
                        vmin=0, vmax=1, rotation=0,
                        cbar_label="similarity",
                        cbar_ticks=None,
                        figsize=(10, 10), savename=None,
                        savedir_override=None,
                        mask_color='#cffcff', nan_color='#ffaffa',
                        colormap=None,
                        full_matrix=False):
        """Plot similarity matrix with custom plotting function 
        (see plot.py for more details)"""

        message = "No similarity matrix defined"
        assert self.similarity_matrix is not None, display_message(message)

        title = check_if_null(check_if_null(title, self.name), "similarity")
        ticks = check_if_null(ticks, range(self.n))

        ticklabs = check_if_null(check_if_null(ticklabs, self.sub_names),
                                 ["" for k in range(self.n)])
        
        plot.plot_similarity_matrix(sims=self.similarity_matrix,
                                    splits = self.splits,
                                    title=title, axis_label=axis_label,
                                    ticks=ticks, ticklabs=ticklabs,
                                    vmin=vmin, vmax=vmax, rotation=rotation,
                                    cbar_label=cbar_label, 
                                    cbar_ticks=cbar_ticks,
                                    figsize=figsize, savename=savename,
                                    savedir_override=savedir_override,
                                    mask_color=mask_color, nan_color=nan_color,
                                    colormap=colormap,
                                    full_matrix=full_matrix)


    def group_coordinates(self, splits, indices=False, name=None):
        """Groups coordinates into subgroups
        
        :param splits: dict|array-like(int)     where to split the coordinates
        
        integer or array-like collection of integers
        - the number of elements in all groups

        dictionary
        - [name of group] -> [size of group: int]

        :param indices: bool    whether to instead treat split values as
                                indices instead of counts. Default: False
        :param name:    list of names for the groups        
        """
        # TODO add error checks
        
        all_coords = self.compute_coords()

        if isinstance(dict, splits):
            names, inds = dict_to_inds(splits)
        else:
            inds = splits if indices else counts_to_inds(splits)
    
        names = check_if_null(names, range(len(inds))) 

        coord_groups = split_coordinates(all_coords, inds)

        grouped_names = dict(zip(names, coord_groups))

        return grouped_names

# WISHLIST add handling for multiple sets of coordinates
# WISHLIST add handling to read from a file


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               DKPS Utilities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
        
def split_coordinates(coords, inds):
    """split coordinates by indices. Coords should be of shape (N, 2)"""

    if inds[-1] < coords.shape[0] - 1:
        inds.append(coords.shape[0] - 1)

    split_coords = []

    for h in range(len(inds)-1):
        lo, hi = inds[h], inds[h+1]
        split_coords.append(coords[lo:hi])
    
    return split_coords


def counts_to_inds(counts):
    """convert a list of counts into list of indices for splitting"""
    inds = []
    last = 0
    for i in counts:
        last = last + i
        inds.append(last)

    return inds


def dict_to_inds(splits, indices=False):
    """converts a dictionary into names and splits
    [name of group] -> [size of group: int]
    
    :param splits: dict     the names and splits
    :param indices: bool    whether the splits/sizes are indices
    """

    names = list(splits.keys())
    counts = list(splits.values())

    out = counts if indices else counts_to_inds(splits)

    return names, out


def coord_variance(coords:np.ndarray):
    """Calculate the mean and axes of variance for coordinates
    :param coords: np.ndarray   coordinates of shape (N, 2)"""

    mean = np.mean(coords, axis = 0)
    centered = coords - mean
    cov = (1/coords.shape[0]) * centered.T @ centered 

    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = np.flip(vals), np.flip(vecs)

    variance = vals[:2]
    directions = np.array([vecs[:, k] for k in (0, 1)])

    return mean, variance, directions
    