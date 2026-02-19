#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                    Working in the Data Kernel Perspective Space 
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
  Code related to inducing the DKPS and creating visualizations of models
  within that space
"""

import warnings
import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union
import glob

from safetensors import safe_open

from src.utils import check_if_null, display_message, GLOBAL_PROJECT_DIR
import src.matrix as matrix
import src.plot as plot
import src.model_analysis as analysis

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
# WISHLIST add handling to save to a file


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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                    Generating DKPS from tensor files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
 
def find_result_files(exp_name:Union[str, list],
               base_name:Union[str, list],
               seed:int=0,
               peft_name:Optional[Union[str, list]]=None,
               custom_name:Optional[str]=None,
               result_type:str="embeds",
               queries:Optional[int]=None,
               replicates:Optional[int]=None,
               decoded:bool=True):
    """
    Finds cached inference files according to the search criteria. 
    Returns paths as a list. 

    Experiment script saves results in the diretory 
        {model_cache_dir}/ result_type 
    with filename
        {exp_name}_r{replicates}_q{queries}_seed{seed}
    extension
        .safetensors if not decoded or embeds
        .pkl if decoded output

    :param exp_name: Union[str, list(str)]    name(s) of the experiments
    :param base_name: Union[str, list(str)]   --base_model(s)
    :param seed: int        --seed
    :param peft_name: Optional[Union[str, list(str)]]    --peft_model(s)
    :param custom_name: Optional[str]   --model_name
    :param result_type: Optional[str]   "embeds" or "outputs". 
                                        Default: embeds
    :param queries: Optional[int]       number of queries
    :param replicates: Optional[int]    number of replicates
    :param decoded: bool    whether the output is decoded. Only relevant
                            if result_type="outputs"
    """
    # TODO add dataset options
    # get results dir
    if result_type not in ("embeds", "outputs"):
        msg = f"""`result_type` must be either 'embeds' or 'outputs' but
                got {result_type} instead"""
        raise ValueError(display_message(msg))
    
    # Normalize inputs to lists
    if isinstance(exp_name, str):
        exp_names = [exp_name]
    else:
        exp_names = exp_name
    
    if isinstance(base_name, str):
        base_names = [base_name]
    else:
        base_names = base_name
    
    if peft_name is None:
        peft_names = [None]
    elif isinstance(peft_name, str):
        peft_names = [peft_name]
    else:
        peft_names = peft_name

    ## seed
    seed_pattern = f"_seed{seed}"
    if result_type == "embeds" or not decoded:
        extension = ".safetensors"
    else:  # result_type == "outputs" and decoded
        extension = ".pkl"
    # Collect all matching files
    all_matches = []

    # Iterate over all combinations of base_name and peft_name
    for base in base_names:
        for peft in peft_names:
            base_cache = analysis.make_cache_dir_name(base, peft, custom_name)
            main_dir = os.path.join(base_cache, result_type)
            
            # Iterate over experiment names
            for exp in exp_names:
                pattern_parts = [exp]
                ## replicates
                if replicates is not None:
                    pattern_parts.append(f"_r{replicates}")
                else:
                    pattern_parts.append("_r*")
                ## queries
                if queries is not None:
                    pattern_parts.append(f"_q{queries}")
                else:
                    pattern_parts.append("_q*")
                
                pattern_parts.append(seed_pattern)
                
                filename_pattern = "".join(pattern_parts) + extension
                full_pattern = os.path.join(main_dir, filename_pattern)
                
                # Find matching files for this exp_name
                matching_files = glob.glob(full_pattern)
                all_matches.extend(matching_files)

    return all_matches


def _aggregate_avg(tensor: torch.Tensor, replicates: Optional[int] = None) -> torch.Tensor:
    """
    Average aggregate a tensor over the first dimension.
    
    :param tensor: torch.Tensor of shape (r, d)
    :param replicates: Optional[int] number of replicates to average over.
                       If None, averages over all.
    :return: torch.Tensor of shape (1, d)
    """
    if replicates is not None:
        # Average over first `replicates` entries
        return tensor[:replicates, :].mean(dim=0, keepdim=True)
    else:
        # Average over all replicates
        return tensor.mean(dim=0, keepdim=True)


def results_from_data_ids(filenames: list, 
                          data_ids: list, 
                          aggregate: Optional[str] = "avg",
                          replicates: Optional[int] = None,
                          key_override: str = "drop_key"):
    """
    Load and optionally aggregate tensors from safetensors files based on data IDs.
    
    :param filenames: list(str)     list of filenames to check/load
    :param data_ids:  list(any)     list of ids from dataset to use as keys
    :param aggregate: Optional[str] aggregation method ("avg" or None)
    :param replicates: Optional[int] number of replicates to aggregate over
    :param key_override: str        "strict", "drop_key", or "drop_file"
    :return: torch.Tensor of shape (len(filenames), len(data_ids) * agg_size, d)
             or list of tensors if stacking is not possible
    """
    if key_override not in ("strict", "drop_key", "drop_file"):
        msg = f"key_override must be 'strict', 'drop_key', or \
                'drop_file', got {key_override}"
        raise ValueError(display_message(msg))
    
    if aggregate is not None and aggregate != "avg":
        msg = f"Only 'avg' aggregation is currently supported, got \
            {aggregate}"
        raise ValueError(display_message(msg))
    
    # Step 1: Check all files are safetensors and validate keys
    valid_files = []
    valid_keys = set(str(k) for k in data_ids)  # Convert all keys to strings
    files_to_drop = []
    keys_to_drop = set()
    missing_info = {}  # Track which keys are missing from which files
    
    for filepath in filenames:
        # Check file extension
        if not filepath.endswith('.safetensors'):
            if key_override == "strict":
                msg = f"File {filepath} is not a safetensors file"
                raise ValueError(display_message(msg))
            elif key_override == "drop_file":
                files_to_drop.append(filepath)
                continue
            else:  # drop_key - doesn't make sense here, treat as drop_file
                files_to_drop.append(filepath)
                continue
        
        # Check keys exist in file
        try:
            with safe_open(filepath, framework="pt") as f:
                file_keys = set(f.keys())
                missing_keys = valid_keys - file_keys
                
                if missing_keys:
                    missing_info[filepath] = missing_keys
                    
                    if key_override == "strict":
                        msg = f"File {filepath} is missing keys: {missing_keys}"
                        raise KeyError(display_message(msg))
                    elif key_override == "drop_file":
                        files_to_drop.append(filepath)
                        continue
                    elif key_override == "drop_key":
                        keys_to_drop.update(missing_keys)
                
                valid_files.append(filepath)
                
        except Exception as e:
            if key_override == "strict":
                msg = f"Error reading file {filepath}: {e}"
                raise RuntimeError(display_message(msg))
            else:
                files_to_drop.append(filepath)
                continue
    
    # Issue warnings and update lists
    if files_to_drop:
        msg = f"Dropped {len(files_to_drop)} file(s): {files_to_drop}"
        warnings.warn(display_message(msg))
    
    if keys_to_drop:
        msg = f"Dropped {len(keys_to_drop)} key(s): {keys_to_drop}"
        warnings.warn(display_message(msg))
    
    if not valid_files:
        msg = "No valid files remaining after filtering"
        raise ValueError(display_message(msg))
    
    if not valid_keys:
        msg = "No valid keys remaining after filtering"
        raise ValueError(display_message(msg))
    
    # Update data_ids to only include valid keys (maintain order)
    final_data_ids = [str(k) for k in data_ids if str(k) in valid_keys]
    
    # Step 2: Load and aggregate tensors
    all_results = []
    
    for filepath in valid_files:
        file_tensors = []
        
        with safe_open(filepath, framework="pt") as f:
            for key in final_data_ids:
                # Load tensor for this key
                tensor = f.get_tensor(key)  # (r, d)
                # WISHLIST additional aggration options
                if aggregate == "avg":
                    tensor = _aggregate_avg(tensor, replicates)
                # else: keep tensor as is (r, d)
                
                file_tensors.append(tensor)
        
        concatenated = torch.cat(file_tensors, dim=0) # (len(final_data_ids) * agg_size, d)
        all_results.append(concatenated)
    
    try:
        # Try to stack into single tensor: 
        result = torch.stack(all_results, dim=0)
            # (len(valid_files), len(final_data_ids) * agg_size, d)
        return result
    except RuntimeError:
        # If shapes don't match, return as list
        msg = "Could not stack results into single tensor, returning list"
        warnings.warn(display_message(msg))
        return all_results


def induce_dkps(reps:Union[list, torch.Tensor],
                space_name:str,
                similarity:str="fro",
                is_cov:bool=False,
                get_coordinates:bool=False,
                mds_dim:int=2,
                mdskwargs:Optional[dict]=None):
    """
    Calculates similarity matrix and optionally induces DKPS coordinates.

    Assumes no alignment.
    
    :param reps: Union[list, torch.Tensor]  list or tensor of representations
    :param space_name: str                  name of space for saving results
    :param similarity: str                  similarity type ("fro" or "bw")
    :param is_cov: bool                     whether reps are covariance matrices
    :param get_coordinates: bool            whether to compute MDS coordinates
    :param mds_dim: int                     dimension for MDS embedding
    :param mdskwargs: Optional[dict]        additional kwargs for compute_MDS
    :return: coords if get_coords else sim_mat
    """
    from safetensors.torch import save_file
    import torch
    import os
    
    # Convert to list if tensor
    reps = list(reps) if isinstance(reps, torch.Tensor) else reps
    
    # Check that all entries have the same shape
    shapes = [r.shape if isinstance(r, torch.Tensor) else r.shape for r in \
                                                                        reps]
    if len(set(shapes)) > 1:
        msg = f"All representations must have the same shape, got shapes: \
            {shapes}"
        raise ValueError(display_message(msg))
    
    # Convert to Matrix objects
    if not is_cov:
        # If not covariance matrices, compute them
        reps = [matrix.Matrix(torch.tensor(m) @ torch.tensor(m).T) for m in reps]
    else:
        reps = [matrix.Matrix(m) for m in reps]
    
    # Get similarity matrix and save
    sim_mat = matrix.matrix_similarity_matrix(
        *reps,
        sim_type=similarity,
        aligned=False,
        to_numpy=True,  # compute_MDS expects numpy array
        save_results=True,
        space_name=space_name
    )
    
    coords = None
    if get_coordinates:
        # Set up MDS kwargs
        if mdskwargs is None:
            mdskwargs = {}
        
        # Ensure n_components is set
        mdskwargs.setdefault('n_components', mds_dim)
        
        coords = compute_MDS(sim_mat, **mdskwargs)
        
        save_dir = matrix.coord_savedir(space_name)
        os.makedirs(save_dir, exist_ok=True)
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        coord_save_dict = {
            f"{mds_dim}": coords_tensor
        }
        coord_path = os.path.join(save_dir, "coordinates.safetensors")
        save_file(coord_save_dict, coord_path)
    
    if get_coordinates:
        return coords
    else:
        return sim_mat


