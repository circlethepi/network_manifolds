import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union


from src.utils import check_if_null, error_display


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                      Matrix and Linear Algebra Handling   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 
#   description of the file here
#


def get_backend(x):
    if isinstance(x, torch.Tensor):
        return torch
    else:
        return np

def transpose(x):
    return get_backend(x).swapaxes(x, -1, -2)
    

def decompose(mat, decomposition:Optional[str]="eigh", rank=None):
    """
    Decomposes the given matrix according to the given decomposition
    
    :param mat:     the matrix to decompose with shape (m, n)
    :param decomposition:str    which decomposition to use
    :param rank:                rank to cut off after

    :return decomposed:tuple    (values, left_vectors, right_vectors)
    """
    decomposition = decomposition.lower()

    backend = get_backend(mat)

    if decomposition == "svd":
        U, S, Vh = backend.linalg.svd(mat, full_matrices=False)
        # (m, r)   (r,)   (r, n)   where r = min(m, n)  ;  descending values
        # recomposes with U @ S @ Vh
        left_vectors, values, right_vectors = U, S, transpose(Vh)
    
    else:
        eigvals, eigvecs = backend.linalg.eigh(mat) if decomposition == "eigh"\
            else backend.linalg.eig(mat)
        # (r,)    (n, r)  note m = n  ; ascending values
    
        # put in descending order
        if decomposition == "eigh":
            if backend == torch:
                values, left_vectors = eigvals.flip(-1), eigvecs.flip(-1)
            else:
                values, left_vectors = eigvals[..., ::-1], eigvecs[..., ::-1]
            right_vectors = transpose(left_vectors)
        else:
            inds = backend.argsort(-values, axis=-1)
            take_along = torch.take_along_dim if backend == torch \
                         else np.take_along_axis
            values = take_along(eigvals, inds, axis=-1)
            left_vectors = take_along(eigvecs, inds[..., None, :], axis=-1)

            right_vectors = backend.linalg.inv(left_vectors)

    # clip rank
    if check_if_null(rank, False):
        values = values[...,:rank]
        left_vectors = left_vectors[..., :rank]
        right_vectors = right_vectors[..., :rank, :]

    return values, left_vectors, right_vectors



def recompose(values, left_vectors, right_vectors):
    """
    Recomposes matrix from factorization. Assumes same shapes as output from
    decompose

    :param values:  shape (r,)
    :param left_vectors:    shape (m, r)
    :param right_vectors:   shape (r, n)

    :return mat:    shape (m, n)  with rank r = min(m, n)
    """
    return left_vectors @ (values[..., :, None] * right_vectors)



class Matrix:
    """
    Class to handle linear algebra logic for decomposed matrices
    """

    def __init__(self, matrix:Optional[Union[torch.Tensor, np.array]]=None, 
                 decomposition:Optional[str]=None,
                 eigenvalues:Optional[Union[torch.Tensor, np.array]]=None,
                 eigenvectors:Optional[Union[torch.Tensor, np.array]]=None,
                 right_vectors:Optional[Union[torch.Tensor, np.array]]=None,
                 rank=float("inf")
                 ):
        
        self._matrix = matrix  # shape (m, n)

        # decomposition type
        if (check_if_null(decomposition, "NONE") == "NONE") and \
            (decomposition.lower() not in ("svd", "eig", "eigh")):
            message = """
            decomposition must be one of "svd", "eig", "eigh" or `None`. If it
            is `None`, then the default of "eigh" is used instead. 
            """
            raise ValueError(error_display(message))
        self.decomposition = check_if_null(decomposition.lower(), "eigh") 

        message = """
        rank cannot be `None`. Try initializating with the default rank instead
        """
        assert rank is not None, error_display(message)
        if check_if_null(matrix, False):
            rank = min(rank, *matrix.shape[-2:])
        if check_if_null(eigenvalues, False):
            rank = min(rank, eigenvalues.shape[-1])
        self.rank = rank

        self._values = eigenvalues # shape (rank,) 
        self._left_vectors = eigenvectors # shape (m, rank)
        self._right_vectors = right_vectors # shape (rank, n)

        if check_if_null(matrix, True) and check_if_null(eigenvectors, True):
            message = """
            At least one of either `matrix` or `eigenvectors` must be defined
            """
            raise ValueError(error_display(message))
        
    def decompose(self):
        """
        Decomposes the matrix according to the decomposition type and 
        updates corresponding properties 
        """
        if self._values is None:
            self._values, self._left_vectors, self._right_vectors = \
            decompose(self._matrix, decomposition=self.decomposition, 
                        rank=self.rank)
            self.rank = self._values.shape([-1])
        return self
    
    def recompose(self):
        """
        Recomposes the matrix and updates corresponding properties
        """
        if self._matrix is None:
            self._matrix = recompose(self._values, self._left_vectors, 
                                        self._right_vectors)
        return self


    @property 
    def backend(self):
        return get_backend(check_if_null(self._matrix, self._left_vectors))

    @property
    def left_vectors(self):
        return self.decompose()._left_vectors
    
    @property
    def eigenvectors(self):
        return self.decompose()._left_vectors

    @property
    def right_vectors(self):
        return self.decompose()._right_vectors
    
    def clip_rank(self, rank):
        """
        clips the rank of the matrix as indicated
        
        :param rank:    the new rank of the matrix

        :return clipped:Matrix  the same matrix but with clipped rank
        """
        if rank is None or (self.rank <= rank):
            return self
        else:
            self.decompose()
            return Matrix(decomposition=self.deccomposition,
                            eigenvalues=self._values[..., :rank],
                            left_vectors=self._left_vectors[..., :rank],
                            right_vectors=self._right_vectors[..., :rank, :],
                            rank=rank)
        
    
    @property
    def T(self):
        """
        Transpose :-)
        """
        return Matrix(
            matrix=self._matrix.mT if self._matrix is not None else None,
            decomposition=self.decomposition, rank=self.rank,
            eigenvalues=self._values,
            left_vectors=self._right_vectors,
            right_vectors=self._left_vectors
        )
    
    def __getitem__(self, item):
        def ind(m):
            return m[item] if m is not None else None
        
        return Matrix(
            matrix=ind(self._matrix),
            decomposition=self.decomposition, rank=self.rank,
            eigenvalues=ind(self._values),
            left_vectors=ind(self._left_vectors),
            right_vectors=ind(self._right_vectors)
        )
