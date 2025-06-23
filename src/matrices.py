import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union


from src.utils import check_if_null, display_message


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
                 values:Optional[Union[torch.Tensor, np.array]]=None,
                 eigenvectors:Optional[Union[torch.Tensor, np.array]]=None,
                 left_vectors:Optional[Union[torch.Tensor,np.array]]=None,
                 right_vectors:Optional[Union[torch.Tensor, np.array]]=None,
                 rank=float("inf")
                 ):
        
        self._matrix = matrix  # shape (*, m, n)

        # check decomposition type if decomposition is specified
        if check_if_null(decomposition, False, True) and \
            (decomposition.lower() not in ("svd", "eig", "eigh")):
            message = """decomposition must be one of "svd", "eig", "eigh" or
            `None`. If it is `None`, the default of "eigh" is used instead."""
            raise ValueError(display_message(message))
        self.decomposition = decomposition.lower() if \
                                decomposition is not None else "eigh"

        # set and check rank
        message = """rank cannot be `None`. Try initializating with the default
        rank instead"""
        assert rank is not None, display_message(message)
        if check_if_null(matrix, False, True):
            rank = min(rank, *matrix.shape[-2:])
        if check_if_null(values, False, True):
            rank = min(rank, values.shape[-1])
        self.rank = rank

        # add decomposition information
        # validation for eigenvectors / left singular vectors
        if (check_if_null(eigenvectors, False, True) and \
            check_if_null(left_vectors, False, True)):

            if eigenvectors == left_vectors:
                message = """Both left_vectors and eigenvectors are specified
                with the same values. It is best to use left_vectors with
                decomposition="svd" and eigenvectors with decomposition="eig"
                or "eigh" to avoid confusion. To hide this message, initialize
                with only one of the two parameters specified."""
                print(display_message(message))
            else:
                message = """Both left_vectors and eigenvectors have been
                specified with different values. Only one of left_vectors and
                eigenvectors should be used."""
                raise ValueError(display_message(message)) # stops init
            
        left_vectors = check_if_null(left_vectors, eigenvectors)

        self._values = values # shape (*, rank,) 
        self._left_vectors = left_vectors # shape (*, m, rank)
        self._right_vectors = right_vectors # shape (*, rank, n)

        # initialiation validation
        if check_if_null(matrix, True, False) and \
            (check_if_null(eigenvectors, True, False) and \
             check_if_null(left_vectors, True, False)):
            message = """At least one of either `matrix` or the decomposition
            components must be defined"""
            raise ValueError(display_message(message))
        
    def decompose(self):
        """
        Decomposes the matrix according to the decomposition type and 
        updates corresponding properties 
        """
        if self._values is None:
            self._values, self._left_vectors, self._right_vectors = \
            decompose(self._matrix, decomposition=self.decomposition, 
                        rank=self.rank)
            self.rank = self._values.shape[-1]
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
    def matrix(self):
        return self.recompose()._matrix

    @property
    def left_vectors(self):
        return self.decompose()._left_vectors # shape (*, m, rank)
    
    @property
    def eigenvectors(self):
        return self.decompose()._left_vectors # shape (*, m, rank)

    @property
    def right_vectors(self):
        return self.decompose()._right_vectors # shape (*, rank, n)
    
    @property
    def values(self):
        return self.decompose()._values # shape (*, rank,)

    @property 
    def eigenvalues(self):
        return self.decompose()._values # shape (*, rank,) 
    
    @property
    def singular_values(self):
        return self.decompose()._values # shape (*, rank,) 
    
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
            return Matrix(decomposition=self.decomposition,
                          values=self._values[..., :rank],
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
            values=self._values,
            left_vectors=self._right_vectors,
            right_vectors=self._left_vectors
        )
    
    @property
    def sqrt(self):
        """
        Computes the matrix square root
        If X = USV^T then sqrt(X) = U S^{-1/2} V^T 
        returns Matrix with the square rooted values. Assumes values are Real
        and nonnegative. 
        """
        new_values = self.backend.sqrt(self.backend.abs(self.values))
        return self.new_values(new_values)
    
    def new_values(self, new_values):
        """
        Returns a matrix that is a copy of the original matrix, but with new
        values as specified.

        :param new_values: the new values to use. Should be of shape (*, rank,)
        """
        new_values = check_if_null(new_values, self.values)
        return Matrix(decomposition=self.decomposition, rank=self.rank,
                      values=new_values,
                      left_vectors=self.left_vectors,
                      right_vectors=self.right_vectors)

    def new_vectors(self, new_left_vectors, new_right_vectors=None):
        """
        Returns a matrix that is a copy of the original matrix, but with new 
        left and/or right vectors as specified. 
        """
        new_left_vectors = check_if_null(new_left_vectors, self.left_vectors)
        if self.decomposition == "svd":
            new_right_vectors = check_if_null(new_right_vectors, 
                                            self.right_vectors)
        else:
            new_right_vectors = check_if_null(new_right_vectors, None)
            if check_if_null(new_right_vectors, False):
                message = """Matrix decomposition is set to eig or eigh but new
                right vectors are specified. Fix this or create a new Matrix
                directly"""
                raise ValueError(display_message(message))

        return Matrix(decomposition=self.decomposition, rank=self.rank,
                      singularvalues=self.singular_values,
                      left_vectors=new_left_vectors,
                      right_vectors=new_right_vectors)
    
    def __getitem__(self, item):
        """
        Allows iteration over the first dimension
        """
        def ind(m):
            return m[item] if m is not None else None
        
        return Matrix(ind(self._matrix),
            decomposition=self.decomposition, rank=self.rank,
            values=ind(self._values),
            left_vectors=ind(self._left_vectors),
            right_vectors=ind(self._right_vectors)
        )
    
    def flatten(self):
        """
        If matrix is of dimension (B, M, N) returns a matrix of size (B*M , N)
        """
        if self.matrix.ndim == 3:
            new = self.matrix.reshape( (-1, self.matrix.shape[-1]) )
            return Matrix(new, 
                          decomposition=self.decomposition, rank=self.rank)
        else:
            return self


def compute_covariance(mat, rank:Optional[Union[float, int]]=float("inf"), 
                       full_matrices:bool=False):
    """
    Compute the covariance of a matrix X 
    X has shape (M, N) ; cov given by (1/M)X^T X has shape (N, N)

    :param mat:     the matrix to compute covariance of
    :param rank:float|int|None      optional bound on the rank of the matrix
                                    default: float(inf). If set to None, the 
                                    default value is used instead
    :param full_matrices:bool       whether to compute full matrices with SVD.
                                    default: False

    :return covariance:Matrix
    """

    # check and set rank
    if check_if_null(rank, True, False):
        message = """
        User specified rank=`None` detected; using upper bound of 
        rank=float(inf) instead
        """
        display_message(message)
        rank = float("inf")
    
    M, N = mat.shape[-2:]
    rank = min(rank, N, M)

    if not full_matrices:
        # do the SVD computation and use to get the covariance
        # X = USV^T => X^TX VSU^TUSV^T = VS^2
        mat = Matrix(mat, decomposition="svd", rank=rank)
        cov = Matrix(values=mat.eigenvalues ** 2 / M,
                     left_vectors=mat.right_vectors,
                     right_vectors=mat.right_vectors,
                     decomposition="eigh",
                     rank=rank)
    else:
        # compute the covariance directly
        cov = Matrix(mat.mT @ mat / M, 
                     decomposition="eigh",
                     rank=rank)

    return cov

def compute_cross_covariance(mat1, mat2, rank=float("inf")):
    """
    Computes the cross covariance of two matrices sized (M1, N) and (M2, N) and
    yields (M1, M2)
    ( or batch-wise for (B, M1, N) and (B, M2, N) -> (B, M1, M2) )

    The cross covariance of X and Y is given as C_{XY} = XY^T

    :param mat1: first matrix
    :param mat2: second matrix
    :param rank: optional bound on output rank
    """
    # check and/or convert to Matrix objects
    if not isinstance(mat1, Matrix):
        mat1 = Matrix(mat1)
    if not isinstance(mat2, Matrix):
        mat2 = Matrix(mat2)
    
    return Matrix(mat1.matrix @ mat2.T.matrix,
                  decomposition="svd", rank=min(mat1.rank, mat2.rank, rank))


def compute_bw_sim(mat1, mat2):
    """
    Computes the Bures-Wasserstein 2 similarity between two covariance matrices
    ||C_1^{1/2} C_2^{1/2}||_{nuc} / sqrt{TrC_1TrC_2}
    """
    # check and/or convert to Matrix objects
    if not isinstance(mat1, Matrix):
        mat1 = Matrix(mat1)
    if not isinstance(mat2, Matrix):
        mat2 = Matrix(mat2)

    # check that these are covariances
    message = """Both matrices must be covariance matrices"""
    assert mat1.decomposition == "eigh" and mat2.decomposition == "eigh", \
           display_message(message)
    
    backend = mat1.backend
    message = """Both matrices should have the same backend. I haven't
    implemented support yet for this otherwise"""
    assert mat2.backend == backend, display_message(message)

    # get the numerator
    numer = backend.linalg.norm(mat1.sqrt.matrix @ mat2.sqrt.matrix, ord='nuc')

    # get the denominator
    f_tr = np.linalg.trace if backend == np else torch.trace
    denom = backend.sqrt(f_tr(mat1.matrix) * f_tr(mat2.matrix))
    
    return numer / denom