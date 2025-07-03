import torch
import numpy as np
import random
import os
import re
from typing import Optional, Union
from tqdm import tqdm


from src.utils import check_if_null, display_message


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                      Matrix and Linear Algebra Handling   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 
#   Helpers and handling of linear algebra and related calculations.
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
        left_vectors, values, right_vectors = U, S, Vh
        # values are set to U, S, Vh
    
    else:
        eigvals, eigvecs = backend.linalg.eigh(mat) if decomposition == "eigh"\
            else backend.linalg.eig(mat)
        # (r,)    (n, r)  note m = n  ; ascending values
    
        # put in descending order
        if decomposition == "eigh":
            # mat = V @ S @ Vh
            if backend == torch:
                values, left_vectors = eigvals.flip(-1), eigvecs.flip(-1)
            else:
                values, left_vectors = eigvals[..., ::-1], eigvecs[..., ::-1]
            right_vectors = transpose(left_vectors)
            # values are set to V, S, Vh
        else:
            # normal eigen decomposition
            # mat = V @ S @ V^{-1}
            inds = backend.argsort(-values, axis=-1)
            take_along = torch.take_along_dim if backend == torch \
                         else np.take_along_axis
            values = take_along(eigvals, inds, axis=-1)
            left_vectors = take_along(eigvecs, inds[..., None, :], axis=-1)

            right_vectors = backend.linalg.inv(left_vectors)
            # values are set to V, S, V^{-1}

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

    :param values:  shape (*, r,)
    :param left_vectors:    shape (*, m, r)
    :param right_vectors:   shape (*, r, n)

    :return mat:    shape (m, n)  with rank r = min(m, n)
    """
    return left_vectors @ (values[..., :, None] * right_vectors)



class Matrix:
    """
    Class to handle linear algebra logic for decomposed matrices

    Supports SVD, Eigenvalue decomposition, and Hermitian Eigenvalue
    decomposition (eigh).

    If provided with a matrix, it will decompose it according to the
    decomposition type specified. Otherwise, it will recompose as

    L S R

    where L are the left vectors, S are the values, and R are the (transposed) 
    right vectors.
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
    
    @property
    def shape(self):
        return self.matrix.shape
    
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
                      values=self.values,
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

    def to(self, backend=None, dtype=None, device=None):
        """
        Converts the matrix to the given backend and dtype
        :param backend:    the backend to convert to (torch or numpy)
        :param dtype:      the dtype to convert to
        :param device:     the device to convert to (if torch)
        """
        # check backend and dtype
        backend = check_if_null(backend, self.backend)
        dtype = check_if_null(dtype, self.matrix.dtype if self.matrix is not \
                              None else backend.float32)
        # define conversion 
        convert = torch.tensor if backend == torch else np.array
        convert = convert if backend != self.backend else lambda x: x

        def to(x):
            if backend == torch:
                return convert(x).to(device=device, dtype=dtype) if x is not \
                        None else None
            elif backend == np:
                return convert(x).astype(dtype) if x is not None else None
            else:
                message=f"""Invalid backend {backend}. Only numpy (np) and
                torch are valid."""
                raise ValueError(display_message(message))

        return Matrix(
                matrix=to(self._matrix),
                decomposition=self.decomposition, rank=self.rank,
                values=to(self._values),
                left_vectors=to(self._left_vectors),
                right_vectors=to(self._right_vectors)
               )

def check_matrix(mat, decomposition:Optional[str]=None,
                 rank:Optional[Union[float, int]]=float("inf")):
    """Checks if the matrix is a Matrix object and converts it to one if not"""
    if not isinstance(mat, Matrix):
        if decomposition is None:
            decomposition = "eigh"
        mat = Matrix(mat, decomposition=decomposition, rank=rank)     
    return mat


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                            Matrix Covariances           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# WISHLIST device moving for all of these calculations
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
        # X = U S V^T => X^T X = V S U^T U S V^T = V S^2 V^T
        mat = check_matrix(mat, decomposition="svd", rank=rank)
        cov = Matrix(values=mat.eigenvalues ** 2 / M,
                     left_vectors=mat.right_vectors.mT,
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
    mat1 = check_matrix(mat1, decomposition="svd", rank=rank)
    mat2 = check_matrix(mat2, decomposition="svd", rank=rank)

    if torch.equal(mat1.matrix, mat2.matrix):
        # if the matrices are the same, then return the covariance of mat1
        return compute_covariance(mat1, rank=rank)  

    return Matrix(mat1.matrix @ mat2.T.matrix,
                  decomposition="svd", rank=min(mat1.rank, mat2.rank, rank))


def compute_pairwise_covariances(*matrices, decomposition="svd", keys=None,
                                 symmetry=False):
    """
    Computes the pairwise covariance between a list of matrices. 

    :param decomposition:str  the decomposition type to use for the matrices.
                              Default: "svd". If set to "eigh", then the
                              matrices are assumed to be covariance matrices

    :param keys: list[str]    the keys to use for the matrices if not dict form
    :param matrices: Union[Matrix, torch.Tensor, np.array, dict]    the matrices 
                    to compute the pairwise covariance for. Each should be of 
                    shape (B, M_i, N); if B > 1 then the pairwise covariance 
                    is computed for each batch
    :param symmetry: Bool       whether to symmetrize the dictionary


    :return cross_dict: dict   [(i, j) -> Matrix]  the pairwise covariance
    """
    # make the matrix into a dict of [key -> Matrix] pairs
    if not isinstance(matrices[0], dict):
        # if keys is not provided, then use the indices as keys
        if check_if_null(keys, True, False):
            keys = list(range(len(matrices)))
        else:
            # check that keys and matrices have the same length
            message = """keys and matrices must have the same length if keys is
            provided"""
            assert len(keys) == len(matrices), display_message(message)
        
        # check and/or convert to Matrix objects
        matrices = [check_matrix(m, decomposition=decomposition) for \
                    m in matrices]

        matrix_dict = dict(zip(keys, matrices))
    else:
        matrices = matrices[0] # given a dict as the single tuple entry
        # verify matrices are Matrix objects
        new_values = [check_matrix(m, decomposition=decomposition) \
                      for m in matrices.values()]
        matrices = dict(zip(matrices.keys(), new_values))
        matrix_dict = matrices


    # assert that all matrices have the same backend
    message = """All matrices should have the same backend. I haven't
    implemented support yet for this otherwise"""
    backend = list(matrix_dict.values())[0].backend
    assert all(m.backend == backend for m in list(matrix_dict.values())), \
           display_message(message)
    

    # compute the pairwise covariances
    cross_dict = {}
    for key1, mat1 in tqdm(matrix_dict.items(), 
                           desc=f"Computing cross-covariances for {key1}"):
        for key2, mat2 in matrix_dict.items():

            if key1 == key2: # don't compute self-covariance
                continue

            # for symmetry
            if ( (key2, key1) in cross_dict ): 
                if symmetry: 
                    cross_dict[(key1, key2)] = cross_dict[(key2, key1)].T
                continue

            # if keys are distinct and not calculated already,
            # compute the cross covariance and add to dict
            cross_dict[(key1, key2)] = compute_cross_covariance(mat1.matrix,
                                                                 mat2.matrix)

    return cross_dict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               Matrix Alignments          
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def compute_alignment(mat1, mat2):
    """
    Computes the alignment matrix between two matrices mat1 and mat2 with sizes
    (M1, N) and (M2, N) and yields alignment matrix of shape (M1, M2)
    ( or batch-wise for (B, M1, N) and (B, M2, N) -> (B, M1, M2) )

    :param mat1: first matrix
    :param mat2: second matrix

    :return align_mat: Matrix
        the alignment matrix of shape (M1, M2) where the (i, j)th entry is the
        alignment between the ith row of mat1 and the jth row of mat2
    """
    # check and/or convert to Matrix objects
    mat1 = check_matrix(mat1, decomposition="svd")
    mat2 = check_matrix(mat2, decomposition="svd")

    # check that the backends are the same
    backend = mat1.backend
    message = """Both matrices should have the same backend. I haven't
    implemented support yet for this otherwise"""
    assert mat2.backend == backend, display_message(message)

    # compute the cross covariance
    cross_cov = compute_cross_covariance(mat1.matrix, mat2.matrix)
    # compute the alignment matrix
    align_mat = Matrix(cross_cov.left_vectors @ cross_cov.right_vectors,
                       decomposition="svd", rank=min(mat1.rank, mat2.rank))

    return align_mat


def convert_cross_cov_to_alignment(cross_cov_dict):
    """
    Converts a cross covariance dictionary to an alignment dictionary

    :param cross_cov_dict: dict     cross covariance dictionary
                                    [key1, key2] -> cross cov Matrix object
    :return align_dict: dict        the alignment dictionary
                                    [key1, key2] -> alignment Matrix object
    """
    align_dict = {}

    for key, cross_cov in tqdm(cross_cov_dict.items(), 
                        desc=f"Converting cross-covariance dict to alignment"):
        cross_cov = check_matrix(cross_cov, decomposition="svd")

        # get the alignment matrix
        alignment = Matrix(cross_cov.left_vectors @ cross_cov.right_vectors,
                           decomposition="svd")
        align_dict[key] = alignment
    return align_dict


def compute_pairwise_alignments(*matrices, decomposition="svd", keys=None):
    """ 
    Computes the orthogonal alignment dictionary between a list of matrices. 
    
    Matrices can also be a dictionary object with [key] -> matrix. The output
    dictionary will then be of the form 
            [key1, key2] -> alignment 
    Such that  ( alignment @ matrices[key2] ) is aligned to matrices[key1]


    
    :param matrices: Union[Matrix, torch.Tensor, np.array, dict]  the matrices to
                    compute the alignment dictionary for. Each should be of 
                    shape (B, M_i, N), if B > 1 then the alignment dictionary 
                    is computed for each batch
    :return align_dict: dict    the alignment dictionary of the form
                    
                    dict (i, j) -> orthogonal alignment matrix X that minimizes 
                    ||X M_i - M_j ||_F such that X^T X=I
    """

    # make the matrix into a dict of [key -> Matrix] pairs
    if not isinstance(matrices, dict):
        # if keys is not provided, then use the indices as keys
        if check_if_null(keys, True, False):
            keys = list(range(len(matrices)))
        else:
            # check that keys and matrices have the same length
            message = """keys and matrices must have the same length if keys is
            provided"""
            assert len(keys) == len(matrices), display_message(message)
        
        # check and/or convert to Matrix objects
        matrices = [check_matrix(m, decomposition=decomposition) for \
                    m in matrices]

        matrix_dict = dict(zip(keys, matrices))
    else:
        # verify matrices are Matrix objects
        new_values = [check_matrix(m, decomposition=decomposition) \
                      for m in matrices.values()]
        matrices = dict(zip(matrices.keys(), new_values))
        matrix_dict = matrices

    # get the cross covariances
    cross_dict = compute_pairwise_covariances(matrix_dict)

    # compute the alignments now
    align_dict = convert_cross_cov_to_alignment(cross_dict)

    return align_dict


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               Matrix Similarities          
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def compute_bw_sim(mat1, mat2, normalize_spectrum=False):
    """
    Computes the Bures-Wasserstein 2 similarity between two covariance matrices
    ||C_1^{1/2} C_2^{1/2}||_{nuc} / sqrt{TrC_1TrC_2}
    """
    # check and/or convert to Matrix objects
    mat1 = check_matrix(mat1, decomposition="eigh")
    mat2 = check_matrix(mat2, decomposition="eigh")

    if normalize_spectrum:
        mat1 = Matrix(left_vectors=mat1.left_vectors,
                      values=mat1.values /sum(mat1.values),
                      right_vectors=mat1.right_vectors)
        mat2 = Matrix(left_vectors=mat2.left_vectors,
                      values=mat2.values / sum(mat2.values),
                      right_vectors=mat2.right_vectors)


    # check that these are covariances
    message = """Both matrices must be covariance matrices"""
    assert mat1.decomposition == "eigh" and mat2.decomposition == "eigh", \
           display_message(message)
    
    # check that the backends are the same
    backend = mat1.backend
    message = """Both matrices should have the same backend. I haven't
    implemented support yet for this otherwise"""
    assert mat2.backend == backend, display_message(message)

    # get the numerator
    numer = backend.linalg.norm(mat1.sqrt.matrix @ mat2.sqrt.matrix, ord='nuc')

    # get the denominator
    f_tr = np.linalg.trace if backend == np else torch.trace
    denom = backend.sqrt(f_tr(mat1.matrix) * f_tr(mat2.matrix))
    sim = backend.abs(numer / denom)

    # since that numerical instability be honkin
    climp = backend.clip if backend == np else torch.clamp
    # eps = backend.finfo(mat1.matrix.dtype).eps
    sim = climp(sim, 0, 1)
    
    return sim

def compute_frobenius_sim(mat1, mat2):
    """
    Computes the Frobenius similarity between two matrices
    :param mat1: first matrix
    :param mat2: second matrix
    :return: float  the Frobenius similarity between the two matrices

    Currently assumes that the matrices are real
    """
    # check and/or convert to Matrix objects
    mat1 = check_matrix(mat1, decomposition="svd")
    mat2 = check_matrix(mat2, decomposition="svd")

    # check that the backends are the same
    backend = mat1.backend
    message = """Both matrices should have the same backend. I haven't
    implemented support yet for this otherwise"""
    assert mat2.backend == backend, display_message(message)

    if backend == torch:
        return (mat1.matrix * mat2.matrix).sum(axis=(-2, -1)) / \
               (mat1.matrix.norm(dim=(-2, -1)) * \
                mat2.matrix.norm(dim=(-2, -1)))
    else:
        return (mat1.matrix * mat2.matrix).sum(axis=(-2, -1)) / \
               (np.linalg.norm(mat1.matrix, axis=(-2, -1)) * \
                np.linalg.norm(mat2.matrix, axis=(-2, -1)))


def matrix_similarity(mat1, mat2, sim_type="bw"):
    """
    Wrapper function for different similarity metrics between two matrices.
    Computes the similarity between two matrices according to the given type.
    Currently supported similarities:
        - "bures wasserstein" or "bw" : Bures-Wasserstein p=2 similarity
        - "frobenius" or "fro" : Frobenius similarity

    :param mat1: first matrix
    :param mat2: second matrix
    :param sim_type:str  type of similarity to compute. Default: "bw" for 
                         Bures-Wasserstein 2 similarity. 
                            
    """
    sim_type = sim_type.lower()
    
    if sim_type == "bures wasserstein" or sim_type == "bw":
        return compute_bw_sim(mat1, mat2)
    elif sim_type == "frobenius" or sim_type == "fro":
        return compute_frobenius_sim(mat1, mat2)
    else:
        message = f"""Unknown similarity type {sim_type}. Try one of "bw" or
        "frobenius" instead"""
        raise ValueError(display_message(message))


def matrix_similarity_matrix(*matrices, sim_type="bw", rank=float("inf"), 
                             aligned=False, align_dict:Optional[dict]=None, 
                             numpy=True,
                             ):
    """
    Computes the similarity matrix between a list of matrices according to the
    given type

    :param sim_type:str     type of similarity to compute. Default: "bw" for 
                            Bures-Wasserstein 2 similarity. 
    :param rank:float|int|None      optional bound on the rank of the matrices
    :param matrices:Union[Matrix, torch.Tensor, np.array]       the matrices
    :param align_dict: dict|None    alignment dictionary provided for the given
                                    matrices. Default: None. If provided, the 
                                    keys should be indices corresponding to the
                                    given matrices such that 

                    align_dict[i, j] -> A  s.t.  A @ matrices[j] is aligned to 
                                                    matrices[i]
    
    :param numpy: bool      whether to return the similarity matrix as a numpy
                            array (this is useful if you want to pass it into 
                            sklearn's MDS implementation). Default: True 

    :return sim_mat: torch.Tensor|np.array
        the similarity matrix of shape (n, n) where n is the number of matrices
        and the (i, j)th entry is the similarity between matrices i and j
    """
    # check and/or convert to Matrix objects
    sim_type = sim_type.lower()
    decomp = {"bw": "eigh", "frobenius": "svd"}.get(sim_type, "eigh")
        # decomp type checking in matrix_similarity method  
          
    matrices = [check_matrix(m, decomposition=decomp, rank=rank) \
                for m in matrices]
    
    # compute the similarity matrix
    n = len(matrices)
    sim_mat = torch.eye(n, n, dtype=torch.float64)

    # if align_dict is provided, then already aligned
    if not aligned:
        aligned = check_if_null(align_dict, False, True)

    # for bures wasserstein, need to get align_dict if not aligned already
    if sim_type in ("bw", "bures wasserstein") and (not aligned):
        # get the alignment dict
        align_dict = compute_pairwise_alignments(*matrices)
            # keys here are [i, j] -> Matrix (alignment)


    def get_align_mat(key1, key2):
        if align_dict is not None:
            return align_dict[key1, key2]
        else:
            return None
        
    
    # calculate the similarities and make upper triangular matrix
    for i in tqdm(range(n), desc=f"Calculating similarities for matrix {i}"):
        for j in range(i+1, n):

            mat_i = matrices[i]
            mat_j = matrices[j]

            # get alignment and if necessary calculate the aligned matrix
            mat_align = get_align_mat(i, j)
            if check_if_null(mat_align, False, True):
                mat_j = Matrix(mat_align.matrix @ mat_j.matrix, 
                               decomposition=decomp)
            
            # actual similarity calculation
            sim = matrix_similarity(mat_i, mat_j, sim_type=sim_type)
            if get_backend(sim) != torch:
                sim = torch.from_numpy(sim).item()
            sim_mat[i,j] = sim
            

    # make make symmetric
    sim_mat += sim_mat.triu(1).T

    return sim_mat


