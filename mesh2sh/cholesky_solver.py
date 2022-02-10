import torch
from scipy.sparse import coo_matrix
from sksparse.cholmod import cholesky

'''
We want to use a Cholesky solver with sparse tensors
There is no good way to do it currently using pytorch
So we use an extension of scipy to do it :
    (https://scikit-sparse.readthedocs.io/en/latest/index.html)
Scipy sparse array are assumed to be 2D
'''


def sparse_coo_from_scipy_to_pytorch(sparse_array):
    indices = torch.stack([torch.from_numpy(sparse_array.row), torch.from_numpy(sparse_array.col)], dim=0)
    values = torch.from_numpy(sparse_array.data)
    return torch.sparse_coo_tensor(indices, values, sparse_array.shape)


def sparse_coo_from_pytorch_to_scipy(sparse_array):
    assert sparse_array.is_sparse, 'Array needs to be sparse'
    sparse_array = sparse_array.cpu()
    if len(sparse_array.shape) == 1:
        sparse_array = sparse_array.unsqueeze(1)
    try:
        values = sparse_array.values()
    except RuntimeError:
        values = sparse_array._values()
    try:
        indices = sparse_array.indices()
    except RuntimeError:
        indices = sparse_array._indices()
    return coo_matrix((values, (indices[0], indices[1])), shape=sparse_array.shape)


def solver_cholesky(A, b, return_dense=True):
    """
    Solve Ax=b
    """
    device = A.device
    dtype = A.dtype
    A = A.cpu()
    b = b.cpu()

    assert A.is_sparse and not b.is_sparse

    A = sparse_coo_from_pytorch_to_scipy(A)
    A = A.tocsc()

    factor = cholesky(A)
    x_sci = factor(b)
    if return_dense:
        return torch.from_numpy(x_sci).to(device).to(dtype)
    else:
        x_sci = coo_matrix(x_sci)
        return sparse_coo_from_scipy_to_pytorch(x_sci).to(device).to(dtype)
