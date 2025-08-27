import sys
import jax.numpy as jnp
import sympy as sp
import numpy as np
import time

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu
from jax import random

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    dataset_name = dataset_path.split("/")[-1][:-4]
    data = np.load(dataset_path)
    A_data = data["A_data"].astype(np.float64)
    indices = data["A_indices"]
    N = data["eigenvectors"].shape[-1]
    N_samples = data["eigenvectors"].shape[0]

    solutions = []
    rhs = []
    key = random.PRNGKey(45)
    keys = random.split(key, N_samples)
    for data in A_data:
        A = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsc()
        b = np.array(random.normal(keys[2], (N**2,))).astype(np.float64)
        solution = spsolve(A, b)
        rhs.append(b)
        solutions.append(solution)
    rhs = np.array(rhs)
    solutions = np.array(solutions)
    np.savez(f"{dataset_name}.npz", rhs=rhs, solutions=solutions)