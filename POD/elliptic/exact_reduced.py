import sys
import os
import jax.numpy as jnp
import sympy as sp
import numpy as np
import time

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu

if __name__ == "__main__":
    eig_dataset_path = sys.argv[1]
    regression_dataset_path = sys.argv[2]
    data = np.load(eig_dataset_path)
    dataset_name = eig_dataset_path.split("/")[-1][:-4]
    A_data = data["A_data"].astype(np.float64)
    indices = data["A_indices"]
    eigenvectors = data["eigenvectors"].astype(np.float64)
    data = np.load(regression_dataset_path)
    rhs = data["rhs"][:, 0].astype(np.float64)
    solutions = data["solutions"][:, 0].astype(np.float64)
    N = rhs.shape[-1]
    header = "dataset_name,N_basis,error"
    if not os.path.isfile(f'exact_reduced_model/results.csv'):
        with open(f'exact_reduced_model/results.csv', "w") as f:
            f.write(header)
    Errors = []
    write_data = ""
    for N_basis in [8, 10]:
        errors = []
        for i in range(1000):
            A = coo_matrix((A_data[-i], (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsr()
            Q = eigenvectors[-i][:N_basis].reshape(N_basis, -1)
            A_ = Q @ (A @ Q.T)
            b_ = Q @ rhs[-i].reshape(-1,)
            solution = np.linalg.solve(A_, b_.reshape(-1,))
            solution = solution @ Q
            e = np.linalg.norm(solution - solutions[-i].reshape(-1,)) / np.linalg.norm(solutions[-i].reshape(-1,))
            errors.append(e)
        errors = np.array(errors)
        write_data += f"\n{dataset_name},{N_basis},{np.mean(errors)}"
        Errors.append(errors)
    Errors = np.array(Errors)
    with open("exact_reduced_model/results.csv", "a") as f:
        f.write(write_data)
    np.savez(f"exact_reduced_model/{dataset_name}_errors.npz", errors=Errors)