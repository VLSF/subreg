import sys
import os
import jax.numpy as jnp
import sympy as sp
import numpy as np
import time

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu

if __name__ == "__main__":
    N_train = 4000
    N_basis = [10, 50, 100, 150, 200]
    eig_dataset_path = sys.argv[1]
    regression_dataset_path = sys.argv[2]
    data = np.load(eig_dataset_path)
    dataset_name = eig_dataset_path.split("/")[-1][:-4]
    A_data = data["A_data"].astype(np.float64)
    indices = data["A_indices"]
    data = np.load(regression_dataset_path)
    rhs = data["rhs"][:, 0].astype(np.float64)
    solutions = data["solutions"][:, 0].astype(np.float64)
    vt = np.array(jnp.linalg.svd(jnp.array(solutions[:N_train]).reshape(N_train, -1))[2])
    N = rhs.shape[-1]
    header = "dataset_name,POD_type,N_basis,error"
    if not os.path.isfile(f'POD/results.csv'):
        with open(f'POD/results.csv', "w") as f:
            f.write(header)
    ind_test = np.arange(solutions.shape[0] - N_train) + N_train
    Data = dict()
    for N_b in N_basis:
        errors_non_intrusive = np.linalg.norm((solutions[N_train:].reshape(solutions.shape[0] - N_train, -1) @ vt[:N_b].T) @ vt[:N_b] - solutions[N_train:].reshape(solutions.shape[0] - N_train, -1), axis=1) / np.linalg.norm(solutions[N_train:].reshape(solutions.shape[0] - N_train, -1), axis=1)
        Data[f"{N_b}, non-intrusive"] = errors_non_intrusive
        errors_intrusive = []
        for i in ind_test:
            A = coo_matrix((A_data[i], (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsr()
            A_ = vt[:N_b] @ (A @ vt[:N_b].T)
            rhs_ = vt[:N_b] @ rhs[i].reshape(-1,)
            sol_ = np.linalg.solve(A_, rhs_)
            sol = vt[:N_b].T @ sol_
            errors_intrusive.append(np.linalg.norm(solutions[i].reshape(-1,) - sol) / np.linalg.norm(solutions[i].reshape(-1,)))
        errors_intrusive = np.array(errors_intrusive)
        Data[f"{N_b}, intrusive"] = errors_intrusive
        with open(f'POD/results.csv', "a") as f:
            f.write(f"\n{dataset_name},non-intrusive,{N_b},{np.mean(errors_non_intrusive)}")
        with open(f'POD/results.csv', "a") as f:
            f.write(f"\n{dataset_name},intrusive,{N_b},{np.mean(errors_intrusive)}")
    np.savez(f"POD/{dataset_name}_metrics.npz", **Data)