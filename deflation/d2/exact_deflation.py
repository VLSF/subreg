import sys
import os
import jax.numpy as jnp
import sympy as sp
import numpy as np
import time

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu

def CG(x, b, N_it):
    r = b - A @ x
    p = r + 0.0
    R = [np.linalg.norm(r), ]
    X = [x, ]
    for _ in range(N_it):
        Ap = A @ p
        r_norm_m = r.T @ r
        alpha = r_norm_m / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        beta = (r.T @ r) / r_norm_m
        p = r + beta * p
        R.append(np.linalg.norm(b - A @ x))
        X.append(x)
    R = np.array(R)
    X = np.array(X)
    return X, R

def deflated_CG(x_, W, b, N_it):
    AW_inv = np.linalg.inv(W.T @ (A @ W))
    x = x_ + W @ (AW_inv @ (W.T @ (b - A @ x_)))
    r = b - A @ x
    mu = AW_inv @ (W.T @ (A @ r))
    p = r  - W @ mu
    R = [np.linalg.norm(r), ]
    X = [x, ]
    for _ in range(N_it):
        Ap = A @ p
        r_norm_m = r.T @ r
        alpha = r_norm_m / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        beta = (r.T @ r) / r_norm_m
        mu = AW_inv @ (W.T @ (A @ r))
        p = r - W @ mu + beta * p
        R.append(np.linalg.norm(b - A @ x))
        X.append(x)
    R = np.array(R)
    X = np.array(X)
    return X, R

if __name__ == "__main__":
    N_it = 201
    dataset_path = sys.argv[1]
    dataset_name = dataset_path.split("/")[-1][:-4]
    data = np.load(dataset_path)
    A_data = data["A_data"].astype(np.float64)
    indices = data["A_indices"]
    W = data["eigenvectors"].astype(np.float64)
    N = W.shape[-1]
    N_samples = W.shape[0]
    data = np.load(f"{dataset_name}.npz")
    rhs = data["rhs"]
    solutions = data["solutions"]

    header = "dataset_name,N_deflation,errors_50,error_100,errors_150,error_200"
    if not os.path.isfile(f'exact_deflation/results.csv'):
        with open(f'exact_deflation/results.csv', "w") as f:
            f.write(header)

    errors = []
    for i in range(100):
        A = coo_matrix((A_data[-i], (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsc()
        b = rhs[-i]
        exact = solutions[-i]
        X_CG, _ = CG(exact*0, b, N_it)
        X_deflate10, _ = deflated_CG(exact*0, W[-i][:10].reshape(10, -1).T, b, N_it)
        X_deflate20, _ = deflated_CG(exact*0, W[-i][:20].reshape(20, -1).T, b, N_it)
        X_deflate30, _ = deflated_CG(exact*0, W[-i][:30].reshape(30, -1).T, b, N_it)
        X_deflate40, _ = deflated_CG(exact*0, W[-i][:40].reshape(40, -1).T, b, N_it)
        e = []
        for x in [X_CG, X_deflate10, X_deflate20, X_deflate30, X_deflate40]:
            er = np.linalg.norm(np.expand_dims(exact, 0) - x, axis=1) / np.linalg.norm(exact)
            e.append(er)
        errors.append(np.array(e))
    errors = np.array(errors)
    write_data = ""
    for i, n in enumerate([0, 10, 20, 30, 40]):
        write_data += f"\n{dataset_name},{n},{np.mean(errors[:, i, 50])},{np.mean(errors[:, i, 100])},{np.mean(errors[:, i, 150])},{np.mean(errors[:, i, 200])}"
    with open('exact_deflation/results.csv', "a") as f:
        f.write(write_data)
    np.savez(f"exact_deflation/{dataset_name}_errors.npz", errors=errors)