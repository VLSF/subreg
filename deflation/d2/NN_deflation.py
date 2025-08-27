import numpy as np
import equinox as eqx
import sys
import os
import jax.numpy as jnp
import sympy as sp
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu
from jax import random, vmap
from jax.lax import scan, dot_general, dynamic_slice_in_dim
from jax.tree_util import tree_map, tree_flatten
from jax.nn import gelu

def normalize_conv(A, s1=1.0, s2=1.0):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight * s1)
    try:
        A = eqx.tree_at(lambda x: x.bias, A, A.bias * s2)
    except:
        pass
    return A

class FFNO_normalised(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list
    A: jnp.array

    def __init__(self, N_layers, N_features, N_modes, D, key, s1=1.0, s2=1.0, s3=1.0):
        n_in, n_processor, n_out = N_features

        keys = random.split(key, 3 + 2*N_layers)
        self.encoder = normalize_conv(eqx.nn.Conv(D, n_in, n_processor, 1, key=keys[-1]), s1=s1, s2=s2)
        self.decoder = normalize_conv(eqx.nn.Conv(D, n_processor, n_out, 1, key=keys[-2]), s1=s1, s2=s2)
        self.convs1 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
        self.A = random.normal(keys[-3], [N_layers, n_processor, n_processor, N_modes, D], dtype=jnp.complex64) * s3

    def __call__(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder(u)
        for conv1, conv2, A in zip(self.convs1, self.convs2, self.A):
            u += gelu(conv2(gelu(conv1(self.spectral_conv(u, A)))))
        u = self.decoder(u)
        norm = jnp.linalg.norm(u.reshape(u.shape[0], -1), axis=1)
        u = u / norm.reshape([-1, ]+ [1,]*(u.ndim-1))
        return u

    def spectral_conv(self, v, A):
        u = 0
        N = v.shape
        for i in range(A.shape[-1]):
            u_ = dynamic_slice_in_dim(jnp.fft.rfft(v, axis=i+1), 0, A.shape[-2], axis=i+1)
            u_ = dot_general(A[:, :, :, i], u_, (((1,), (0,)), ((2, ), (i+1, ))))
            u_ = jnp.moveaxis(u_, 0, i+1)
            u += jnp.fft.irfft(u_, axis=i+1, n=N[i+1])
        return u

def make_prediction_scan(carry, i):
    model, features, coords = carry
    prediction = model(features[i], coords)
    return carry, prediction

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
    
    data = np.load(f"{dataset_name}.npz")
    rhs = data["rhs"]
    solutions = data["solutions"]

    NN_path = sys.argv[2] # /home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/optimal/stochastic_normal
    Args = pd.read_csv(f"{NN_path}/results.csv")

    header = "dataset_name,experiment_type,N_eig,N_subspace,errors_50,error_100,errors_150,error_200"
    if not os.path.isfile(f'NN_deflation/results.csv'):
        with open(f'NN_deflation/results.csv', "w") as f:
            f.write(header)

    data = np.load(dataset_path)
    A_data = data["A_data"].astype(np.float64)
    indices = data["A_indices"]
    
    features = jnp.array(data["features"])
    features = features / jnp.max(jnp.abs(features), axis=[0, 2, 3], keepdims=True)
    features = features[-200:]
    coordinates = jnp.array(data["coordinates"])
    D = features.ndim - 2
    
    N_eig = 10
    Errors = dict()
    for N_subspace in [10, 20, 30, 40]:
        args = Args[Args["path_to_dataset"] == dataset_path.split("/")[-1]]
        args = args[args["N_eig"] == N_eig]
        args = args[args["N_subspace"] == N_subspace]
        args = args.sort_values("test_eigvecs_error").iloc[0]
        
        N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], args["N_subspace"]]
        model = FFNO_normalised(args["N_layers"], N_features, args["N_modes"], D, random.PRNGKey(33))
        model = eqx.tree_deserialise_leaves(f"{NN_path}/model_{args['hash']}.eqx", model)
        predictions = scan(make_prediction_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
        Q = vmap(lambda x: jnp.linalg.qr(x.T)[0].T)(predictions.reshape(predictions.shape[0], predictions.shape[1], -1))
        
        W = np.array(Q).astype(np.float64)
        N = features.shape[-1]
    
        errors = []
        for i in range(100):
            A = coo_matrix((A_data[-i], (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsc()
            b = rhs[-i]
            exact = solutions[-i]
            X_deflate, _ = deflated_CG(exact*0, W[-i].T, b, N_it)
            errors.append(np.linalg.norm(np.expand_dims(exact, 0) - X_deflate, axis=1) / np.linalg.norm(exact))
        errors = np.array(errors)
        Errors[str(N_subspace)] = errors

    write_data = ""
    for key in Errors.keys():
        write_data += f"\n{dataset_name},{args['experiment_type']},{N_eig},{key},{np.mean(Errors[key][:, 50])},{np.mean(Errors[key][:, 100])},{np.mean(Errors[key][:, 150])},{np.mean(Errors[key][:, 200])}"
    with open('NN_deflation/results.csv', "a") as f:
        f.write(write_data)
        
    np.savez(f"NN_deflation/{dataset_name}_{args['experiment_type']}_errors.npz", **Errors)