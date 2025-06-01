import os
import jax.numpy as jnp
import itertools
import sympy as sp
import numpy as np
import equinox as eqx
import time
import optax

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu
from jax import random, vmap, jit, config
from jax.lax import scan, dot_general, dynamic_slice_in_dim
from jax.tree_util import tree_map
from scipy.special import roots_legendre
from functools import partial
from jax.nn import gelu

def get_weights_and_indices(M, decay):
    ind = jnp.array([*itertools.product(range(M), repeat=2)])
    ind_ord = jnp.argsort(jnp.linalg.norm(ind, axis=1))
    ind = ind[ind_ord]
    w = 1/(1 + decay*jnp.linalg.norm(ind, axis=1)**2)
    return ind, w

def random_series(x, y, w_, ind_, c_a_x):
    s1 = jnp.real(jnp.sum(c_a_x*jnp.exp(1j*ind_[:, 0]*jnp.expand_dims(x, 0) + 1j*ind_[:, 1]*jnp.expand_dims(y, 0))*w_, axis=0))
    return s1

def a_rand(X, Y, params, key):
    ind_, w_, alpha, beta = params
    c_a = random.normal(key, (w_.shape[0], 1, 1))
    a = alpha + (beta-alpha)*(jnp.tanh(slope*random_series(X, Y, w_, ind_, c_a)) + 1)/2
    return a

def get_discretization_data(N):
    N_x = N_y = N

    x, y = np.linspace(0, 1, N_x+2)[1:-1], np.linspace(0, 1, N_y+2)[1:-1]
    X, Y = np.meshgrid(x, y, indexing="ij")

    h_x, h_y = x[1] - x[0], y[1] - y[0]
    x_half, y_half = np.linspace(h_x/2, 1-h_x/2, N_x+1), np.linspace(h_y/2, 1-h_y/2, N_y+1)
    X_half, _ = np.meshgrid(x_half, y, indexing="ij")
    _, Y_half = np.meshgrid(x, y_half, indexing="ij")

    i, j = np.arange(N_x, dtype=int), np.arange(N_y, dtype=int)
    I, J = np.meshgrid(i, j, indexing="ij")
    lex = lambda i, j, N_x: (j + i*N_x)
    diag = lex(I, J, N_x)
    i_m = lex(I-1, J, N_x)
    i_p = lex(I+1, J, N_x)
    j_m = lex(I, J-1, N_x)
    j_p = lex(I, J+1, N_x)

    mask_i_m = (I-1) >= 0
    mask_i_p = (I+1) < N_x
    mask_j_m = (J-1) >= 0
    mask_j_p = (J+1) < N_y

    rows = np.concatenate([diag.reshape(-1,), diag[mask_i_m], diag[mask_i_p], diag[mask_j_m], diag[mask_j_p]])
    cols = np.concatenate([diag.reshape(-1,), i_m[mask_i_m], i_p[mask_i_p], j_m[mask_j_m], j_p[mask_j_p]])
    indices = np.stack([cols, rows], 1)
    return mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices

def get_dataset(N, N_samples, N_eigenvalues, params, key):
    mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices = get_discretization_data(N)

    Keys = random.split(key, (N_samples, 2))
    Vals, Vecs, features, Data = [], [], [], []
    for keys in Keys:
        a1_p = a_rand(X_half[1:], Y, params, keys[0])
        a1_m = a_rand(X_half[:-1], Y, params, keys[0])
        a2_p = a_rand(X, Y_half[:, 1:], params, keys[1])
        a2_m = a_rand(X, Y_half[:, :-1], params, keys[1])
        a1 = a_rand(X, Y, params, keys[0])
        a2 = a_rand(X, Y, params, keys[1])

        data = np.concatenate([
            (a1_p + a1_m + a2_p + a2_m).reshape(-1,),
            -a1_m[mask_i_m],
            -a1_p[mask_i_p],
            -a2_m[mask_j_m],
            -a2_p[mask_j_p],
        ])

        A = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsc()
        A_ = splu(A)
        L_ = LinearOperator(A.shape, matvec=lambda x: A_.solve(x), dtype=A.dtype)
        
        vals, vecs = eigsh(L_, k=N_eigenvalues)
        vecs = np.array(vecs)
        vecs = np.moveaxis(vecs.reshape(N, N, -1), 2, 0)[:N_eigenvalues]
        vals = vals[:N_eigenvalues]
        vals = 1 / vals
        order = np.argsort(vals)
        vecs = vecs[order]
        vals = vals[order]

        Vals.append(vals[:N_eigenvalues])
        Vecs.append(vecs[:N_eigenvalues])
        features.append(np.stack([a1, a2]))
        Data.append(data)

    Vals = np.stack(Vals)
    Data = np.stack(Data)
    Vecs = np.stack(Vecs)
    coords = np.stack([X, Y])
    features = np.stack(features)
    Vals, Vecs, coords, features, Data, indices = jnp.array(Vals), jnp.array(Vecs), jnp.array(coords), jnp.array(features), jnp.array(Data), jnp.array(indices)
    data = {
        "eigenvalues": Vals,
        "eigenvectors": Vecs,
        "coordinates": coords,
        "features": features,
        "A_data": Data,
        "A_indices": indices
    }
    return data

if __name__ == "__main__":
    N_samples = 5000
    N_eigenvalues = 50
    N = 100
    M = 100
    slope = 1
    alpha = 1
    beta = 50
    decay = 1e-1
    
    key = random.PRNGKey(44)
    ind, w = get_weights_and_indices(M, decay)
    ind_ = ind.reshape(ind.shape[0], ind.shape[1], 1, 1)
    w_ = w.reshape(-1, 1, 1)
    params = [ind_, w_, alpha, beta]
    
    data = get_dataset(N, N_samples, N_eigenvalues, params, key)
    jnp.savez("twin_flattops.npz", **data)