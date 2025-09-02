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

if __name__ == "__main__":
    eig_dataset_path = sys.argv[1]
    regression_dataset_path = sys.argv[2]
    NN_path = sys.argv[3] # /home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/optimal/stochastic_normal
    Args = pd.read_csv(f"{NN_path}/results.csv")
    
    data = np.load(eig_dataset_path)
    dataset_name = eig_dataset_path.split("/")[-1][:-4]
    A_data = data["A_data"]
    indices = data["A_indices"]
    features = jnp.array(data["features"])
    features = features / jnp.max(jnp.abs(features), axis=[0, 2, 3], keepdims=True)
    features = features[-1000:]
    coordinates = jnp.array(data["coordinates"])
    D = features.ndim - 2

    data = np.load(regression_dataset_path)
    rhs = data["rhs"][:, 0]
    solutions = data["solutions"][:, 0]
    N = rhs.shape[-1]
    header = "dataset_name,experiment_type,N_eig,N_subspace,error"
    if not os.path.isfile(f'subspace_reduced_model/results.csv'):
        with open(f'subspace_reduced_model/results.csv', "w") as f:
            f.write(header)
    
    N_eig = 10
    Errors = dict()
    for N_subspace in [10, 20, 30, 40]:
        args = Args[Args["path_to_dataset"] == eig_dataset_path.split("/")[-1]]
        args = args[args["N_eig"] == N_eig]
        args = args[args["N_subspace"] == N_subspace]
        args = args.sort_values("test_eigvecs_error").iloc[0]
        
        N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], args["N_subspace"]]
        model = FFNO_normalised(args["N_layers"], N_features, args["N_modes"], D, random.PRNGKey(33))
        model = eqx.tree_deserialise_leaves(f"{NN_path}/model_{args['hash']}.eqx", model)
        predictions = scan(make_prediction_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
        eigenvectors = np.array(vmap(lambda x: jnp.linalg.qr(x.T)[0].T)(predictions.reshape(predictions.shape[0], predictions.shape[1], -1)))
        
        errors = []
        for i in range(1000):
            A = coo_matrix((A_data[-i], (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsr()
            Q = eigenvectors[-i].reshape(N_subspace, -1)
            A_ = Q @ (A @ Q.T)
            b_ = Q @ rhs[-i].reshape(-1,)
            solution = np.linalg.solve(A_, b_.reshape(-1,))
            solution = solution @ Q
            e = np.linalg.norm(solution - solutions[-i].reshape(-1,)) / np.linalg.norm(solutions[-i].reshape(-1,))
            errors.append(e)
        Errors[str(N_subspace)] = errors
    
    write_data = ""
    for key in Errors.keys():
        write_data += f"\n{dataset_name},{args['experiment_type']},{N_eig},{key},{np.mean(Errors[key])}"
    with open('subspace_reduced_model/results.csv', "a") as f:
        f.write(write_data)
        
    np.savez(f"subspace_reduced_model/{dataset_name}_{args['experiment_type']}_errors.npz", **Errors)