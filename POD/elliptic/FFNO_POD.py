import sys
import os
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import pandas as pd

from scipy.sparse import coo_matrix
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

class FFNO(eqx.Module):
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
        return u

    def comput_hidden(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder(u)
        for conv1, conv2, A in zip(self.convs1, self.convs2, self.A):
            u += gelu(conv2(gelu(conv1(self.spectral_conv(u, A)))))
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

def make_prediction_hidden_scan(carry, i):
    model, features, coords = carry
    prediction = model.comput_hidden(features[i], coords)
    return carry, prediction

if __name__ == "__main__":
    dataset_eig_path_load = sys.argv[1] # '/home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/lone_flattop.npz'
    dataset_reg_path_load = sys.argv[2] # '/home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/lone_flattop_regression.npz'
    NN_path = sys.argv[3] # '/home/jovyan/vfanaskov/eigenvectors/elliptic_regression_2D/FFNO'
    dataset_name = dataset_reg_path_load.split("/")[-1]
    dataset_reg_path_filter = "/home/jovyan/vfanaskov/elliptic_regression_2D/" + dataset_name

    header = "dataset_name,model_hash,N_basis,test_error"
    if not os.path.isfile(f'FFNO_POD/results.csv'):
        with open(f'FFNO_POD/results.csv', "w") as f:
            f.write(header)

    Args = pd.read_csv(f"{NN_path}/results.csv")
    Args = Args[Args['dataset_reg_path'] == dataset_reg_path_filter]
    
    data = np.load(dataset_eig_path_load)
    features = jnp.array(data["features"])
    coordinates = jnp.array(data["coordinates"])
    A_data = data["A_data"]
    indices = data["A_indices"]
    
    data = np.load(dataset_reg_path_load)
    rhs = jnp.array(data["rhs"])
    sol = jnp.array(data["solutions"])
    features = jnp.concatenate([jnp.array(data["rhs"]), features], axis=1)
    features = features / jnp.max(jnp.abs(features), axis=[0, 2, 3], keepdims=True)
    targets = jnp.array(data["solutions"])
    targets = targets / jnp.max(jnp.abs(targets), axis=[0, 2, 3], keepdims=True)

    Data = dict()
    for i in range(3):
        args = Args.sort_values("val_error").iloc[i]
        D = features.ndim - 2
        N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], targets.shape[1]]
        model = FFNO(args["N_layers"], N_features, args["N_modes"], D, random.PRNGKey(33))
        model = eqx.tree_deserialise_leaves(f"{NN_path}/model_{args['hash']}.eqx", model)
        predictions_hidden = scan(make_prediction_hidden_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
        predictions = scan(make_prediction_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
        extract_modes = lambda x: jnp.linalg.svd(x.reshape(x.shape[0], -1), full_matrices=False)[2]
        _, Q = scan(lambda a, b: (a, extract_modes(b)), None, predictions_hidden)
        N_basis = jnp.linspace(10, Q.shape[1], 10, dtype=jnp.int32)
        
        for n in N_basis:
            errors_test = []
            ind = args["N_train"] + args["N_val"] + jnp.arange(features.shape[0] - (args["N_train"] + args["N_val"]))
            for sample in ind:
                A = coo_matrix((A_data[sample], (indices[:, 0], indices[:, 1])), shape=(Q.shape[-1], Q.shape[-1])).tocsr()
                exact = sol[sample].reshape(-1,)
                b = rhs[sample].reshape(-1,)
                q = Q[sample][:n]
                
                sol_ = np.linalg.solve((q @ (A @ q.T)), q @ b)
                sol_ = np.array(sol_ @ q)
                err = np.linalg.norm(exact - sol_) / np.linalg.norm(exact)
                errors_test.append(err)
            errors_test = jnp.array(errors_test)
            Data[f"{args['hash']}, {n}"] = errors_test
            write_data = f"\n{dataset_name},{args['hash']},{n},{jnp.mean(errors_test)}"
            with open('FFNO_POD/results.csv', "a") as f:
                f.write(write_data)
    jnp.savez(f"FFNO_POD/metrics_{dataset_name}", **Data)