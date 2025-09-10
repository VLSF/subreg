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

class conv_trunk_2D(eqx.Module):
    convs: list
    projector: list
    encoder: list

    def __init__(self, N_x, N_features, N_layers, kernel_size, key):
        N_in, N_encoder, N_out = N_features
        N_layers = min(jnp.log2(N_x).astype(int).item(), N_layers)
        keys = random.split(key, (N_layers, 2))
        self.convs = []
        N = N_encoder
        x = jnp.zeros((N_encoder, N_x, N_x))
        for key in keys:
            c = [
                eqx.nn.Conv(2, N, 2*N, kernel_size = kernel_size, padding = 'SAME', key = key[0]),
                eqx.nn.Conv(2, 2*N, 2*N, kernel_size = kernel_size, stride=2, key = key[1])
            ]
            x = c[1](c[0](x))
            N = 2*N
            self.convs.append(c)
        keys = random.split(keys[-1, -1])
        self.projector = [random.normal(keys[0], (N_out, x.size)) / jnp.sqrt(N_out + x.shape[1]*N//2), jnp.zeros((N_out,))]
        self.encoder = [jnp.zeros((N_encoder, 1, 1)), random.normal(keys[1], (N_encoder, N_in)) / jnp.sqrt(N_encoder + N_in)]

    def __call__(self, u, coords):
        # u.shape = (N_features, N_x, N_x); coords.shape = (2, N_x, N_x)
        x = jnp.concatenate([u, coords], axis=0)
        x = self.encoder[0] + dot_general(self.encoder[1], x, (((1,), (0,)), ((), ())))
        for conv in self.convs:
            x = gelu(conv[1](gelu(conv[0](x))))
        x = self.projector[0] @ x.reshape(-1,) + self.projector[1]
        return x

class MLP(eqx.Module):
    weights: list
    biases: list

    def __init__(self, N_features, N_layers, key):
        N_in, N_processor, N_out = N_features
        keys = random.split(key, N_layers+1)
        Ns = [N_in,] + [N_processor,]*N_layers + [N_out,]
        self.biases = [jnp.zeros((No, 1)) for No in Ns[1:]]
        self.weights = [random.normal(key, (No, Ni)) / jnp.sqrt(No + Ni) for Ni, No, key in zip(Ns[:-1], Ns[1:], keys)]

    def __call__(self, coords):
        # coords.shape = (N_in, N_x, N_y, ...)
        c = coords.reshape(coords.shape[0], -1)
        for w, b in zip(self.weights, self.biases):
            c = gelu(w @ c + b)
        c = c.reshape([c.shape[0],] + list(coords.shape[1:]))
        return c

class DeepONet(eqx.Module):
    trunk: eqx.Module
    branch: eqx.Module

    def __init__(self, trunk_params, branch_params, key):
        N_x, N_features, N_layers, kernel_size = trunk_params
        N_features_, N_layers_ = branch_params
        keys = random.split(key)
        self.trunk = conv_trunk_2D(N_x, N_features, N_layers, kernel_size, keys[0])
        self.branch = MLP(N_features_, N_layers_, keys[1])

    def __call__(self, u, coords, coords_x):
        coeff = self.trunk(u, coords)
        phi = self.branch(coords_x)
        res = jnp.expand_dims(dot_general(phi, coeff, (((0,), (0,)), ((), ()))), 0)
        return res

def make_prediction_scan(carry, i):
    model, features, x, coords = carry
    prediction = model(features[i], x, coords)
    return carry, prediction

if __name__ == "__main__":
    dataset_eig_path_load = sys.argv[1] # '/home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/lone_flattop.npz'
    dataset_reg_path_load = sys.argv[2] # '/home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/lone_flattop_regression.npz'
    NN_path = sys.argv[3] # '/home/jovyan/vfanaskov/eigenvectors/elliptic_regression_2D/DeepONet'
    dataset_name = dataset_reg_path_load.split("/")[-1]
    dataset_reg_path_filter = "/home/jovyan/vfanaskov/elliptic_regression_2D/" + dataset_name

    header = "dataset_name,model_hash,N_basis,test_error"
    if not os.path.isfile(f'DeepONet_POD/results.csv'):
        with open(f'DeepONet_POD/results.csv', "w") as f:
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
        trunk_params = [coordinates.shape[-1], [coordinates.shape[0] + features.shape[1], args["N_encoder_trunk"], args["N_basis"]], args["N_layers_trunk"], args["kernel_size_trunk"]]
        branch_params = [[coordinates.shape[0], args["N_features_branch"], args["N_basis"]], args["N_layers_branch"]]
        model = DeepONet(trunk_params, branch_params, random.PRNGKey(33))
        model = eqx.tree_deserialise_leaves(f"{NN_path}/model_{args['hash']}.eqx", model)
        predictions_hidden = model.branch(coordinates)
        Q = jnp.linalg.svd(predictions_hidden.reshape(predictions_hidden.shape[0], -1), full_matrices=False)[2]
        N_basis = jnp.linspace(10, Q.shape[0], 10, dtype=jnp.int32)
        
        for n in N_basis:
            errors_test = []
            ind = args["N_train"] + args["N_val"] + jnp.arange(features.shape[0] - (args["N_train"] + args["N_val"]))
            for sample in ind:
                A = coo_matrix((A_data[sample], (indices[:, 0], indices[:, 1])), shape=(Q.shape[-1], Q.shape[-1])).tocsr()
                exact = sol[sample].reshape(-1,)
                b = rhs[sample].reshape(-1,)
                q = Q[:n]
                
                sol_ = np.linalg.solve((q @ (A @ q.T)), q @ b)
                sol_ = np.array(sol_ @ q)
                err = np.linalg.norm(exact - sol_) / np.linalg.norm(exact)
                errors_test.append(err)
            errors_test = jnp.array(errors_test)
            Data[f"{args['hash']}, {n}"] = errors_test
            write_data = f"\n{dataset_name},{args['hash']},{n},{jnp.mean(errors_test)}"
            with open('DeepONet_POD/results.csv', "a") as f:
                f.write(write_data)
    jnp.savez(f"DeepONet_POD/metrics_{dataset_name}", **Data)