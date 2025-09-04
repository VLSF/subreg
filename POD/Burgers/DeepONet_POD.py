import os
import sys
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix
from jax import random, vmap
from jax.lax import scan, dot_general, dynamic_slice_in_dim
from jax.tree_util import tree_map, tree_flatten
from jax.nn import gelu

class conv_trunk_1D(eqx.Module):
    convs: list
    projector: list
    encoder: list

    def __init__(self, N_x, N_features, N_layers, kernel_size, key):
        N_in, N_encoder, N_out = N_features
        N_layers = min(jnp.log2(N_x).astype(int).item(), N_layers)
        keys = random.split(key, (N_layers, 2))
        self.convs = []
        N = N_encoder
        x = jnp.zeros((N_encoder, N_x))
        for key in keys:
            c = [
                eqx.nn.Conv(1, N, 2*N, kernel_size = kernel_size, padding = 'SAME', key = key[0]),
                eqx.nn.Conv(1, 2*N, 2*N, kernel_size = kernel_size, stride=2, key = key[1])
            ]
            x = c[1](c[0](x))
            N = 2*N
            self.convs.append(c)
        keys = random.split(keys[-1, -1])
        self.projector = [random.normal(keys[0], (N_out, x.shape[1]*N)) / jnp.sqrt(N_out + x.shape[1]*N//2), jnp.zeros((N_out,))]
        self.encoder = [jnp.zeros((N_encoder, 1)), random.normal(keys[1], (N_encoder, N_in)) / jnp.sqrt(N_encoder + N_in)]

    def __call__(self, u, coords):
        # u.shape = (N_features, N_x); coords.shape = (1, N_x)
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
        self.trunk = conv_trunk_1D(N_x, N_features, N_layers, kernel_size, keys[0])
        self.branch = MLP(N_features_, N_layers_, keys[1])

    def __call__(self, u, coords, coords_x):
        coeff = self.trunk(u, coords)
        phi = self.branch(coords_x)
        res = jnp.expand_dims(dot_general(phi, coeff, (((0,), (0,)), ((), ()))), 0)
        return res

def l2_loss(model, input, target, x, coords):
    X = model(input, x, coords).reshape(target.shape[0], -1)
    error = jnp.mean(jnp.sum((X - target.reshape(target.shape[0], -1))**2, axis=1))
    return error

def batch_l2_loss(model, input, target, x, coords):
    res = vmap(l2_loss, in_axes=(None, 0, 0, None, None))(model, input, target, x, coords)
    return jnp.mean(res)

l2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def l2_make_step_scan(carry, n, optim):
    model, features, targets, x, coords, opt_state = carry
    loss, grads = l2_compute_loss_and_grads(model, features[n], targets[n], x, coords)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, coords, opt_state], loss

def make_prediction_scan(carry, i):
    model, features, x, coords = carry
    prediction = model(features[i], x, coords)
    return carry, prediction

def make_prediction_hidden_scan(carry, i):
    model, features, x, coords = carry
    prediction = model.branch(coords)
    return carry, prediction

def DeepPOD_loss(model, input, target, x):
    Q = model(input, x)
    Q = Q.reshape(Q.shape[0], -1)
    Q = jnp.linalg.qr(Q.T)[0]
    target_pr = (target.reshape(-1, target.shape[-1]) @ Q) @ Q.T
    e = jnp.sum((target.reshape(-1, target.shape[-1]) - target_pr)**2)
    return e
    
def batch_DeepPOD_loss(model, input, target, x):
    res = vmap(DeepPOD_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

DeepPOD_compute_loss_and_grads = eqx.filter_value_and_grad(batch_DeepPOD_loss)

def DeepPOD_make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = DeepPOD_compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss

def get_coordinates(N):
    x = jnp.linspace(0, 1, N+2)[1:-1]
    x_extended = jnp.linspace(0, 1, 2*N + 3)[1:-1]
    return x, x_extended

def get_A(a_extended):
    A = jnp.diag(-a_extended[::2][:-1]-a_extended[::2][1:]) + jnp.diag(a_extended[::2][1:-1], k=-1) + jnp.diag(a_extended[::2][1:-1], k=+1)
    return A

def get_B(N):
    B = jnp.diag(jnp.ones((N-1,))/2, k=+1) - jnp.diag(jnp.ones((N-1,))/2, k=-1)
    B = B.at[0, :3].set(jnp.array([-3/2, 2, -1/2]))
    B = B.at[-1, -3:].set(jnp.array([1/2, -2, 3/2]))
    return B

def get_reduced_matrices(A, B, W):
    A_ = W @ A @ W.T
    B_ = W @ B
    return A_, B_

def integration_step_pod(u, f, A_, B_, W, N_steps):
    v = jnp.copy(u)
    for _ in range(N_steps):
        r = B_ @ (W.T @ v)**2 / 2 + A_ @ v - u + f
        v = v - jnp.linalg.solve(B_ @ jnp.diag(W.T @ v) @ W.T + A_, r)
    return v

def integrate_Burgers_pod(u, f, a_extended, B, W, N_newton, t, h):
    u_ = W @ u
    f_ = W @ f
    A = get_A(a_extended)
    A = jnp.eye(B.shape[1]) - (t[1] - t[0]) * get_A(a_extended) / h**2
    A_, B_ = get_reduced_matrices(A, B, W)
    def integration_step_(u, t):
        u = integration_step_pod(u, f_, A_, B_, W, N_newton)
        return u, u

    _, U = scan(integration_step_, u_, t)
    U = U @ W
    U = jnp.concatenate([jnp.expand_dims(u, 0), U], axis=0)
    return U

def compute_metrics(Q, N_basis, u0, forcing, targets, coordinates, a_extended, T_max=0.1, N_newton=3):
    N_t = targets.shape[2] - 1
    N = targets.shape[-1]
    t = jnp.linspace(0, T_max, N_t)
    dt = t[1] - t[0]
    h = coordinates[0, 1] - coordinates[0, 0]
    B = dt * get_B(N) / h

    res = vmap(integrate_Burgers_pod, in_axes=(0, 0, 0, None, None, None, None, None))(u0, forcing, a_extended, B, Q[:N_basis], N_newton, t, h)
    rel_errors = vmap(jnp.linalg.norm)(targets[:, 0] - res) / vmap(jnp.linalg.norm)(targets[:, 0])
    return rel_errors
    
if __name__ == "__main__":
    cut_off = 1e1
    dataset_path = sys.argv[1]#"/home/jovyan/vfanaskov/eigenvectors/Burgers_1p1/Burgers_III.npz"
    dataset_name = dataset_path.split("/")[-1][:-4]
    
    NN_path = sys.argv[2]#"/home/jovyan/vfanaskov/eigenvectors/Burgers_1p1/grid_search_Burgers_regression_es/DeepONet"
    Args = pd.read_csv(f"{NN_path}/results.csv")

    data = np.load(dataset_path)
    features = jnp.concatenate([data["solutions"][:, :1], jnp.expand_dims(data["diff"][:, 1::2], 1), jnp.expand_dims(data["forcing"], 1)], 1)
    a_extended = jnp.array(data["diff"])
    forcing = jnp.array(data["forcing"])
    
    targets = jnp.expand_dims(jnp.array(data["solutions"]), 1)
    Targets = targets + 0.0
    norm_factor = jnp.max(jnp.abs(targets), axis=[0, 2, 3], keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    targets = targets / norm_factor
    x = jnp.array(data["coords"])
    t = jnp.linspace(0, 1, targets.shape[2])
    coordinates = jnp.stack(jnp.meshgrid(t, x, indexing="ij"), 0)
    c = jnp.expand_dims(jnp.array(data["coords"]), axis=0)
    x = jnp.expand_dims(x, 0)
    
    norm_factor = jnp.max(jnp.abs(features), axis=[0, 2], keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    features = features / norm_factor
    u0 = jnp.array(data["solutions"][:, 0])
    
    args_ = Args[Args["dataset_name"] == dataset_name.split("_")[-1]]

    header = "dataset_name,model_hash,N_basis,train_error,validation_error,test_error"
    if not os.path.isfile(f'D_POD/results.csv'):
        with open(f'DeepONet_POD/results.csv', "w") as f:
            f.write(header)

    Data = dict()
    for i in range(3):
        args = args_.sort_values("val_error").iloc[i]        
        D = features.ndim - 2
        trunk_params = [x.shape[1], [x.shape[0] + features.shape[1], args["N_encoder_trunk"], args["N_basis"]], args["N_layers_trunk"], args["kernel_size_trunk"]]
        branch_params = [[coordinates.shape[0], args["N_features_branch"], args["N_basis"]], args["N_layers_branch"]]
        model = DeepONet(trunk_params, branch_params, random.PRNGKey(33))
        model = eqx.tree_deserialise_leaves(f"{NN_path}/model_{args['hash']}.eqx", model)
        
        predictions_hidden = scan(make_prediction_hidden_scan, [model, features, x, coordinates], jnp.arange(1))[1]
        _, Q = scan(lambda a, b: (a, (jnp.linalg.svd(b.reshape(-1, b.shape[-1]), full_matrices=False)[2])), None, predictions_hidden)
        N_basis = jnp.linspace(10, Q.shape[-1], 10, dtype=jnp.int32)

        for n in N_basis:
            rel_errors = compute_metrics(Q[0], n, u0, forcing, Targets, c, a_extended)
            Data[f"{args['hash']}, {n}"] = rel_errors
            rel_errors = rel_errors[rel_errors < cut_off]
            train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
            val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
            test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])
            write_data = f"\n{dataset_name},{args['hash']},{n},{jnp.mean(train_rel_errors)},{jnp.mean(val_rel_errors)},{jnp.mean(test_rel_errors)}"
            with open('DeepONet_POD/results.csv', "a") as f:
                f.write(write_data)
        
    jnp.savez("DeepONet_POD/metrics.npz", **Data)