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

def make_prediction_hidden_scan(carry, i):
    model, features, coords = carry
    prediction = model.comput_hidden(features[i], coords)
    return carry, prediction

def make_prediction_scan(carry, i):
    model, features, coords = carry
    prediction = model(features[i], coords)
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

    res = vmap(integrate_Burgers_pod, in_axes=(0, 0, 0, None, 0, None, None, None))(u0, forcing, a_extended, B, Q[:, :N_basis], N_newton, t, h)
    rel_errors = vmap(jnp.linalg.norm)(targets[:, 0] - res) / vmap(jnp.linalg.norm)(targets[:, 0])
    return rel_errors

if __name__ == "__main__":
    cut_off = 1e1
    dataset_path = sys.argv[1]#"/home/jovyan/vfanaskov/eigenvectors/Burgers_1p1/Burgers_III.npz"
    dataset_name = dataset_path.split("/")[-1][:-4]
    
    NN_path = sys.argv[2]#"/home/jovyan/vfanaskov/eigenvectors/Burgers_1p1/grid_search_Burgers_regression_es/FFNO"
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
    norm_factor = jnp.max(jnp.abs(features), axis=[0, 2], keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    features = features / norm_factor
    features = jnp.stack([features,]*coordinates.shape[1], axis=2)
    u0 = jnp.array(data["solutions"][:, 0])
    
    args_ = Args[Args["dataset_name"] == dataset_name.split("_")[-1]]

    header = "dataset_name,model_hash,N_basis,train_error,validation_error,test_error"
    if not os.path.isfile(f'FFNO_POD/results.csv'):
        with open(f'FFNO_POD/results.csv', "w") as f:
            f.write(header)

    Data = dict()
    for i in range(3):
        args = args_.sort_values("val_error").iloc[i]        
        D = features.ndim - 2
        N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], targets.shape[1]]
        model = FFNO(args["N_layers"], N_features, args["N_modes"], D, random.PRNGKey(33))
        model = eqx.tree_deserialise_leaves(f"{NN_path}/model_{args['hash']}.eqx", model)
        predictions_hidden = scan(make_prediction_hidden_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
        _, Q = scan(lambda a, b: (a, (jnp.linalg.svd(b.reshape(-1, b.shape[-1]), full_matrices=False)[2])), None, predictions_hidden)
        N_basis = jnp.linspace(10, Q.shape[-1], 10, dtype=jnp.int32)

        for n in N_basis:
            rel_errors = compute_metrics(Q, n, u0, forcing, Targets, c, a_extended)
            Data[f"{args['hash']}, {n}"] = rel_errors
            rel_errors = rel_errors[rel_errors < cut_off]
            train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
            val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
            test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])
            write_data = f"\n{dataset_name},{args['hash']},{n},{jnp.mean(train_rel_errors)},{jnp.mean(val_rel_errors)},{jnp.mean(test_rel_errors)}"
            with open('FFNO_POD/results.csv', "a") as f:
                f.write(write_data)
        
    jnp.savez("FFNO_POD/metrics.npz", **Data)