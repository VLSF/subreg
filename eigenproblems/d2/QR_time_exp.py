import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse

from scipy.sparse import coo_matrix
from jax import random, vmap, jit
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

def stochastic_normal_loss(model, input, target, x, eps=1e-3):
    target = target.reshape(-1,)
    basis = model(input, x)
    basis = basis.reshape(basis.shape[0], -1)
    b = basis @ target
    A = basis @ basis.T
    sol = jnp.linalg.solve(A + jnp.eye(A.shape[0])*eps, b) @ basis
    return jnp.linalg.norm(target - sol)**2

def batch_stochastic_normal_loss(model, input, target, x):
    res = vmap(stochastic_normal_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

stochastic_normal_compute_loss_and_grads = eqx.filter_value_and_grad(batch_stochastic_normal_loss)

def stochastic_normal_make_step_scan(carry, n, optim):
    model, features, targets, x, key, opt_state = carry
    key, key_ = random.split(key)
    coeff = random.normal(key_, (n.shape[0], targets.shape[1], 1, 1))
    targets_ = jnp.sum(targets[n]*coeff, axis=1)
    loss, grads = stochastic_normal_compute_loss_and_grads(model, features[n], targets_, x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, key, opt_state], loss

def subspace_loss(model, input, target, x):
    Q = model(input, x)
    Q = Q.reshape(Q.shape[0], -1)
    Q = jnp.linalg.qr(Q.T)[0]
    return target.shape[0] - jnp.sum((target.reshape(target.shape[0], -1) @ Q)**2)

def batch_subspace_loss(model, input, target, x):
    res = vmap(subspace_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

subspace_compute_loss_and_grads = eqx.filter_value_and_grad(batch_subspace_loss)

def subspace_make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = subspace_compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss

@jit
def modified_GS(Q):
    n = Q.shape[0]
    for i in range(n):
        for j in range(i):
            Q = Q.at[i].set(Q[i] - (Q[i] @ Q[j]) * Q[j])
        Q = Q.at[i].set(Q[i] / jnp.linalg.norm(Q[i]))
    return Q

def subspace_gs_loss(model, input, target, x):
    Q = model(input, x)
    Q = Q.reshape(Q.shape[0], -1)
    Q = modified_GS(Q).T
    return target.shape[0] - jnp.sum((target.reshape(target.shape[0], -1) @ Q)**2)

def batch_subspace_gs_loss(model, input, target, x):
    res = vmap(subspace_gs_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

subspace_gs_compute_loss_and_grads = eqx.filter_value_and_grad(batch_subspace_gs_loss)

def subspace_gs_make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = subspace_gs_compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss

@jit
def qr_decomposition(A):
    n, m = A.shape
    def update(carry, j):
        Q, R = carry
        v = A[:, j]
        
        mask = jnp.arange(m) < j
        R_col = jnp.where(mask, jnp.dot(Q.T, A[:, j]), 0.0)
        v = v - Q @ jnp.where(mask, R_col, 0.0)
        
        r_jj = jnp.linalg.norm(v)
        r_jj_safe = jnp.maximum(r_jj, 1e-8)
        
        Q = Q.at[:, j].set(v / r_jj_safe)
        R = R.at[:, j].set(R_col)
        R = R.at[j, j].set(r_jj)
        return (Q, R), None
    
    Q = jnp.zeros((n, m))
    R = jnp.zeros((m, m))
    (Q, R), _ = scan(update, (Q, R), jnp.arange(m))
    return Q, R

def subspace_A_loss(model, input, target, x):
    Q = model(input, x)
    Q = Q.reshape(Q.shape[0], -1).T
    Q = qr_decomposition(Q)[0]
    return target.shape[0] - jnp.sum((target.reshape(target.shape[0], -1) @ Q)**2)

def batch_subspace_A_loss(model, input, target, x):
    res = vmap(subspace_A_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

subspace_A_compute_loss_and_grads = eqx.filter_value_and_grad(batch_subspace_A_loss)

def subspace_A_make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = subspace_A_compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-path_to_dataset": {
           "default": "twin_flattops.npz",
            "help": "path to dataset in the .npz format"
        },
        "-experiment_type": {
            "default": "stochastic_normal",
            "type": str,
            "choices": ["stochastic_normal", "subspace", "subspace_gs", "subspace_A"],
            "help": "experiment type: defines loss, architecture and how subspace is predicted"
        },       
        "-learning_rate": {
            "default": 1e-4,
            "type": float,
            "help": "learning rate"
        },
        "-gamma": {
            "default": 0.5,
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate per N_drop epochs"
        },
        "-N_drop": {
            "default": 100,
            "type": int,
            "help": "multiply learning rate by gamma each N_drop epoch"
        },
        "-N_batch": {
            "default": 100,
            "type": int,
            "help": "number of samples used to average gradient"
        },
        "-N_train": {
            "default": 4000,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_epoch": {
            "default": 2,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch"
        },
        "-N_processor": {
            "default": 64,
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-N_eig": {
            "default": 10,
            "type": int,
            "help": "number of eigenvalues to predict"
        },
        "-N_subspace": {
            "default": 10,
            "type": int,
            "help": "dimension of the subspace used to approximate eigenvalues"
        },
        "-N_layers": {
            "default": 4,
            "type": int,
            "help": "number of layers"
        },
        "-N_modes": {
            "default": 10,
            "type": int,
            "help": "number of basis function"
        },
        "-N_samples": {
            "default": 200,
            "type": int,
            "help": "number of reduced eigenproblems solved to test the quality of predicted subspace"
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey that seed all randomness in the code"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    
    header = ",".join([key for key in args.keys()])
    header += ",final_loss,model_size,training_time"

    data = jnp.load(args["path_to_dataset"])
    eigenvectors = jnp.array(data["eigenvectors"])
    features = jnp.array(data["features"])
    features = features / jnp.max(jnp.abs(features), axis=[0, 2, 3], keepdims=True)
    coordinates = jnp.array(data["coordinates"])

    D = features.ndim - 2
    N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
    N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]

    key = random.PRNGKey(args["key"])
    keys = random.split(key, 3)

    Exps = ["stochastic_normal", "subspace", "subspace_A"]
    for N_subspace in [10, 20, 30, 40, 50, 60]:
        args["N_subspace"] = N_subspace 
        for experiment in  Exps:
            args["experiment_type"] = experiment
            
            if not os.path.isfile(f'time_exp/results.csv'):
                with open(f'time_exp/results.csv', "w") as f:
                    f.write(header)
            
            N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], args["N_subspace"]]
            model = FFNO_normalised(args["N_layers"], N_features, args["N_modes"], D, keys[0])
                
            model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
            learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
            optim = optax.lion(learning_rate=learning_rate)
            opt_state = optim.init(eqx.filter(model, eqx.is_array))
            
            n = random.choice(keys[1], args["N_train"], shape = (N_run, args["N_batch"]))
            if args["experiment_type"] == "stochastic_normal":
                carry = [model, features, eigenvectors[:, :args["N_eig"]], coordinates, keys[1], opt_state]
            else:
                carry = [model, features, eigenvectors[:, :args["N_eig"]], coordinates, opt_state]
        
            if args["experiment_type"] == "stochastic_normal":
                make_step_scan_ = lambda a, b: stochastic_normal_make_step_scan(a, b, optim)
            elif args["experiment_type"] == "subspace":
                make_step_scan_ = lambda a, b: subspace_make_step_scan(a, b, optim)
            elif args["experiment_type"] == "subspace_gs":
                make_step_scan_ = lambda a, b: subspace_gs_make_step_scan(a, b, optim)
            elif args["experiment_type"] == "subspace_A":
                make_step_scan_ = lambda a, b: subspace_A_make_step_scan(a, b, optim)

            scan(make_step_scan_, carry, n)[1].block_until_ready()
            
            start = time.time()
            history = scan(make_step_scan_, carry, n)[1].block_until_ready()
            stop = time.time()
            training_time = stop - start
        
            data = "\n" + ",".join([str(args[key]) for key in args.keys()])
            data += f",{history[-1]},{model_size},{training_time}"
        
            with open(f'time_exp/results.csv', "a") as f:
                f.write(data)