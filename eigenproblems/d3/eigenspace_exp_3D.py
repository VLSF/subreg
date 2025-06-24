import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse

from scipy.sparse import coo_matrix
from jax import random, vmap, clear_caches
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

def stochastic_normal_loss(model, input, target, x, eps=1e-5):
    basis = model(input, x)
    basis = basis.reshape(basis.shape[0], -1)
    b = basis @ target.reshape(-1,)
    A = basis @ basis.T
    c = jnp.linalg.solve(A + jnp.eye(A.shape[0])*eps, b)
    return jnp.linalg.norm(target.reshape(-1,) - c @ basis)**2

def batch_stochastic_normal_loss(model, input, target, x):
    res = vmap(stochastic_normal_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

stochastic_normal_compute_loss_and_grads = eqx.filter_value_and_grad(batch_stochastic_normal_loss)

def stochastic_normal_make_step_scan(carry, n, optim):
    model, features, targets, x, key, opt_state = carry
    key, key_ = random.split(key)
    coeff = random.normal(key_, (n.shape[0], targets.shape[1]))
    targets_ = dot_general(targets[n], coeff, (((1,), (1,)), ((0,), (0,))))
    loss, grads = stochastic_normal_compute_loss_and_grads(model, features[n], targets_, x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, key, opt_state], loss

def compute_metrics(model, ind, features, eigenvectors, coordinates):
    predictions = model(features[ind], coordinates)
    q = jnp.linalg.qr(predictions.reshape(predictions.shape[0], -1).T)[0].T
    M = dot_general(eigenvectors[ind].reshape(eigenvectors.shape[1], -1), q, (((1,), (1,)), ((), ())))
    cosines = jnp.linalg.svdvals(M)
    errors = jnp.linalg.norm(eigenvectors[ind].reshape(eigenvectors.shape[1], -1) - M @ q, axis=1)
    return model, (cosines, errors)

def compute_eigenvec_error(model, indices, features, eigenvectors, coordinates, A_data, A_indices):
    eigvecs_errors = []
    for ind in indices:
        predictions = np.array(model(features[ind], coordinates))
        eigenvector = eigenvectors[ind].reshape(eigenvectors.shape[1], -1)
        basis = np.linalg.qr(predictions.reshape(predictions.shape[0], -1).T)[0].T
        A = coo_matrix((A_data[ind], (A_indices[:, 0], A_indices[:, 1])), shape=(basis.shape[1], basis.shape[1])).tocsr()
        A_reduced = (basis @ (A @ basis.T))
        eigenvals, eigenvecs = np.linalg.eigh(A_reduced)
        reconstructed_eigenvectors = eigenvecs.T @ basis
        reconstructed_eigenvectors = reconstructed_eigenvectors / jnp.linalg.norm(reconstructed_eigenvectors, axis=1, keepdims=True)
        errors1 = np.linalg.norm(reconstructed_eigenvectors[:eigenvector.shape[0]] - eigenvector, axis=1)
        errors2 = np.linalg.norm(reconstructed_eigenvectors[:eigenvector.shape[0]] + eigenvector, axis=1)
        errors = jnp.minimum(errors1, errors2)
        eigvecs_errors.append(errors)
    eigvecs_errors = jnp.array(eigvecs_errors)
    return eigvecs_errors

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_name": {
            "help": "name of dataset; it is expected that folder `dataet_name` is available and contain at least `dataset_name_train_0.npz`, `dataset_name_test.npz` and `dataset_name_global.npz`"
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
            "default": 10,
            "type": int,
            "help": "number of samples used to average gradient"
        },
        "-N_epoch": {
            "default": 1000,
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
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey that seed all randomness in the code"
        },
        "-N_samples": {
            "default": 100,
            "type": int,
            "help": "number of reduced eigenproblems solved to test the quality of predicted subspace"
        },
        "-train_chunks": {
            "default": 1,
            "type": int,
            "help": "if n is train chunks expecting to use `dataset_name_train_0.npz`, ..., `dataset_name_train_n-1.npz`"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    exp_hash = "".join([str(args[a]) for a in sorted(args)])
    exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,training_time,train_cosines,test_cosines,train_error,test_error,train_eigvecs_error,test_eigvecs_error"

    if not os.path.isfile("stochastic_normal/results.csv"):
        with open('stochastic_normal/results.csv', "w") as f:
            f.write(header)

    norm_factor = []
    N_samples = []
    for i in range(args["train_chunks"]):
        data = jnp.load(f"{args['dataset_name']}/{args['dataset_name']}_train_{i}.npz")
        features = data["features"]
        norm_factor_ = np.max(features)
        norm_factor.append(norm_factor_.item())
        N_samples.append(features.shape[0])
    norm_factor = max(norm_factor)
    N_train = sum(N_samples)
    N_mean = N_train / len(N_samples)
    
    global_data = jnp.load(f"{args['dataset_name']}/{args['dataset_name']}_global.npz")
    coordinates = jnp.array(global_data["coordinates"])
    A_indices = global_data["A_indices"]

    D = 3
    N_run_per_chunk = args["N_epoch"] * N_train // args["N_batch"] // args["train_chunks"]
    N_drop = args["N_drop"] * N_train // args["N_batch"]

    key = random.PRNGKey(args["key"])
    keys = random.split(key, 3)

    N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], args["N_subspace"]]
    model = FFNO_normalised(args["N_layers"], N_features, args["N_modes"], D, keys[0])

    model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
    learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
    optim = optax.lion(learning_rate=learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    key_ = keys[1]
    History = []
    for i in range(args["train_chunks"]):
        data = jnp.load(f"{args['dataset_name']}/{args['dataset_name']}_train_{i}.npz")
        features = jnp.array(data["features"]) / norm_factor
        eigenvectors = jnp.array(data["eigenvectors"])[:, :args["N_eig"]]
        key_, key_n = random.split(key_)
        n = random.choice(key_n, features.shape[0], shape = (N_run_per_chunk, args["N_batch"]))
        
        carry = [model, features, eigenvectors, coordinates, key_n, opt_state]
        make_step_scan_ = lambda a, b: stochastic_normal_make_step_scan(a, b, optim)
        start = time.time()
        carry, history = scan(make_step_scan_, carry, n)
        stop = time.time()
        training_time = stop - start
        model = carry[0]
        opt_state = carry[-1]
        History.append(history)
    History = jnp.concatenate(History, 0)
    
    eqx.tree_serialise_leaves(f'stochastic_normal/model_{exp_hash}.eqx', model)
    eqx.tree_serialise_leaves(f'stochastic_normal/opt_state_{exp_hash}.eqx', opt_state)

    # compute metrics
    train_cosines = []
    train_errors = []
    for i in range(args["train_chunks"]):
        data = jnp.load(f"{args['dataset_name']}/{args['dataset_name']}_train_{i}.npz")
        features = jnp.array(data["features"]) / norm_factor
        eigenvectors = jnp.array(data["eigenvectors"])[:, :args["N_eig"]]
        compute_metrics_ = lambda a, b: compute_metrics(a, b, features, eigenvectors, coordinates)
        _, (cosines, errors) = scan(compute_metrics_, model, jnp.arange(features.shape[0]))
        train_cosines.append(cosines)
        train_errors.append(errors)
        clear_caches()
    train_cosines = jnp.concatenate(train_cosines, axis=0)
    train_errors = jnp.concatenate(train_errors, axis=0)
    
    ind_ = jnp.arange(args["N_samples"])
    A_data = data["A_data"]
    train_eigvecs_errors = compute_eigenvec_error(model, ind_, features, eigenvectors, coordinates, A_data, A_indices)

    data = jnp.load(f"{args['dataset_name']}/{args['dataset_name']}_test.npz")
    features = jnp.array(data["features"]) / norm_factor
    eigenvectors = jnp.array(data["eigenvectors"])[:, :args["N_eig"]]
    compute_metrics_ = lambda a, b: compute_metrics(a, b, features, eigenvectors, coordinates)
    _, (test_cosines, test_errors) = scan(compute_metrics_, model, jnp.arange(features.shape[0]))

    A_data = data["A_data"]
    test_eigvecs_errors = compute_eigenvec_error(model, ind_, features, eigenvectors, coordinates, A_data, A_indices)
    
    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{History[-1]},{model_size},{training_time},{jnp.mean(train_cosines)},{jnp.mean(test_cosines)},{jnp.mean(train_errors)},{jnp.mean(test_errors)},{jnp.mean(train_eigvecs_errors)},{jnp.mean(test_eigvecs_errors)}"

    with open(f'stochastic_normal/results.csv', "a") as f:
        f.write(data)

    save_data = {
        "train_cosines": train_cosines,
        "train_error": train_errors,
        "train_eigvecs_error": train_eigvecs_errors,
        "test_cosines": test_cosines,
        "test_error": test_errors,
        "test_eigvecs_error": test_eigvecs_errors,
        "history": History,
    }
    
    jnp.savez(f'stochastic_normal/metrics_{exp_hash}.npz', **save_data)
    