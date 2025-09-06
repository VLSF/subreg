import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse

from scipy.sparse import coo_matrix, diags
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


def stochastic_normal2_loss(model, input, target, x, eps=1e-4):
    target = target.reshape(-1,)
    basis = model(input, x)
    basis = basis.reshape(basis.shape[0], -1)
    
    G = basis @ basis.T
    G = G + jnp.eye(G.shape[0])*eps
    P = jnp.linalg.cholesky(G, upper=True)
    basis = jnp.linalg.inv(P.T) @ basis
    
    b = basis @ target
    A = basis @ basis.T
    sol = jnp.linalg.solve(A, b) @ basis
    return jnp.linalg.norm(target - sol)**2

def batch_stochastic_normal2_loss(model, input, target, x):
    res = vmap(stochastic_normal2_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

stochastic_normal2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_stochastic_normal2_loss)

def stochastic_normal2_make_step_scan(carry, n, optim):
    model, features, targets, x, key, opt_state = carry
    key, key_ = random.split(key)
    coeff = random.normal(key_, [n.shape[0], targets.shape[1]] + [1,]*(features.ndim - 2))
    targets_ = jnp.sum(targets[n]*coeff, axis=1)
    loss, grads = stochastic_normal2_compute_loss_and_grads(model, features[n], targets_, x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, key, opt_state], loss

def stochastic_normal_loss(model, input, target, x, eps=1e-4):
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
    coeff = random.normal(key_, [n.shape[0], targets.shape[1]] + [1,]*(features.ndim - 2))
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

def l2_loss(model, input, target, x):
    X = model(input, x)
    error1 = jnp.sum(((X - target).reshape(target.shape[0], -1,))**2, axis=1)
    error2 = jnp.sum(((X + target).reshape(target.shape[0], -1,))**2, axis=1)
    error = jnp.mean(jnp.minimum(error1, error2))
    return error

def batch_l2_loss(model, input, target, x):
    res = vmap(l2_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

l2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def l2_make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = l2_compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss

def make_prediction_scan(carry, i):
    model, features, coords = carry
    prediction = model(features[i], coords)
    Q = jnp.linalg.qr(prediction.reshape(prediction.shape[0], -1).T)[0].T
    return carry, Q

def get_laplacian_1D(N_x, h):
    L = -2*np.eye(N_x) + np.eye(N_x, k=1) + np.eye(N_x, k=-1)
    return L / h**2

def get_laplacian_2D(N_x, h):
    L = -2*np.eye(N_x) + np.eye(N_x, k=1) + np.eye(N_x, k=-1)
    I = np.eye(N_x)
    L = (np.kron(I, L) + np.kron(L, I)) / h**2
    L = coo_matrix(L).tocsr()
    return L

def compute_eigenvectors(L, V, P):
    A = - P @ (L @ P.T) + (P * np.expand_dims(V, 0)) @ P.T
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvecs = eigvecs.T @ P
    return eigvals, eigvecs

def compute_metrics(model, features, coordinates, V, basis, L):
    Q = scan(make_prediction_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
    M = dot_general(Q, basis.reshape(basis.shape[0], basis.shape[1], -1), (((2,), (2,)), ((0,), (0,))))
    cosines = scan(lambda a, b: (a, jnp.linalg.svdvals(b)), None, M)[1]
    
    sol = jnp.moveaxis(dot_general(Q, M, (((1,), (1,)), ((0,), (0,)))), 1, 2)
    Errors = jnp.linalg.norm(sol - basis.reshape(basis.shape[0], basis.shape[1], -1), axis=2)

    Q = np.array(Q)
    eigvecs_errors = []
    for sample in range(features.shape[0]):
        reconstructed_eigenvectors = compute_eigenvectors(L, V[sample], Q[sample])[1]
        errors1 = np.linalg.norm(reconstructed_eigenvectors[:basis.shape[1]] - basis[sample].reshape(basis.shape[1], -1), axis=1)
        errors2 = np.linalg.norm(reconstructed_eigenvectors[:basis.shape[1]] + basis[sample].reshape(basis.shape[1], -1), axis=1)
        errors = jnp.minimum(errors1, errors2)
        eigvecs_errors.append(errors)
    eigvecs_errors = jnp.array(eigvecs_errors)
    return cosines, Errors, eigvecs_errors
    
def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_path": {
            "help": "absolute path to dataset"
        },
       "-results_path": {
            "help": "absolute path to folder where results are stored"
        },  
        "-experiment_type": {
            "default": "subspace",
            "type": str,
            "choices": ["l2", "stochastic_normal", "subspace", "stochastic_normal2"],
            "help": "experiment type: defines loss, architecture and how subspace is predicted"
        },       
        "-learning_rate": {
            "default": 1e-3,
            "type": float,
            "help": "learning rate"
        },
        "-gamma": {
            "default": 0.9,
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
            "default": 800,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_val": {
            "default": 100,
            "type": int,
            "help": "number of samples in the validation set"
        },
        "-N_epoch": {
            "default": 1000,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch"
        },
        "-stop_each": {
            "default": 50,
            "type": int,
            "help": "stop each N_epoch to evaluate the model and make checkpoint"
        },
        "-N_processor": {
            "default": 64,
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-N_basis": {
            "default": 10,
            "type": int,
            "help": "number of basis bectors to predict"
        },
        "-N_subspace": {
            "default": 10,
            "type": int,
            "help": "dimension of the subspace used to approximate basis"
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
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    dataset_name = args['dataset_path'].split("/")[-1]
    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,training_time,train_cosines,val_cosines,test_cosines,train_error,val_errors,test_error,train_eigvecs_error,val_eigvecs_error,test_eigvecs_error,best_n"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)

    data = jnp.load(args['dataset_path'])
    
    basis = jnp.array(data["eigenvectors"])[:, :args["N_basis"]]
    D = basis.ndim - 2
    coordinates_x = jnp.array(data["coordinates"])
    if D == 1:
        coordinates = jnp.expand_dims(jnp.linspace(0, 1, coordinates_x.shape[0]), 0)
        h = coordinates_x[1] - coordinates_x[0]
        L = get_laplacian_1D(coordinates_x.shape[-1], h)
    elif D == 2:
        coordinates = coordinates_x
        h = coordinates_x[0, 0, 1] - coordinates_x[0, 0, 0]
        L = get_laplacian_2D(coordinates_x.shape[-1], h)
    features = jnp.expand_dims(jnp.array(data["potentials"]), 1)
    V = (features[:, 0] + 0.0).reshape(features.shape[0], -1)
    reduced_axes = [0, 2, 3] if D == 2 else [0, 2]
    norm_factor = jnp.max(jnp.abs(features), axis=reduced_axes, keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    features = features / norm_factor

    for N_drop_ in [100, 200]:
        for N_modes_ in [16, 14, 10]:
            for N_layers_ in [5, 4, 3]:
                for learning_rate_ in [1e-3, 1e-4]:
                    args["N_drop"] = N_drop_
                    args["N_modes"] = N_modes_
                    args["N_layers"] = N_layers_
                    args["learning_rate"] = learning_rate_
                    exp_hash = "".join([str(args[a]) for a in sorted(args)])
                    exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
                
                    N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
                    N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
                    N_stop = args["stop_each"] * args["N_train"] // args["N_batch"]
                
                    key = random.PRNGKey(args["key"])
                    keys = random.split(key, 3)
                    
                    if args["experiment_type"] == "l2":
                        N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], args["N_basis"]]
                        model = FFNO_normalised(args["N_layers"], N_features, args["N_modes"], D, keys[0])
                    else:
                        N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], args["N_subspace"]]
                        model = FFNO_normalised(args["N_layers"], N_features, args["N_modes"], D, keys[0])
                        
                    model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
                    learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
                    optim = optax.lion(learning_rate=learning_rate)
                    opt_state = optim.init(eqx.filter(model, eqx.is_array))
                    
                    nn = random.choice(keys[1], args["N_train"], shape = (N_run//N_stop, N_stop, args["N_batch"]))
                    if args["experiment_type"] == "stochastic_normal" or args["experiment_type"] == "stochastic_normal2":
                        carry = [model, features, basis, coordinates, keys[1], opt_state]
                    else:
                        carry = [model, features, basis, coordinates, opt_state]
                
                    if args["experiment_type"] == "l2":
                        make_step_scan_ = lambda a, b: l2_make_step_scan(a, b, optim)
                    elif args["experiment_type"] == "stochastic_normal":
                        make_step_scan_ = lambda a, b: stochastic_normal_make_step_scan(a, b, optim)
                    elif args["experiment_type"] == "stochastic_normal2":
                        make_step_scan_ = lambda a, b: stochastic_normal2_make_step_scan(a, b, optim)
                    elif args["experiment_type"] == "subspace":
                        make_step_scan_ = lambda a, b: subspace_make_step_scan(a, b, optim)
                        
                    training_time = 0
                    models = []
                    opt_states = []
                    val_rel_errors = []
                    training_times = []
                    histories = []
                    for nn_ in nn:
                        start = time.time()
                        carry, history = scan(make_step_scan_, carry, nn_)
                        stop = time.time()
                        training_time = training_time + stop - start
                        model = carry[0]
                        opt_state = carry[-1]
                        models.append(model)
                        opt_states.append(opt_state)
                        rel_errors = compute_metrics(model, features[args["N_train"]:(args["N_train"]+args["N_val"])], coordinates, V[args["N_train"]:(args["N_train"]+args["N_val"])], basis[args["N_train"]:(args["N_train"]+args["N_val"])], L)[2]
                        val_rel_errors.append(jnp.mean(rel_errors))
                        training_times.append(training_time)
                        histories.append(history)
                        if jnp.isnan(val_rel_errors[-1]).item():
                            break
                    val_rel_errors = jnp.array(val_rel_errors)
                    val_rel_errors = jnp.nan_to_num(val_rel_errors, nan=jnp.inf)
                    best_n = jnp.argmin(val_rel_errors)
                    concat_n = min(best_n + 1, len(histories))
                    history = jnp.concatenate(histories[:concat_n])
                    eqx.tree_serialise_leaves(f'{args["results_path"]}/model_{exp_hash}.eqx', models[best_n])
                    eqx.tree_serialise_leaves(f'{args["results_path"]}/opt_state_{exp_hash}.eqx', opt_states[best_n])
                
                    cosines, errors, rel_errors = compute_metrics(models[best_n], features, coordinates, V, basis, L)
                    
                    train_cosines = jnp.mean(cosines[:args["N_train"]])
                    val_cosines = jnp.mean(cosines[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_cosines = jnp.mean(cosines[(args["N_train"]+args["N_val"]):])
                    
                    train_errors = jnp.mean(errors[:args["N_train"]])
                    val_errors = jnp.mean(errors[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_errors = jnp.mean(errors[(args["N_train"]+args["N_val"]):])
                    
                    train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
                    val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])
                
                    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
                    data += f",{exp_hash},{history[-1]},{model_size},{training_time},{train_cosines},{val_cosines},{test_cosines},{train_errors},{val_errors},{test_errors},{train_rel_errors},{val_rel_errors},{test_rel_errors},{best_n}"
                
                    with open(f'{args["results_path"]}/results.csv', "a") as f:
                        f.write(data)
                    
                    jnp.savez(f'{args["results_path"]}/metrics_{exp_hash}.npz', errors=rel_errors, history=history)