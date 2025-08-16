import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse

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
    coeff = random.normal(key_, (n.shape[0], targets.shape[1], 1))
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
    return carry, prediction

def diffusion_1D(dif_coef):
    h = 1 / dif_coef.shape[0]
    main_diag = -(dif_coef[:-1] + dif_coef[1:])
    off_diag = dif_coef[1:-1]
    A = (jnp.diag(main_diag) + jnp.diag(off_diag, k=-1) + jnp.diag(off_diag, k=1)) / h**2
    return A

def convection_1D(conv_coef):
    h = 1 / (conv_coef.shape[0] + 1)
    B = (jnp.diag(conv_coef) - jnp.diag(conv_coef[1:], k=-1)) / (2*h)
    return B

def get_operator(convection, diffusion, reaction, dt):
    A = diffusion_1D(diffusion)
    B = convection_1D(convection)
    S = jnp.diag(reaction)
    M = (jnp.eye(S.shape[0]) - dt * (A - B + S))
    return M

def integrate(carry, t):
    M, u, s = carry # supply s*dt
    u = jnp.linalg.solve(M, u + s)
    return [M, u, s], u

def get_sample(carry, ind):
    P, u0, diffusion, convection, reaction, source, t, dt = carry
    M = P[ind].T @ get_operator(convection[ind], diffusion[ind, ::2], reaction[ind], dt) @ P[ind]
    carry_ = [M, P[ind].T @ u0[ind], P[ind].T @ source[ind]*dt]
    
    carry_, U = scan(integrate, carry_, t[1:])
    U = jnp.concatenate([u0[ind].reshape(1, -1), U @ P[ind].T], axis=0)
    return carry, U

def get_errors(POD_basis, data):
    # POD_basis here should have shape (N_samples, N_x, N_basis)
    t = data["t"]
    dt = t[1] - t[0]
    carry = [POD_basis, data["solutions"][:, 0, :], data["diffusion"], data["convection"], data["reaction"], data["source"], t, dt]
    _, sol_approximate = scan(get_sample, carry, jnp.arange(data["solutions"].shape[0]))
    errors= vmap(jnp.linalg.norm)(data["solutions"] - sol_approximate) / vmap(jnp.linalg.norm)(data["solutions"])
    return errors

def compute_metrics(model, features, coordinates, data):
    predictions = scan(make_prediction_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]

    Q = vmap(lambda x: jnp.linalg.qr(x.T)[0].T)(predictions.reshape(predictions.shape[0], predictions.shape[1], -1))
    M = dot_general(Q, basis, (((2,), (2,)), ((0,), (0,))))
    cosines = vmap(jnp.linalg.svdvals)(M)
    
    sol = jnp.moveaxis(dot_general(Q, M, (((1,), (1,)), ((0,), (0,)))), 1, 2)
    Errors = jnp.linalg.norm(sol - basis, axis=2)

    rel_errors = get_errors(jnp.transpose(Q, (0, 2, 1)), data)
    return cosines, Errors, rel_errors
    
def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
        "-experiment_type": {
            "default": "stochastic_normal",
            "type": str,
            "choices": ["l2", "stochastic_normal", "subspace"],
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
            "default": 900,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_epoch": {
            "default": 10,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch"
        },
        "-N_processor": {
            "default": 64,
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-N_basis": {
            "default": 5,
            "type": int,
            "help": "number of basis bectors to predict"
        },
        "-N_subspace": {
            "default": 5,
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
    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,training_time,train_cosines,test_cosines,train_error,test_error,train_eigvecs_error,test_eigvecs_error"

    if not os.path.isfile(f'grid_search_CDR/{args["experiment_type"]}/results.csv'):
        with open(f'grid_search_CDR/{args["experiment_type"]}/results.csv', "w") as f:
            f.write(header)

    data = jnp.load("CDR_dataset.npz")
    subspace_data = jnp.load("CDR_local_POD.npz")
    
    data = {key: jnp.array(data[key]) for key in data.keys()}
    subspace_data = {key: jnp.array(subspace_data[key]) for key in subspace_data.keys()}

    features = jnp.stack([data["convection"], data["diffusion"][:, 1::2], data["reaction"], data["source"], data["solutions"][:, 0]], 1)
    basis = jnp.transpose(subspace_data["basis"][:, :, :args["N_basis"]], (0, 2, 1))
    
    coordinates = jnp.expand_dims(jnp.array(data["x"]), axis=0)
    norm_factor = jnp.max(jnp.abs(features), axis=[0, 2], keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    features = features / norm_factor

    # for N_drop_ in [100, 200]:
    #     for N_modes_ in [16, 14, 10]:
    #         for N_layers_ in [5, 4, 3]:
    #             for learning_rate_ in [1e-3, 1e-4]:
    #                 args["N_drop"] = N_drop_
    #                 args["N_modes"] = N_modes_
    #                 args["N_layers"] = N_layers_
    #                 args["learning_rate"] = learning_rate_
    exp_hash = "".join([str(args[a]) for a in sorted(args)])
    exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()

    D = features.ndim - 2
    N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
    N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]

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
    
    n = random.choice(keys[1], args["N_train"], shape = (N_run, args["N_batch"]))
    if args["experiment_type"] == "stochastic_normal":
        carry = [model, features, basis, coordinates, keys[1], opt_state]
    else:
        carry = [model, features, basis, coordinates, opt_state]

    if args["experiment_type"] == "l2":
        make_step_scan_ = lambda a, b: l2_make_step_scan(a, b, optim)
    elif args["experiment_type"] == "stochastic_normal":
        make_step_scan_ = lambda a, b: stochastic_normal_make_step_scan(a, b, optim)
    elif args["experiment_type"] == "subspace":
        make_step_scan_ = lambda a, b: subspace_make_step_scan(a, b, optim)
        
    start = time.time()
    carry, history = scan(make_step_scan_, carry, n)
    stop = time.time()
    training_time = stop - start
    model = carry[0]
    opt_state = carry[-1]

    eqx.tree_serialise_leaves(f'grid_search_CDR/{args["experiment_type"]}/model_{exp_hash}.eqx', model)
    eqx.tree_serialise_leaves(f'grid_search_CDR/{args["experiment_type"]}/opt_state_{exp_hash}.eqx', opt_state)

    cosines, errors, rel_errors = compute_metrics(model, features, coordinates, data)
    
    train_cosines = jnp.mean(cosines[:args["N_train"]])
    test_cosines = jnp.mean(cosines[args["N_train"]:])
    
    train_errors = jnp.mean(errors[:args["N_train"]])
    test_errors = jnp.mean(errors[args["N_train"]:])
    
    train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
    test_rel_errors = jnp.mean(rel_errors[args["N_train"]:])

    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{history[-1]},{model_size},{training_time},{train_cosines},{test_cosines},{train_errors},{test_errors},{train_rel_errors},{test_rel_errors}"

    with open(f'grid_search_CDR/{args["experiment_type"]}/results.csv', "a") as f:
        f.write(data)
    
    jnp.savez(f'grid_search_CDR/{args["experiment_type"]}/metrics_{exp_hash}.npz', cosines=cosines, errors=errors, eigvecs_errors=rel_errors, history=history)