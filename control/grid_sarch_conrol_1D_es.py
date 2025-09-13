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
    coeff = random.normal(key_, (n.shape[0], targets.shape[1], 1))
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
    Q = jnp.linalg.qr(prediction.reshape(prediction.shape[0], -1).T)[0].T
    return carry, Q

def diffusion_1D(dif_coef):
    h = 1 / dif_coef.shape[0]
    main_diag = -(dif_coef[:-1] + dif_coef[1:])
    off_diag = dif_coef[1:-1]
    A = (jnp.diag(main_diag) + jnp.diag(off_diag, k=-1) + jnp.diag(off_diag, k=1)) / h**2
    return A

def integrate_C(carry, t):
    C11_, C12_, w11, w12, W_, U, l_c, dt = carry
    CW = C11_ @ W_
    C12_ = (C12_ - dt * C11_ - dt * CW @ (W_.T @ C12_) * l_c) / w12
    C11_ = (C11_ - dt * CW @ CW.T * l_c) / w11
    return [C11_, C12_, w11, w12, W_, U, l_c, dt], [U @ C11_ @ U.T, U @ C12_ @ U.T]

def get_weight_matrix(A, dt):
    vals, vecs = jnp.linalg.eigh(A)
    weights11 = jnp.ones((vals.shape[0], vals.shape[0])) - dt * (vals.reshape(-1, 1) + vals.reshape(1, -1))
    weigths12 = jnp.ones((vals.shape[0], vals.shape[0])) - dt * vals.reshape(-1, 1)
    return weights11, weigths12, vecs

def get_optimal_control(psi, A, W, b, t, l_c):
    # output alpha, beta
    # closed-loop optimal control alpha @ phi + beta
    dt = t[1] - t[0]
    w11, w12, U = get_weight_matrix(A, dt)
    W_ = U.T @ W
    psi_ = U.T @ psi
    C11_ = psi_ @ psi_.T
    C12_ = C11_ * 0
    _, [C11, C12] = scan(integrate_C, [C11_, C12_, w11, w12, W_, U, l_c, dt], t[1:])
    C11 = jnp.concatenate([C11[::-1], jnp.expand_dims(U @ C11_ @ U.T, 0)], axis=0)
    C12 = jnp.concatenate([C12[::-1], jnp.expand_dims(C12_, 0)], axis=0)
    return -W.T @ C11 * l_c, -W.T @ C12 @ b * l_c

def integrate_closed_loop(carry, i):
    M, W, phi, b, alpha, beta, dt = carry
    phi = jnp.linalg.solve(M, phi - dt * b + dt * W @ (alpha[i] @ phi + beta[i]))
    return [M, W, phi, b, alpha, beta, dt], phi

def compute_truncated_control(diffusion, W, psi, b, phi, T, t, l_c = 15):
    A = diffusion_1D(diffusion[::2])
    Q = T.T
    A_ = Q.T @ A @ Q
    W_ = Q.T @ W
    psi_ = Q.T @ psi
    M_ = (jnp.eye(A_.shape[0]) - (t[1] - t[0]) * A_)
    b_ = Q.T @ b
    phi_ = Q.T @ phi
    alpha_, beta_ = get_optimal_control(psi_, A_, W_, b_, t, l_c)
    carry_closed_loop_ = [M_, W_, phi_, b_, alpha_, beta_, t[1] - t[0]]
    _, phi_tr = scan(integrate_closed_loop, carry_closed_loop_, jnp.arange(t.shape[0]-1))
    phi_tr = phi_tr @ Q.T
    return alpha_, beta_, phi_tr[-1]

def compute_metrics(model, features, coordinates, basis, diffusion, W, psi, b, phi, t, phi_opt):
    Q = scan(make_prediction_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
    M = dot_general(Q, basis, (((2,), (2,)), ((0,), (0,))))
    cosines = vmap(jnp.linalg.svdvals)(M)

    sol = jnp.moveaxis(dot_general(Q, M, (((1,), (1,)), ((0,), (0,)))), 1, 2)
    Errors = jnp.linalg.norm(sol - basis, axis=2)

    phi_tr = vmap(compute_truncated_control, in_axes=(0, 0, 0, 0, 0, 0, None))(diffusion, W, psi, b, phi, Q, t)[-1]
    delta = phi_opt - phi_tr
    rel_error = jnp.linalg.norm(delta, axis=1) / jnp.linalg.norm(phi_opt, axis=1)
    rel_error_projected = jnp.linalg.norm(vmap(lambda x, y: x.T @ y, in_axes=(0, 0))(psi, delta), axis=1) / jnp.linalg.norm(vmap(lambda x, y: x.T @ y, in_axes=(0, 0))(psi, phi_opt), axis=1)
    return cosines, Errors, rel_error, rel_error_projected

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
            "default": 1000,
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
    header += ",hash,final_loss,model_size,training_time,train_cosines,val_cosines,test_cosines,train_error,val_errors,test_error,train_final_state_error,val_final_state_error,test_final_state_error,train_error_projected,val_error_projected,test_error_projected,best_n"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)

    data = jnp.load(args['dataset_path'])
    basis = jnp.array(data["basis"])[:, :, :args["N_basis"]]
    basis = vmap(lambda x: jnp.linalg.qr(x)[0].T)(basis)
    coordinates = jnp.expand_dims(jnp.array(data["coords"]), axis=0)
    features = jnp.concatenate([
        jnp.array(data["W"]),
        jnp.array(data["psi"]),
        jnp.expand_dims(jnp.array(data["diffusion"][:, 1::2]) / jnp.max(jnp.abs(data["diffusion"][:, ::2]), axis=1, keepdims=True), 2)
    ], axis=2)
    features = jnp.transpose(features, (0, 2, 1))

    phi_opt = jnp.array(data["solution_controlled"])[:, -1]
    W = jnp.array(data["W"])
    psi = jnp.array(data["psi"])
    b = jnp.array(data["b"])
    phi = jnp.array(data["phi"])
    t = jnp.array(data["t"])
    diffusion = jnp.array(data["diffusion"])

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

                    D = features.ndim - 2
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
                        rel_errors = compute_metrics(model, features[args["N_train"]:(args["N_train"]+args["N_val"])], coordinates, basis[args["N_train"]:(args["N_train"]+args["N_val"])], diffusion[args["N_train"]:(args["N_train"]+args["N_val"])], W[args["N_train"]:(args["N_train"]+args["N_val"])], psi[args["N_train"]:(args["N_train"]+args["N_val"])], b[args["N_train"]:(args["N_train"]+args["N_val"])], phi[args["N_train"]:(args["N_train"]+args["N_val"])], t, phi_opt[args["N_train"]:(args["N_train"]+args["N_val"])])[-1]
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

                    cosines, errors, rel_errors, rel_errors_projected = compute_metrics(models[best_n], features, coordinates, basis, diffusion, W, psi, b, phi, t, phi_opt)

                    train_cosines = jnp.mean(cosines[:args["N_train"]])
                    val_cosines = jnp.mean(cosines[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_cosines = jnp.mean(cosines[(args["N_train"]+args["N_val"]):])

                    train_errors = jnp.mean(errors[:args["N_train"]])
                    val_errors = jnp.mean(errors[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_errors = jnp.mean(errors[(args["N_train"]+args["N_val"]):])

                    train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
                    val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])

                    train_rel_errors_projected = jnp.mean(rel_errors_projected[:args["N_train"]])
                    val_rel_errors_projected = jnp.mean(rel_errors_projected[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_rel_errors_projected = jnp.mean(rel_errors_projected[(args["N_train"]+args["N_val"]):])

                    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
                    data += f",{exp_hash},{history[-1]},{model_size},{training_time},{train_cosines},{val_cosines},{test_cosines},{train_errors},{val_errors},{test_errors},{train_rel_errors},{val_rel_errors},{test_rel_errors},{train_rel_errors_projected},{val_rel_errors_projected},{test_rel_errors_projected},{best_n}"

                    with open(f'{args["results_path"]}/results.csv', "a") as f:
                        f.write(data)

                    jnp.savez(f'{args["results_path"]}/metrics_{exp_hash}.npz', errors=rel_errors, history=history)
