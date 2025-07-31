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

def compare_eigenvectors(x, y):
    return jnp.minimum(jnp.linalg.norm((x - y).reshape(-1,)), jnp.linalg.norm((x + y).reshape(-1,)))**2

def l2_loss(model, input, target, x):
    X = model(input, x)
    error = compare_eigenvectors(X[0], target[0])
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

def hidden_make_prediction_scan(carry, i):
    model, features, coords = carry
    prediction = model.get_subspace(features[i], coords)
    return carry, prediction

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-path_to_dataset": {
            "help": "path to dataset in the .npz format"
        },
        "-experiment_type": {
            "default": "l2",
            "type": str,
            "choices": ["l2", "stochastic_normal", "subspace"],
            "help": "experiment type: defines loss, architecture and how subspace is predicted"
        },
        "-eigenvector": {
            "default": 0,
            "type": int,
            "help": "select N-th smallest eigenvector as a target"
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
            "default": 4000,
            "type": int,
            "help": "number of samples in the training set"
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
        "-N_samples": {
            "default": 100,
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
    header += ",hash,final_loss,model_size,training_time,train_error,test_error"

    if not os.path.isfile(f'optimal_one_vector/{args["experiment_type"]}/results.csv'):
        with open(f'optimal_one_vector/{args["experiment_type"]}/results.csv', "w") as f:
            f.write(header)
    
    data = jnp.load(args["path_to_dataset"])
    features = jnp.array(data["features"])
    features = features / jnp.max(jnp.abs(features), axis=[0, 2, 3], keepdims=True)
    coordinates = jnp.array(data["coordinates"])
    A_data = np.array(data["A_data"])
    A_indices = np.array(data["A_indices"])

    params_one_vector = {
    	0: {
    		'learning_rate': [0.001, 0.001, 0.001],
    		'N_drop': [200, 100, 100],
    		'N_modes': [16, 16, 16],
    		'N_layers': [4, 4, 5]
    	},
    	1: {
    		'learning_rate': [0.001, 0.001, 0.001],
    		'N_drop': [200, 100, 100],
    		'N_modes': [14, 16, 14],
    		'N_layers': [4, 4, 4]
    	},
    	2: {
    		'learning_rate': [0.001, 0.001, 0.001],
    		'N_drop': [200, 100, 200],
    		'N_modes': [16, 16, 14],
    		'N_layers': [5, 5, 5]
    	},
    	3: {
    		'learning_rate': [0.001, 0.001, 0.001],
    		'N_drop': [200, 200, 100],
    		'N_modes': [16, 14, 16],
    		'N_layers': [5, 5, 5]
    	},
    	4: {
    		'learning_rate': [0.001, 0.001, 0.001],
    		'N_drop': [200, 200, 200],
    		'N_modes': [14, 16, 10],
    		'N_layers': [5, 5, 5]
    	},
    	5: {
    		'learning_rate': [0.001, 0.001, 0.001],
    		'N_drop': [100, 200, 100],
    		'N_modes': [16, 16, 10],
    		'N_layers': [5, 5, 5]
    	}
    }
    
    for eigenvector_ in params_one_vector:
        args["eigenvector"] = eigenvector_
        eigenvectors = jnp.array(data["eigenvectors"])[:, args["eigenvector"]:(args["eigenvector"]+1)]
        for ii in range(3):
            for key_ in params_one_vector[eigenvector_].keys():
                args[key_] = params_one_vector[eigenvector_][key_][ii]
            exp_hash = "".join([str(args[a]) for a in sorted(args)])
            exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()

            D = features.ndim - 2
            N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
            N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
        
            key = random.PRNGKey(args["key"])
            keys = random.split(key, 3)
            
            if args["experiment_type"] == "l2":
                N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], 1]
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
                carry = [model, features, eigenvectors[:, :args["N_eig"]], coordinates, keys[1], opt_state]
            else:
                carry = [model, features, eigenvectors[:, :args["N_eig"]], coordinates, opt_state]
        
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
        
            eqx.tree_serialise_leaves(f'optimal_one_vector/{args["experiment_type"]}/model_{exp_hash}.eqx', model)
            eqx.tree_serialise_leaves(f'optimal_one_vector/{args["experiment_type"]}/opt_state_{exp_hash}.eqx', opt_state)
        
            predictions = scan(make_prediction_scan, [model, features, coordinates], jnp.arange(features.shape[0]))[1]
            errors1 = jnp.linalg.norm((predictions - eigenvectors).reshape(eigenvectors.shape[0], -1), axis=1)
            errors2 = jnp.linalg.norm((predictions + eigenvectors).reshape(eigenvectors.shape[0], -1), axis=1)
            errors = jnp.minimum(errors1, errors2)

            train_errors = jnp.mean(errors[:args["N_train"]])
            test_errors = jnp.mean(errors[args["N_train"]:])
            
            data_ = "\n" + ",".join([str(args[key]) for key in args.keys()])
            data_ += f",{exp_hash},{history[-1]},{model_size},{training_time},{train_errors},{test_errors}"
        
            with open(f'optimal_one_vector/{args["experiment_type"]}/results.csv', "a") as f:
                f.write(data_)
            
            jnp.savez(f'optimal_one_vector/{args["experiment_type"]}/metrics_{exp_hash}.npz', errors=errors, history=history)