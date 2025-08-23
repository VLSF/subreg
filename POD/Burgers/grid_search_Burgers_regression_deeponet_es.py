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
    
def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_name": {
            "help": "I, II or III"
        },
        "-experiment_type": {
            "default": "DeepONet",
            "type": str,
            "choices": ["DeepONet", ],
            "help": "experiment type: defines architecture used"
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
        "-N_layers_trunk": {
            "default": 4,
            "type": int,
            "help": "number of layers in trunk net"
        },
        "-N_layers_branch": {
            "default": 4,
            "type": int,
            "help": "number of layers in branch net"
        },
        "-kernel_size_trunk": {
            "default": 3,
            "type": int,
            "help": "kernel size of conv in trunk net"
        },
        "-N_encoder_trunk": {
            "default": 5,
            "type": int,
            "help": "number of features in trunk net after encoder, doupls each layer"
        },
        "-N_basis": {
            "default": 100,
            "type": int,
            "help": "number of basis functions in branch net"
        },
        "-N_features_branch": {
            "default": 64,
            "type": int,
            "help": "number of hidden neurons in branch net"
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
    header += ",hash,final_loss,model_size,training_time,train_error,test_error,val_error,best_n"

    if not os.path.isfile(f'grid_search_Burgers_regression_es/{args["experiment_type"]}/results.csv'):
        with open(f'grid_search_Burgers_regression_es/{args["experiment_type"]}/results.csv', "w") as f:
            f.write(header)

    data = jnp.load(f'Burgers_{args["dataset_name"]}.npz')

    if args["dataset_name"] == "III":
        features = jnp.concatenate([data["solutions"][:, :1], jnp.expand_dims(data["diff"][:, 1::2], 1), jnp.expand_dims(data["forcing"], 1)], 1)
        a_extended = jnp.array(data["diff"])
        forcing = jnp.array(data["forcing"])
    elif args["dataset_name"] == "II":
        forcing = jnp.concatenate([data["forcing"],]*data["solutions"].shape[0], axis=0)
        features = jnp.concatenate([data["solutions"][:, :1], jnp.expand_dims(data["diff"][:, 1::2], 1), jnp.expand_dims(forcing, 1)], 1)
        a_extended = jnp.array(data["diff"])
    elif args["dataset_name"] == "I":
        forcing = jnp.concatenate([data["forcing"],]*data["solutions"].shape[0], axis=0)
        a_extended = jnp.concatenate([data["diff"],]*data["solutions"].shape[0], axis=0)
        features = jnp.concatenate([data["solutions"][:, :1], jnp.expand_dims(a_extended[:, 1::2], 1), jnp.expand_dims(forcing, 1)], 1)

    targets = jnp.expand_dims(jnp.array(data["solutions"]), 1)
    norm_factor = jnp.max(jnp.abs(targets), axis=[0, 2, 3], keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    targets = targets / norm_factor
    x = jnp.array(data["coords"])
    t = jnp.linspace(0, 1, targets.shape[2])
    coordinates = jnp.stack(jnp.meshgrid(t, x, indexing="ij"), 0)
    x = jnp.expand_dims(x, 0)
    norm_factor = jnp.max(jnp.abs(features), axis=[0, 2], keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    features = features / norm_factor

    for N_drop_ in [100, 200]:
        for N_basis_ in [100, 150, 200]:
            for N_layers_ in [6, 5, 4, 3]:
                for learning_rate_ in [1e-3, 1e-4]:
                    for kernel_size_trunk_ in [9, 7, 5, 3]:
                        args["N_drop"] = N_drop_
                        args["N_basis"] = N_basis_
                        args["N_layers_trunk"] = N_layers_
                        args["N_layers_branch"] = N_layers_
                        args["learning_rate"] = learning_rate_
                        args["kernel_size_trunk"] = kernel_size_trunk_
                        args["N_encoder_trunk"] = 25
                        args["N_features_branch"] = N_basis_

                        exp_hash = "".join([str(args[a]) for a in sorted(args)])
                        exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
                    
                        D = features.ndim - 2
                        N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
                        N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
                        N_stop = args["stop_each"] * args["N_train"] // args["N_batch"]
                    
                        key = random.PRNGKey(args["key"])
                        keys = random.split(key, 3)
                        
                        if args["experiment_type"] == "DeepONet":
                            trunk_params = [x.shape[1], [x.shape[0] + features.shape[1], args["N_encoder_trunk"], args["N_basis"]], args["N_layers_trunk"], args["kernel_size_trunk"]]
                            branch_params = [[coordinates.shape[0], args["N_features_branch"], args["N_basis"]], args["N_layers_branch"]]
                            model = DeepONet(trunk_params, branch_params, keys[0])
                        
                        model_size = sum(tree_map(lambda x: jnp.size(x) if not (x is None) else 0, tree_flatten(model)[0], is_leaf=eqx.is_array))
                        learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
                        optim = optax.lion(learning_rate=learning_rate)
                        opt_state = optim.init(eqx.filter(model, eqx.is_array))
                        
                        nn = random.choice(keys[1], args["N_train"], shape = (N_run//N_stop, N_stop, args["N_batch"]))
                        carry = [model, features, targets, x, coordinates, opt_state]
                        make_step_scan_ = lambda a, b: l2_make_step_scan(a, b, optim)
                    
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
                            _, predictions = scan(make_prediction_scan, [model, features[args["N_train"]:(args["N_train"]+args["N_val"])], x, coordinates], jnp.arange(features[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0]))
                            rel_errors = jnp.linalg.norm(predictions.reshape(predictions.shape[0], -1) - targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1) / jnp.linalg.norm(targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1)
                            val_rel_errors.append(jnp.mean(rel_errors))
                            if jnp.isnan(val_rel_errors[-1]).item():
                                break
                            training_times.append(training_time)
                            histories.append(history)
                            
                        val_rel_errors = jnp.array(val_rel_errors)
                        val_rel_errors = jnp.nan_to_num(val_rel_errors, nan=jnp.inf)
                        best_n = jnp.argmin(val_rel_errors)
                        concat_n = min(best_n + 1, len(histories))
                        history = jnp.concatenate(histories[:concat_n])
                    
                        eqx.tree_serialise_leaves(f'grid_search_Burgers_regression_es/{args["experiment_type"]}/model_{exp_hash}.eqx', models[best_n])
                        eqx.tree_serialise_leaves(f'grid_search_Burgers_regression_es/{args["experiment_type"]}/opt_state_{exp_hash}.eqx', opt_states[best_n])
                    
                        _, predictions = scan(make_prediction_scan, [models[best_n], features, x, coordinates], jnp.arange(features.shape[0]))
                        rel_errors = jnp.linalg.norm(predictions.reshape(predictions.shape[0], -1) - targets.reshape(targets.shape[0], -1), axis=1) / jnp.linalg.norm(targets.reshape(targets.shape[0], -1), axis=1)
                        train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
                        val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
                        test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])
                    
                        data = "\n" + ",".join([str(args[key]) for key in args.keys()])
                        data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{best_n}"
                    
                        with open(f'grid_search_Burgers_regression_es/{args["experiment_type"]}/results.csv', "a") as f:
                            f.write(data)
                        
                        jnp.savez(f'grid_search_Burgers_regression_es/{args["experiment_type"]}/metrics_{exp_hash}.npz', rel_errors=rel_errors, history=history)