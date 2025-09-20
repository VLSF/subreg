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

class MLP(eqx.Module):
    weights: list
    biases: list

    def __init__(self, N_features, N_layers, key):
        N_in, N_processor, N_out = N_features
        keys = random.split(key, N_layers+2)
        Ns = [N_in,] + [N_processor,]*N_layers + [N_out,]
        self.biases = [jnp.zeros((No, )) for No in Ns[1:]]
        self.weights = [random.normal(key, (No, Ni)) / jnp.sqrt(No + Ni) for Ni, No, key in zip(Ns[:-1], Ns[1:], keys)]

    def __call__(self, f):
        f = self.weights[0] @ f + self.biases[0]
        for w, b in zip(self.weights[1:-1], self.biases[1:-1]):
            f = gelu(w @ f + b) + f
        f = self.weights[-1] @ f + self.biases[-1]
        return f

def l2_loss(model, feature, target):
    X = model(feature)
    error = jnp.sum((X - target)**2)
    return error

def batch_l2_loss(model, feature, target):
    res = vmap(l2_loss, in_axes=(None, 0, 0))(model, feature, target)
    return jnp.mean(res)

l2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def l2_make_step_scan(carry, n, optim):
    model, features, targets, opt_state = carry
    loss, grads = l2_compute_loss_and_grads(model, features[n], targets[n])
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, opt_state], loss

def make_prediction_scan(carry, i):
    model, features = carry
    prediction = model(features[i])
    return carry, prediction
    
def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_path": {
            "help": "absolute path to dataset for regression problem"
        },
       "-results_path": {
            "help": "absolute path to folder where results are stored"
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
            "default": 3000,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch"
        },
        "-stop_each": {
            "default": 50,
            "type": int,
            "help": "stop each N_epoch to evaluate the model and make checkpoint"
        },
        "-N_layers": {
            "default": 3,
            "type": int,
            "help": "number of layers in MLP"
        },
        "-N_processor": {
            "default": 300,
            "type": int,
            "help": "number of features after encoder"
        },
        "-N_basis_f": {
            "default": 50,
            "type": int,
            "help": "number of basis functions in POD for features compression"
        },
        "-N_basis_t": {
            "default": 100,
            "type": int,
            "help": "number of basis functions in POD for targets compression"
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey that seed all randomness in the code"
        },
        "-optim": {
            "default": 'adam',
            "type": str,
            "choices": ["lion", "adam"],
            "help": "optimiser"
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
    header += ",hash,final_loss,model_size,training_time,train_error,test_error,val_error,best_n"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)

    data = jnp.load(args['dataset_path'])

    if dataset_name == "Burgers_III.npz":
        features = jnp.concatenate([data["solutions"][:, :1], jnp.expand_dims(data["diff"][:, 1::2], 1), jnp.expand_dims(data["forcing"], 1)], 1)
        a_extended = jnp.array(data["diff"])
        forcing = jnp.array(data["forcing"])
    elif dataset_name == "Burgers_II.npz":
        forcing = jnp.concatenate([data["forcing"],]*data["solutions"].shape[0], axis=0)
        features = jnp.concatenate([data["solutions"][:, :1], jnp.expand_dims(data["diff"][:, 1::2], 1), jnp.expand_dims(forcing, 1)], 1)
        a_extended = jnp.array(data["diff"])
    elif dataset_name == "Burgers_I.npz":
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
    norm_factor = jnp.max(jnp.abs(features), axis=[0, 2], keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    features = features / norm_factor

    targets = targets.reshape(targets.shape[0], -1)
    features = features.reshape(features.shape[0], features.shape[1], -1)
    
    v_features = [jnp.linalg.svd(features[:args["N_train"], i], full_matrices=False)[2] for i in range(features.shape[1])]
    v_t = jnp.linalg.svd(targets[:args["N_train"]], full_matrices=False)[2]

    phi_features = [v[:args['N_basis_f']].T for v in v_features]
    features_ = jnp.concatenate([features[:, i] @ phi_features[i] for i in range(features.shape[1])], axis=1)
    phi_t = v_t[:args['N_basis_t']].T
    targets_ = targets @ phi_t

    for N_basis_f_ in [50, 80, 100]:
        phi_features = [v[:N_basis_f_].T for v in v_features]
        features_ = jnp.concatenate([features[:, i] @ phi_features[i] for i in range(features.shape[1])], axis=1)
        for N_basis_t_ in [100, 250, 400]:
            phi_t = v_t[:N_basis_t_].T
            targets_ = targets @ phi_t
            for N_layers_ in [3, 5, 7]:
                for N_processor_ in [100, 300, 500]:
                    for learning_rate_ in [1e-3, 1e-4]:
                        for N_drop_ in [100, 200]:
                            for optim_ in ["lion", "adam"]:
                                args["learning_rate"] = learning_rate_
                                args["N_basis_f"] = N_basis_f_
                                args["N_basis_t"] = N_basis_t_
                                args["N_layers"] = N_layers_
                                args["N_processor"] = N_processor_
                                args["N_drop"] = N_drop_
                                args["optim"] = optim_
        
                                exp_hash = "".join([str(args[a]) for a in sorted(args)])
                                exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
                                
                                D = features.ndim - 2
                                N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
                                N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
                                N_stop = args["stop_each"] * args["N_train"] // args["N_batch"]
                                
                                key = random.PRNGKey(args["key"])
                                keys = random.split(key, 3)
                                
                                N_features_MLP = [features_.shape[1], args["N_processor"], targets_.shape[1]]
                                model = MLP(N_features_MLP, args["N_layers"], keys[0])
                                
                                model_size = sum(tree_map(lambda x: jnp.size(x) if not (x is None) else 0, tree_flatten(model)[0], is_leaf=eqx.is_array))
                                learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
                                if args["optim"] == "lion":
                                    optim = optax.lion(learning_rate=learning_rate)
                                else:
                                    optim = optax.adam(learning_rate=learning_rate)
                                opt_state = optim.init(eqx.filter(model, eqx.is_array))
                                
                                nn = random.choice(keys[1], args["N_train"], shape = (N_run//N_stop, N_stop, args["N_batch"]))
                                carry = [model, features_, targets_, opt_state]
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
                                    _, predictions = scan(make_prediction_scan, [model, features_[args["N_train"]:(args["N_train"]+args["N_val"])]], jnp.arange(features_[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0]))
                                    predictions = predictions @ phi_t.T
                                    rel_errors = jnp.linalg.norm(predictions - targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1) / jnp.linalg.norm(targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1)
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
                                
                                _, predictions = scan(make_prediction_scan, [models[best_n], features_], jnp.arange(features_.shape[0]))
                                predictions = predictions @ phi_t.T
                                rel_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
                                
                                train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
                                val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
                                test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])
                                
                                data = "\n" + ",".join([str(args[key]) for key in args.keys()])
                                data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{best_n}"
                                
                                with open(f'{args["results_path"]}/results.csv', "a") as f:
                                    f.write(data)
                                
                                jnp.savez(f'{args["results_path"]}/metrics_{exp_hash}.npz', rel_errors=rel_errors, history=history)