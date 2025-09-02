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

    def spectral_conv(self, v, A):
        u = 0
        N = v.shape
        for i in range(A.shape[-1]):
            u_ = dynamic_slice_in_dim(jnp.fft.rfft(v, axis=i+1), 0, A.shape[-2], axis=i+1)
            u_ = dot_general(A[:, :, :, i], u_, (((1,), (0,)), ((2, ), (i+1, ))))
            u_ = jnp.moveaxis(u_, 0, i+1)
            u += jnp.fft.irfft(u_, axis=i+1, n=N[i+1])
        return u

def l2_loss(model, input, target, x):
    X = model(input, x).reshape(target.shape[0], -1)
    error = jnp.mean(jnp.sum((X - target.reshape(target.shape[0], -1))**2, axis=1))
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
    
def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_eig_path": {
            "help": "absolute path to dataset with eigenvectors"
        },
       "-dataset_reg_path": {
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
            "default": 4000,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_val": {
            "default": 200,
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
    header += ",hash,final_loss,model_size,training_time,train_error,test_error,val_error,best_n"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)

    data = np.load(args['dataset_eig_path'])
    features = jnp.array(data["features"])
    coordinates = jnp.array(data["coordinates"])
    
    data = np.load(args['dataset_reg_path'])
    features = jnp.concatenate([jnp.array(data["rhs"]), features], axis=1)
    features = features / jnp.max(jnp.abs(features), axis=[0, 2, 3], keepdims=True)
    targets = jnp.array(data["solutions"])
    targets = targets / jnp.max(jnp.abs(targets), axis=[0, 2, 3], keepdims=True)

    for N_drop_ in [100, 200]:
        for N_modes_ in [16, 14, 10]:
            for N_layers_ in [5, 4, 3]:
                for learning_rate_ in [1e-4, 5e-5]:
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
                    
                    N_features = [coordinates.shape[0] + features.shape[1], args["N_processor"], targets.shape[1]]
                    model = FFNO(args["N_layers"], N_features, args["N_modes"], D, keys[0])
                        
                    model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
                    learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
                    optim = optax.lion(learning_rate=learning_rate)
                    opt_state = optim.init(eqx.filter(model, eqx.is_array))
                    
                    nn = random.choice(keys[1], args["N_train"], shape = (N_run//N_stop, N_stop, args["N_batch"]))
                    carry = [model, features, targets, coordinates, opt_state]
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
                        _, predictions = scan(make_prediction_scan, [model, features[args["N_train"]:(args["N_train"]+args["N_val"])], coordinates], jnp.arange(features[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0]))
                        rel_errors = jnp.linalg.norm(predictions.reshape(predictions.shape[0], -1) - targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1) / jnp.linalg.norm(targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1)
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
                
                    _, predictions = scan(make_prediction_scan, [models[best_n], features, coordinates], jnp.arange(features.shape[0]))
                    rel_errors = jnp.linalg.norm(predictions.reshape(predictions.shape[0], -1) - targets.reshape(targets.shape[0], -1), axis=1) / jnp.linalg.norm(targets.reshape(targets.shape[0], -1), axis=1)
                    train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
                    val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
                    test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])
                
                    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
                    data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{best_n}"
                
                    with open(f'{args["results_path"]}/results.csv', "a") as f:
                        f.write(data)
                    
                    jnp.savez(f'{args["results_path"]}/metrics_{exp_hash}.npz', rel_errors=rel_errors, history=history)