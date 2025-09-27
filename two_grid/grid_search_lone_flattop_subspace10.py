import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse

from scipy.sparse import coo_matrix
from jax import random, vmap, disable_jit
from jax.lax import scan, dot_general
from jax.tree_util import tree_map, tree_flatten
from jax.nn import gelu

# Model
class ConvNet(eqx.Module):
    convs: list

    def __init__(self, D, features, odd_kernel, key, N_convs):
        keys = random.split(key, N_convs)
        self.convs = [eqx.nn.Conv(D, features, features, odd_kernel, padding=(odd_kernel - 1) // 2, key=key) for key in keys[:-1]]

    def __call__(self, x):
        for c in self.convs:
            x = gelu(c(x))
        return x
    
class UNet_normalised(eqx.Module):
    transofrm_convs: list
    before_convs: list
    after_d_convs: list
    after_convs: list
    right_after_convs: list
    pool: list = eqx.field(static=True)
    conv_t: list

    def __init__(self, D, staring_N, features, kernel_size, N_convs, key, depth=5):
        input_features, internal_features, output_features = features
        features_layers = [internal_features*2**i for i in range(depth)]
        even_kernel = 2 * (kernel_size // 2)
        odd_kernel = 2 * (kernel_size // 2) + 1
        N_down = [staring_N//2**i for i in range(depth)]
        upsampling_kernels = []
        for i in range(depth-1):
            if 2*N_down[::-1][i] == N_down[::-1][i+1]:
                upsampling_kernels.append(even_kernel)
            else:
                upsampling_kernels.append(odd_kernel)

        keys = random.split(key, 3)
        self.transofrm_convs = [eqx.nn.Conv(D, input_features, internal_features, odd_kernel, padding=(odd_kernel - 1) // 2, key=keys[0])]
        self.transofrm_convs += [eqx.nn.Conv(D, internal_features, output_features, odd_kernel, padding=(odd_kernel - 1) // 2, key=keys[1])]

        keys = random.split(keys[-1], depth+1)
        self.before_convs = [ConvNet(D, features, odd_kernel, key, N_convs) for features, key in zip(features_layers, keys[:-1])]

        keys = random.split(keys[-1], depth)
        self.after_d_convs = [eqx.nn.Conv(D, feature, 2*feature, odd_kernel, padding=(odd_kernel - 1) // 2, key=key) for feature, key in zip(features_layers[:-1], keys[:-1])]

        keys = random.split(keys[-1], depth+2)
        self.after_convs = [ConvNet(D, features, odd_kernel, key, N_convs) for features, key in zip(features_layers, keys[:-1])][::-1]

        keys = random.split(keys[-1], depth)
        self.right_after_convs = [eqx.nn.Conv(D, 2*feature, feature, odd_kernel, padding=(odd_kernel - 1) // 2, key=key) for feature, key in zip(features_layers[:-1], keys[:-1])][::-1]

        self.pool = [eqx.nn.AdaptivePool(N_down[i+1], D, jnp.max) for i in range(depth-1)]

        keys = random.split(keys[-1], depth-1)
        self.conv_t = [eqx.nn.ConvTranspose(D, f_in, f_out, kernel_size, padding=(kernel_size - 2) // 2, stride=2, key=key) for f_in, f_out, kernel_size, key in zip(features_layers[::-1][:-1], features_layers[::-1][1:], upsampling_kernels, keys)]

    def __call__(self, x):
        X = [self.before_convs[0](self.transofrm_convs[0](x))]
        for i, p in enumerate(self.pool):
            x_ = self.before_convs[i+1](self.after_d_convs[i](p(X[-1])))
            X.append(x_)
        x_ = self.after_convs[0](X[-1])

        for i, up in enumerate(self.conv_t):
            x_ = up(x_)
            x_ = self.after_convs[i+1](self.right_after_convs[i](jnp.vstack([X[len(X)-2-i], x_])))
        x_ = self.transofrm_convs[1](x_)
        
        norm = jnp.linalg.norm(x_.reshape(x_.shape[0], -1), axis=1)
        x_ = x_ / (norm.reshape([-1, ]+ [1,]*(x_.ndim-1)) + 1e-8)
        return x_


# Losses
def stochastic_normal_loss(model, input, target, eps=1e-3):
    target = target.reshape(-1,)
    basis = model(input)
    basis = basis.reshape(basis.shape[0], -1)
    b = basis @ target
    A = basis @ basis.T
    sol = jnp.linalg.solve(A + jnp.eye(A.shape[0])*eps, b) @ basis
    return jnp.linalg.norm(target - sol)**2

def batch_stochastic_normal_loss(model, input, target):
    res = vmap(stochastic_normal_loss, in_axes=(None, 0, 0))(model, input, target)
    return jnp.mean(res)

stochastic_normal_compute_loss_and_grads = eqx.filter_value_and_grad(batch_stochastic_normal_loss)

def stochastic_normal_make_step_scan(carry, n, optim):
    model, features, targets, key, opt_state = carry
    key, key_ = random.split(key)
    coeff = random.normal(key_, (n.shape[0], targets.shape[1], 1, 1))
    targets_ = jnp.sum(targets[n]*coeff, axis=1)
    loss, grads = stochastic_normal_compute_loss_and_grads(model, features[n], targets_)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, key, opt_state], loss

def subspace_loss(model, input, target):
    Q = model(input)
    Q = Q.reshape(Q.shape[0], -1)
    Q = jnp.linalg.qr(Q.T)[0]
    return target.shape[0] - jnp.sum((target.reshape(target.shape[0], -1) @ Q)**2)

def batch_subspace_loss(model, input, target):
    res = vmap(subspace_loss, in_axes=(None, 0, 0))(model, input, target)
    return jnp.mean(res)

subspace_compute_loss_and_grads = eqx.filter_value_and_grad(batch_subspace_loss)

def subspace_make_step_scan(carry, n, optim):
    model, features, targets, opt_state = carry
    loss, grads = subspace_compute_loss_and_grads(model, features[n], targets[n])
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, opt_state], loss



# Assessment on the problem
def jacobi_method(x, b, w, A, D, *args, **kwargs):
    return x + w*(b - A @ x) / D

def two_grid_method(x, b, w, A, D, A_proj, proj_mat):
    x = x + w * (b - A @ x) / D
    r = b - A @ x
    e = np.linalg.solve(A_proj, proj_mat.T @ r)
    x = x + proj_mat @ e
    x = x + w * (b - A @ x) / D
    return x

def power_estimate(A, proj_mat, omega, n_est=10, method=two_grid_method):
    A_proj = proj_mat.T @ (A @ proj_mat)
    D = A.diagonal()
    r = 0
    col = np.random.randn(proj_mat.shape[0],)
    col /= np.linalg.norm(col, ord=np.inf)
    for _ in range(n_est):
        col = method(col, np.zeros_like(col), omega, A, D, A_proj, proj_mat)
        col /= np.linalg.norm(col, ord=np.inf)
    col_ = method(col, np.zeros_like(col), omega, A, D, A_proj, proj_mat)
    r = np.linalg.norm(col_, ord=np.inf)
    return r


# Train loop
def make_prediction_scan(carry, i):
    model, features = carry
    prediction = model(features[i])
    return carry, prediction

def compute_metrics(model, features, eigenvectors, A_data, A_indices, N_eig, N_train, N_samples, key, N_est):
    predictions = scan(make_prediction_scan, [model, features], jnp.arange(features.shape[0]))[1]
    Q = vmap(lambda x: jnp.linalg.qr(x.T)[0].T)(predictions.reshape(predictions.shape[0], predictions.shape[1], -1))
    M = dot_general(Q, eigenvectors[:, :N_eig].reshape(eigenvectors.shape[0], N_eig, -1), (((2,), (2,)), ((0,), (0,))))
    cosines = vmap(jnp.linalg.svdvals)(M)
    
    sol = jnp.moveaxis(dot_general(Q, M, (((1,), (1,)), ((0,), (0,)))), 1, 2)
    Errors = jnp.linalg.norm(sol - eigenvectors[:, :N_eig].reshape(eigenvectors.shape[0], N_eig, -1), axis=2)

    keys = random.split(key)
    ind_train = random.choice(keys[0], N_train, shape=(N_samples,))
    ind_test = random.choice(keys[1], eigenvectors.shape[0] - N_train, shape=(N_samples,)) + N_train
    samples = jnp.concatenate([ind_train, ind_test])
    N = predictions.shape[-1]

    radii_power = [] # V, Q_W, Jacboi w/ omega=1
    for sample in samples:
        A = coo_matrix((A_data[sample], (A_indices[:, 0], A_indices[:, 1])), shape=(N**2, N**2)).tocsr()
        # eigenvectors_ = np.array(eigenvectors[sample, :N_eig, :]).reshape(N_eig, -1).T
        basis = np.array(Q[sample, :, :]).T

        # radii_power[0].append(
        #     np.max(np.abs(power_estimate(A, eigenvectors_, omega=.9, n_est=N_est, method=two_grid_method)))
        # )
        radii_power.append(
            np.max(np.abs(power_estimate(A, basis, omega=.9, n_est=N_est, method=two_grid_method)))
        )
        # radii_power[2].append(
        #     np.max(np.abs(power_estimate(A, basis, omega=1., n_est=N_est, method=jacobi_method)))
        # )
    return cosines, Errors, radii_power

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
        "-path_to_save_dir": {
            "default": '/mnt/local/data/vtrifonov/subreg-rom/two_grid/2d/results',
            "help": "path to the directory where to save the results"
        },
        "-experiment_name": {
            "default": "unet_lone_flattop_grid32_w0.9_subspace10",
            "help": "name of the experiment"
        },
       "-path_to_dataset": {
            "default": "/mnt/local/data/vtrifonov/subreg-rom/two_grid/datasets/lone_flattop_N32_w0.9_1200samples.npz",
            "help": "path to dataset in the .npz format"
        },
        "-experiment_type": {
            "default": "stochastic_normal",
            "type": str,
            "choices": ["stochastic_normal", "subspace"],
            "help": "experiment type: defines loss, architecture and how subspace is predicted"
        },
        "-learning_rate": {
            "default": 1e-3,
            "type": float,
            "help": "initial learning rate"
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
            "default": 64,
            "type": int,
            "help": "number of samples used to average gradient"
        },
        "-N_train": {
            "default": 1000,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_epoch": {
            "default": 900,
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
            "help": "dimension of the subspace used to for chosen subspace regression task"
        },
        "-N_layers": {
            "default": 4,
            "type": int,
            "help": "number of layers"
        },
        "-N_samples": {
            "default": 200,
            "type": int,
            "help": "number of reduced eigenproblems solved to test the quality of predicted subspace"
        },
        "-N_est":{
            "default": 200,
            "type": int,
            "help": "number iteration steps to asses the spectral radius"
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

    for experiment_type_ in ["subspace", "stochastic_normal"]:
        for N_subspace_ in [10]:#, 20, 30, 40]:
            for N_drop_ in [100, 200]:
                for N_processor_ in [16, 32, 64]:
                    for N_layers_ in [3, 4, 5]:
                        for learning_rate_ in [1e-3, 1e-4]:
                            args["N_drop"] = N_drop_
                            args["N_layers"] = N_layers_
                            args["N_subspace"] = N_subspace_
                            args["N_processor"] = N_processor_
                            args["learning_rate"] = learning_rate_
                            args["experiment_type"] = experiment_type_

                            exp_hash = "".join([str(args[a]) for a in sorted(args)])
                            exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
                            header = ",".join([key for key in args.keys()])
                            header += ",hash,final_loss,model_size,training_time,train_cosines,test_cosines,train_error,test_error,"
                            header += "train_radius_power_V,test_radius_power_V,train_radius_power_W,test_radius_power_W,train_radius_power_Jac,test_radius_power_Jac"

                            if not os.path.isdir(f'{args["path_to_save_dir"]}/{args["experiment_name"]}'):
                                os.mkdir(f'{args["path_to_save_dir"]}/{args["experiment_name"]}')
#                             else:
#                                 assert False
                            if not os.path.isfile(f'{args["path_to_save_dir"]}/{args["experiment_name"]}/results.csv'):
                                with open(f'{args["path_to_save_dir"]}/{args["experiment_name"]}/results.csv', "w") as f:
                                    f.write(header)

                            data = jnp.load(args["path_to_dataset"])
                            eigenvectors = jnp.array(data["eigenvectors"])
                            features = jnp.array(data["features"])
                            features = features / jnp.max(jnp.abs(features), axis=[0, 2, 3], keepdims=True)
                            # coordinates = jnp.array(data["coordinates"])
                            A_data = np.array(data["A_data"])
                            A_indices = np.array(data["A_indices"])

                            D = features.ndim - 2
                            N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
                            N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]

                            key = random.PRNGKey(args["key"])
                            keys = random.split(key, 3)

                            N_features = [features.shape[1], args["N_processor"], args["N_subspace"]]
                            # model = FFNO_normalised(args["N_layers"], N_features, args["N_modes"], D, keys[0])
                            model = UNet_normalised(D, features.shape[2], N_features, 3, args['N_layers'], keys[0])

                            model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
                            learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
#                                 optim = optax.chain(
#                                     optax.clip_by_global_norm(1.0),
#                                     optax.lion(learning_rate=learning_rate)
#                                 )
                            optim = optax.lion(learning_rate=learning_rate)
                            opt_state = optim.init(eqx.filter(model, eqx.is_array))

                            n = random.choice(keys[1], args["N_train"], shape = (N_run, args["N_batch"]))
                            if args["experiment_type"] == "stochastic_normal":
                                carry = [model, features, eigenvectors[:, :args["N_eig"]], keys[1], opt_state]
                            else:
                                carry = [model, features, eigenvectors[:, :args["N_eig"]], opt_state]

                            if args["experiment_type"] == "stochastic_normal":
                                make_step_scan_ = lambda a, b: stochastic_normal_make_step_scan(a, b, optim)
                            elif args["experiment_type"] == "subspace":
                                make_step_scan_ = lambda a, b: subspace_make_step_scan(a, b, optim)

                            start = time.time()
                            carry, history = scan(make_step_scan_, carry, n)
                            stop = time.time()
                            training_time = stop - start
                            model = carry[0]
                            opt_state = carry[-1]

                            eqx.tree_serialise_leaves(f'{args["path_to_save_dir"]}/{args["experiment_name"]}/model_{exp_hash}.eqx', model)
                            eqx.tree_serialise_leaves(f'{args["path_to_save_dir"]}/{args["experiment_name"]}/opt_state_{exp_hash}.eqx', opt_state)

                            cosines, errors, radii_power = compute_metrics(model, features, eigenvectors, A_data, A_indices, args["N_eig"],
                                                                                       args["N_train"], args["N_samples"], keys[2], args["N_est"])

                            train_cosines = jnp.mean(cosines[:args["N_train"]])
                            test_cosines = jnp.mean(cosines[args["N_train"]:])
                            train_errors = jnp.mean(errors[:args["N_train"]])
                            test_errors = jnp.mean(errors[args["N_train"]:])

                            train_radius_power_V = -42# np.mean(radii_power[0][:args["N_samples"]])
                            test_radius_power_V = -42# np.mean(radii_power[0][args["N_samples"]:])
                            train_radius_power_W = np.mean(radii_power[:args["N_samples"]])
                            test_radius_power_W = np.mean(radii_power[args["N_samples"]:])
                            train_radius_power_Jac = -42# np.mean(radii_power[2][:args["N_samples"]])
                            test_radius_power_Jac = -42# np.mean(radii_power[2][args["N_samples"]:])

                            data = "\n" + ",".join([str(args[key]) for key in args.keys()])
                            data += f",{exp_hash},{history[-1]},{model_size},{training_time},{train_cosines},{test_cosines},{train_errors},{test_errors},"
                            data += f"{train_radius_power_V},{test_radius_power_V},{train_radius_power_W},{test_radius_power_W},{train_radius_power_Jac},{test_radius_power_Jac}"

                            jnp.savez(f'{args["path_to_save_dir"]}/{args["experiment_name"]}/metrics_{exp_hash}.npz',
                                    cosines=cosines, errors=errors, radii_power=radii_power, history=history)
                            with open(f'{args["path_to_save_dir"]}/{args["experiment_name"]}/results.csv', "a") as f:
                                f.write(data)