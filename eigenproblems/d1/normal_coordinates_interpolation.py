import time
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import hashlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.interpolate import RBFInterpolator, griddata
from scipy.sparse import coo_matrix

def exp_G(Y, T, t):
    u, sigma, vt = np.linalg.svd(T, full_matrices=False)
    Y_ = Y @ (vt.T @ (np.cos(sigma*t).reshape(-1, 1) * vt)) + u @ (np.sin(sigma*t).reshape(-1, 1) * vt)
    return Y_

def log_G(Y0, Y1):
    X = Y1.T @ Y0
    u, sigma, vt = np.linalg.svd(Y1.T @ Y0, full_matrices=False)
    Y1_ = Y1 @ (u @ vt)
    L = Y1_ - Y0 @ (Y0.T @ Y1_)
    u, sigma, vt = np.linalg.svd(L, full_matrices=False)
    sigma[sigma > 1.0] = 1.0
    sigma[sigma < -1.0] = -1.0
    T = u @ (np.arcsin(sigma).reshape(-1, 1) * vt)
    return T

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

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_path": {
            "help": "absolute path to dataset"
        },
       "-results_path": {
            "help": "absolute path to folder where results are stored"
        },  
        "-interpolation_type": {
            "default": "nearest",
            "type": str,
            "choices": ["nearest", "rbf", "GP"],
            "help": "defines which interpolation is used in tangent space"
        },
        "-features_type": {
            "default": "rough",
            "type": str,
            "choices": ["rough", "true"],
            "help": "which features to use for interpolation"
        },
        "-N_basis": {
            "default": 10,
            "type": int,
            "help": "number of basis bectors used to compute error"
        },
        "-N_subspace": {
            "default": 10,
            "type": int,
            "help": "dimension of the subspace used to approximate basis"
        },
        "-N_test": {
            "default": 100,
            "type": int,
            "help": "number of test samples"
        },
        "-N_neighbours": {
            "default": 10,
            "type": int,
            "help": "number of neighbours used to build normal coordinates"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    header = ",".join([key for key in args.keys()])
    header += ",exp_hash,test_error,time"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)

    data = np.load(args['dataset_path'])
    subspaces = data["eigenvectors"][:, :args['N_subspace']]
    D = subspaces.ndim - 2
    subspaces = subspaces.reshape(subspaces.shape[0], subspaces.shape[1], -1)
    features = data["potentials"] if args['features_type'] == 'rough' else data["true_features"]
    features = features.reshape(features.shape[0], -1)
    potentials = data["potentials"]
    potentials = potentials.reshape(potentials.shape[0], -1)
    coordinates_x = np.array(data["coordinates"])
    if D == 1:
        coordinates = np.expand_dims(np.linspace(0, 1, coordinates_x.shape[0]), 0)
        h = coordinates_x[1] - coordinates_x[0]
        L = get_laplacian_1D(coordinates_x.shape[-1], h)
    elif D == 2:
        coordinates = coordinates_x
        h = coordinates_x[0, 0, 1] - coordinates_x[0, 0, 0]
        L = get_laplacian_2D(coordinates_x.shape[-1], h)

    Data = dict()
    for N_neighbours_ in [5, 10, 15]:
        for N_subspace_ in [10, 20, 30]:
            for features_type_ in ['rough', 'true']:
                args['N_neighbours'] = N_neighbours_
                args['N_subspace'] = N_subspace_
                args['features_type'] = features_type_
                test_errors = []
                T = 0
                exp_hash = "".join([str(args[a]) for a in sorted(args)])
                exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
                for sample in range(args['N_test']):
                    subspace = subspaces[-sample]
                    feature = features[-sample]
                    V = potentials[-sample]
                    
                    neighbours = np.argsort(np.linalg.norm(features[:-args['N_test']] - feature.reshape(1, -1), axis=1))[:args['N_neighbours']]
                    s_nei = subspaces[neighbours]
                    f_nei = features[neighbours]
                    f_nei = f_nei - f_nei[:1]
                    v_interp = [np.zeros_like(subspaces[0]).T, ]
                    for i in range(1, s_nei.shape[0]):
                        v = log_G(s_nei[0].T, s_nei[i].T)
                        v_interp.append(v)
                    v_interp = np.array(v_interp)
                
                    t0 = time.time()
                    if args["interpolation_type"] == "nearest":
                        tangent_interpolated = griddata(f_nei, v_interp.reshape(v_interp.shape[0], -1), np.expand_dims(feature, 0), method='nearest').reshape(v_interp[0].shape)
                    elif args["interpolation_type"] == "rbf":
                        interpolator = RBFInterpolator(f_nei, v_interp.reshape(v_interp.shape[0], -1), degree=-1, kernel='inverse_quadratic', epsilon=0.1)
                        tangent_interpolated = interpolator(np.expand_dims(feature, 0))[0].reshape(v_interp[0].shape)
                    elif args["interpolation_type"] == "GP":
                        kernel = ConstantKernel() * Matern(length_scale=1.0, nu=1.5) + ConstantKernel() * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
                        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0).fit(f_nei, v_interp.reshape(v_interp.shape[0], -1))
                        tangent_interpolated = model.predict(np.expand_dims(feature, 0)).reshape(v_interp[0].shape)
                    t1 = time.time()
                    
                    subspace_interpolated = exp_G(s_nei[0].T, tangent_interpolated, 1.0).T
                    eig = compute_eigenvectors(L, V, subspace_interpolated)[1]
                    errors = np.minimum(np.linalg.norm(eig[:args['N_basis']] - subspace[:args['N_basis']], axis=1), np.linalg.norm(eig[:args['N_basis']] + subspace[:args['N_basis']], axis=1))
                    T += t1 - t0
                    test_errors.append(errors)
                test_errors = np.array(test_errors)
                data = "\n" + ",".join([str(args[key]) for key in args.keys()])
                data += f",{exp_hash},{np.mean(test_errors)},{T}"
                with open(f'{args["results_path"]}/results.csv', "a") as f:
                    f.write(data)
                Data[exp_hash] = test_errors

    dataset_name = args["dataset_path"].split("/")[-1]
    np.savez(f'{args["results_path"]}/metrics_{args["interpolation_type"]}_{dataset_name}', **Data)