import sys
import os
import tqdm
import warnings
import numpy as np
import hashlib
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.linear_model import LinearRegression

def sklearn_GP_scalar(x_train, x_test, y_train, model):
    test_pred = []
    train_pred = []
    warnings_list = []
    for k in tqdm.tqdm(range(y_train.shape[-1])):
        with warnings.catch_warnings(record=True) as cx_manager:
            model.fit(x_train, y_train[:, k])
            warnings_list.append([i.message for i in cx_manager])
        test_pred.append(model.predict(x_test))
        train_pred.append(model.predict(x_train))
    test_pred, train_pred = np.stack(test_pred, 1), np.stack(train_pred, 1)
    return test_pred, train_pred, warnings_list


if __name__ == "__main__":
    regression_path = sys.argv[1] # '/home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/lone_flattop_regression.npz'
    eig_path = sys.argv[2] # '/home/jovyan/vfanaskov/eigenvectors/eigenvalues_fine_2D/lone_flattop.npz'
    res_path = sys.argv[3] # '/home/jovyan/vfanaskov/eigenvectors/elliptic_regression_2D/kernel'
    dataset_name = eig_path.split("/")[-1]
    N_train = 4000
    
    data_regression = np.load(regression_path)
    data_a = np.load(sys.argv[2])
    targets = data_regression['solutions']
    rhs = data_regression['rhs']
    a_features = data_a['features']
    N_samples = rhs.shape[0]
    features = np.concatenate([rhs.reshape(N_samples, -1), a_features.reshape(N_samples, -1)], axis=1)
    features = features / np.max(np.abs(features), keepdims=True)
    targets = targets.reshape(N_samples, -1)
    targets = targets / np.max(np.abs(targets))

    header = "dataset,kernel,modes_f,modes_t,train_error,test_error"
    if not os.path.isfile(f'{res_path}/results.csv'):
        with open(f'{res_path}/results.csv', "w") as f:
            f.write(header)

    Data = dict()
    transformations_in = dict()
    transformations_out = dict()
    for N_modes_in in [50, 100, 150, 200]:
        if str(N_modes_in) in transformations_in.keys():
            input_pca_transform = transformations_in[str(N_modes_in)]
            x_train = input_pca_transform.transform(features[:N_train])
            x_test = input_pca_transform.transform(features[N_train:])
            print("shortcut in")
        else:  
            input_pca_transform = PCA(n_components=N_modes_in)
            x_train = input_pca_transform.fit_transform(features[:N_train])
            x_test = input_pca_transform.transform(features[N_train:])
            transformations_in[str(N_modes_in)] = input_pca_transform
        for N_modes_out in [50, 100, 150, 200]:
            if str(N_modes_out) in transformations_out.keys():
                output_pca_transform = transformations_out[str(N_modes_out)]
                y_train = output_pca_transform.transform(targets[:N_train])
                print("shortcut out")
            else:  
                output_pca_transform = PCA(n_components=N_modes_out)
                y_train = output_pca_transform.fit_transform(targets[:N_train])
                transformations_out[str(N_modes_out)] = output_pca_transform
            for kernel_name in ["RBF", "Matern"]:
                print(kernel_name, N_modes_out, y_train.shape, N_modes_in, x_train.shape)
                kernel = Matern(nu = 2.5) if kernel_name == "Matern" else RBF()
                model = GaussianProcessRegressor(kernel, alpha = 1e-10)
                test_pred, train_pred, warnings_list = sklearn_GP_scalar(x_train, x_test, y_train, model)
                test_pred, train_pred = output_pca_transform.inverse_transform(test_pred), output_pca_transform.inverse_transform(train_pred)
                train_errors = np.linalg.norm(train_pred - targets[:N_train], axis=1) / np.linalg.norm(targets[:N_train], axis=1)
                test_errors = np.linalg.norm(test_pred - targets[N_train:], axis=1) / np.linalg.norm(targets[N_train:], axis=1)
    
                Data[f"{kernel_name}, {N_modes_in}, {N_modes_out}, train errors"] = train_errors
                Data[f"{kernel_name}, {N_modes_in}, {N_modes_out}, test errors"] = test_errors
                with open(f'{res_path}/results.csv', "a") as f:
                    f.write(f"\n{dataset_name},{kernel_name},{N_modes_in},{N_modes_out},{np.mean(train_errors)},{np.mean(test_errors)}")

    np.savez(res_path + f"/metrics_{dataset_name}", **Data)