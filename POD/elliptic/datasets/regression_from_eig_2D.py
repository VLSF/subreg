import sys
import numpy as np
from jax import random

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    key = random.PRNGKey(47)
    N_basis = 10
    data = np.load(dataset_name + '.npz')
    eigenvalues = data['eigenvalues']
    eigenvectors = data['eigenvectors']
    coefficients = np.array(random.normal(key, (eigenvalues.shape[0], N_basis, 1, 1)))
    solutions = np.sum(eigenvectors[:, :N_basis] * coefficients / eigenvalues[:, :N_basis].reshape(eigenvalues.shape[0], N_basis, 1, 1), axis=1)
    rhss = np.sum(eigenvectors[:, :N_basis] * coefficients, axis=1)
    solutions = np.expand_dims(solutions, 1)
    rhss = np.expand_dims(rhss, 1)
    data = {
        "rhs": rhss,
        "solutions": solutions
    }
    np.savez(dataset_name + "_regression.npz", **data)