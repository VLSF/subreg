import sys
import jax.numpy as jnp
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from jax import random

def get_weights(N, h, decay, power):
    k = 2*jnp.pi*jnp.fft.fftfreq(N, d=h)
    K_x, K_y= jnp.meshgrid(k, k, indexing="ij")
    w = 1 / (1 + decay*(K_x**2 + K_y**2))**power
    return w, k

def random_series(w, k, c_a):
    res = jnp.real(jnp.fft.ifftn(c_a * w, norm="ortho"))
    return res

def get_sample(A_data, A_indices, key, N, decay, power, w, k):
    h = 1/N
    c_a = random.normal(key, w.shape, dtype=jnp.complex64)
    rhs = np.array(random_series(w, k, c_a).reshape(-1,))
    A = coo_matrix((A_data, (A_indices[:, 0], A_indices[:, 1])), shape=(N**2, N**2)).tocsc()
    A_ = splu(A)
    sol = A_.solve(rhs)
    return sol, rhs

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    path_to_dataset = sys.argv[2]
    data = jnp.load(path_to_dataset)
    A_data = np.array(data["A_data"])
    A_indices = np.array(data["A_indices"])

    N = 100
    h = 1/N
    decay = 0.01
    power = 2
    keys = random.split(random.PRNGKey(33), A_data.shape[0])
    w, k = get_weights(N, h, decay, power)
    
    solutions, rhs = [], []
    data = {
        "solutions": [],
        "rhs": []
    }
    for a_data, key in zip(A_data, keys):
        s, r = get_sample(a_data, A_indices, key, N, decay, power, w, k)
        data['solutions'].append(s)
        data['rhs'].append(r)
    for key in data.keys():
        data[key] = np.array(key)

    np.savez(dataset_name + "_standard_regression.npz", **data)