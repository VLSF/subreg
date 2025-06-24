import sys
import jax.numpy as jnp
import numpy as np
import cupy as cp

from cupyx.scipy.sparse import coo_matrix as coo_matrix_cu
from cupyx.scipy.sparse.linalg import eigsh as eigsh_cu
from scipy.sparse import coo_matrix
from jax import random

def get_weights(N, h, decay, power):
    k = 2*jnp.pi*jnp.fft.fftfreq(N, d=h)
    K_x, K_y, K_z = jnp.meshgrid(k, k, k, indexing="ij")
    w = 1 / (1 + decay*(K_x**2 + K_y**2 + K_z**2))**power
    return w, k

def random_series(X, Y, Z, w, k, c_a):
    res = jnp.real(jnp.fft.ifftn(c_a * w, norm="ortho"))
    return res

def a_rand(X, Y, Z, params, key):
    w, k, alpha, beta, slope, A = params
    c_a = random.normal(key, X.shape, dtype=jnp.complex64) * A
    a = alpha + (beta-alpha)*(jnp.tanh(slope*random_series(X, Y, Z, w, k, c_a)) + 1)/2
    return a
    
def get_discretization_data(N):
    N_x = N_y = N_z = N

    x, y, z = np.linspace(0, 1, 2*N_x+3)[1:-1], np.linspace(0, 1, 2*N_y+3)[1:-1], np.linspace(0, 1, 2*N_z+3)[1:-1]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    i, j, k = np.arange(N_x, dtype=int), np.arange(N_y, dtype=int), np.arange(N_z, dtype=int)
    I, J, K = np.meshgrid(i, j, k, indexing="ij")
    lex = lambda i, j, k, N_x, N_y: (k + (j + i*N_x)*N_y)
    diag = lex(I, J, K, N_x, N_y)
    i_m = lex(I-1, J, K, N_x, N_y)
    i_p = lex(I+1, J, K, N_x, N_y)
    j_m = lex(I, J-1, K, N_x, N_y)
    j_p = lex(I, J+1, K, N_x, N_y)
    k_m = lex(I, J, K-1, N_x, N_y)
    k_p = lex(I, J, K+1, N_x, N_y)

    mask_i_m = (I-1) >= 0
    mask_i_p = (I+1) < N_x
    mask_j_m = (J-1) >= 0
    mask_j_p = (J+1) < N_y
    mask_k_m = (K-1) >= 0
    mask_k_p = (K+1) < N_z

    rows = np.concatenate([diag.reshape(-1,), diag[mask_i_m], diag[mask_i_p], diag[mask_j_m], diag[mask_j_p], diag[mask_k_m], diag[mask_k_p]])
    cols = np.concatenate([diag.reshape(-1,), i_m[mask_i_m], i_p[mask_i_p], j_m[mask_j_m], j_p[mask_j_p], k_m[mask_k_m], k_p[mask_k_p]])
    indices = np.stack([cols, rows], 1)
    return mask_i_m, mask_i_p, mask_j_m, mask_j_p, mask_k_m, mask_k_p, X, Y, Z, indices

def get_matrix(discretization_data, params, key):
    mask_i_m, mask_i_p, mask_j_m, mask_j_p, mask_k_m, mask_k_p, X, Y, Z, indices = discretization_data
    
    N = (X.shape[0] - 1) // 2
    keys = random.split(key, 3)
    a1 = a_rand(X, Y, Z, params, keys[0])
    a2 = a_rand(X, Y, Z, params, keys[1])
    a3 = a_rand(X, Y, Z, params, keys[2])
    
    a1_p = a1[::2][1:, 1::2, 1::2]
    a1_m = a1[::2][:-1, 1::2, 1::2]
    a2_p = a2[:, ::2][1::2, 1:, 1::2]
    a2_m = a2[:, ::2][1::2, :-1, 1::2]
    a3_p = a3[:, :, ::2][1::2, 1::2, 1:]
    a3_m = a3[:, :, ::2][1::2, 1::2, :-1]
    
    a1 = a1[1::2, 1::2, 1::2]
    a2 = a2[1::2, 1::2, 1::2]
    a3 = a3[1::2, 1::2, 1::2]
    a = np.stack([a1, a2, a3], axis=0)
    data = np.concatenate([
        (a1_p + a1_m + a2_p + a2_m + a3_p + a3_m).reshape(-1,),
        -a1_m[mask_i_m],
        -a1_p[mask_i_p],
        -a2_m[mask_j_m],
        -a2_p[mask_j_p],
        -a3_m[mask_k_m],
        -a3_p[mask_k_p],
    ])

    A = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(N**3, N**3))
    return A, a, data

def get_dataset(N, N_samples, N_eigenvalues, params, key):
    discretization_data = get_discretization_data(N)
    coords = np.stack([discretization_data[-4][1::2, 1::2, 1::2], discretization_data[-3][1::2, 1::2, 1::2], discretization_data[-2][1::2, 1::2, 1::2]], axis=0)

    Keys = random.split(key, (N_samples, 2))
    Vals, Vecs, features, Data = [], [], [], []
    for keys in Keys:
        A, a, data = get_matrix(discretization_data, params, keys[0])
        vals, vecs = eigsh_cu(coo_matrix_cu(A).tocsr(), k=N_eigenvalues, which="SA")
        vals, vecs = vals.get(), vecs.get()
        Vals.append(vals)
        Vecs.append(vecs.T.reshape(N_eigenvalues, N, N, N))
        features.append(a)
        Data.append(data)

    Vals = np.stack(Vals)
    Data = np.stack(Data)
    Vecs = np.stack(Vecs)
    features = np.stack(features)
    Vals, Vecs, coords, features, Data, indices = jnp.array(Vals), jnp.array(Vecs), jnp.array(coords), jnp.array(features), jnp.array(Data), jnp.array(discretization_data[-1])
    data = {
        "eigenvalues": Vals,
        "eigenvectors": Vecs,
        "coordinates": coords,
        "features": features,
        "A_data": Data,
        "A_indices": indices
    }
    return data

if __name__ == "__main__":
    N_samples = 1000
    N_samples_ = 20
    N_eigenvalues = 50
    N = 30
    
    alpha = 1
    beta = 50
    slope = 2
    A = 1000
    decay = 1e-2
    power = 1.5

    h = 0.5/(N+1)
    w, k = get_weights(2*N+1, h, decay, power)
    params = [w, k, alpha, beta, slope, A]
    
    key = random.PRNGKey(33)
    keys = random.split(key, N_samples//N_samples_)

    for i, key in enumerate(keys):
        data = get_dataset(N, N_samples_, N_eigenvalues, params, key)
        jnp.savez(f"triple_flattops_low_res/triple_flattops_{i}.npz", eigenvalues=data["eigenvalues"], eigenvectors=data["eigenvectors"], features=data["features"], A_data=data["A_data"])
        if i == 0:
            jnp.savez(f"triple_flattops_low_res/triple_flattops_global.npz", A_indices=data["A_indices"], coordinates=data["coordinates"])