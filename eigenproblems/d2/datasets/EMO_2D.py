import sys
import numpy as np

from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu
from jax import random

def compute_p(r, r_e, phi):
    r_ = r / r_e
    n = np.expand_dims(np.arange(phi.shape[0]-1), 0)
    p = (1 - (r_ - 1) / (r_ + 1)) * np.sum((np.expand_dims((r_ - 1) / (r_ + 1), 1) ** n) * np.expand_dims(phi[:-1], 0), axis=1)
    p = p + (r_ - 1) / (r_ + 1) * phi[-1]
    return p

def compute_EMO(r, r_e, D, phi):
    r_ = r / r_e
    n = np.expand_dims(np.arange(phi.shape[0]), 0)
    V = D*(1 - np.exp(-(r_-1) / (r_ + 1) * compute_p(r, r_e, phi)))**2
    return V

def compute_mixed_EMO(r, r_e, D, phi1, phi2):
    V = np.concatenate([compute_EMO(r[r <= r_e], r_e, D, phi1), compute_EMO(r[r > r_e], r_e, D, phi2)], 0)
    return V

def get_laplacian_2D(N_x, h):
    L = -2*np.eye(N_x) + np.eye(N_x, k=1) + np.eye(N_x, k=-1)
    I = np.eye(N_x)
    L = (np.kron(I, L) + np.kron(L, I)) / h**2
    L = coo_matrix(L)
    return L

def compute_eigenvectors(L, V, N_eig):
    A = - L + diags([V.reshape(-1,),], [0,])
    A_ = splu(A.tocsc())
    L_ = LinearOperator(A.shape, matvec=lambda x: A_.solve(x), dtype=A.dtype)
    eigvals, eigvecs = eigsh(L_, k=N_eig)
    return eigvals, eigvecs

def EMO_2D_dataset_I(N_samples=1000):
    # Dynamics on the Double Morse Potential: A Paradigm for Roaming Reactions with no Saddle Points, https://arxiv.org/abs/1709.06973
    N_eig = 30
    N = 100
    R = 7
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    L = get_laplacian_2D(N, x[1] - x[0])
    coords = np.stack(np.meshgrid(x, y)).reshape(2, -1)
    ind = np.arange(0, N**2)
    
    apmlitude1 = 3
    amplitude2 = 3
    p_ord1 = 2
    p_ord2 = 2
    N_r = 100
    r_e_min = 1
    r_e_max = 5
    D_min = 10
    D_max = 40

    key = random.PRNGKey(33)
    keys = random.split(key, 5)
    mask = -np.array([(-1)**i for i in range(p_ord1-1)] + [1,]).reshape(1, -1)
    phi1 = np.array(random.uniform(keys[0], (N_samples, p_ord1)) * apmlitude1).astype(np.float64)*mask
    phi1[:, -1] = 10 + np.abs(phi1[:, -1])
    phi2 = np.array(random.uniform(keys[1], (N_samples, p_ord2)) * amplitude2).astype(np.float64)*mask
    phi2[:, -1] = 10 + np.abs(phi2[:, -1])
    r_e = np.array(random.uniform(keys[2], (N_samples,)) * (r_e_max - r_e_min) + r_e_min).astype(np.float64)
    D = np.array(random.uniform(keys[3], (N_samples,)) * (D_max - D_min) + D_min).astype(np.float64)
    dr = np.array(2*random.uniform(keys[4], (N_samples, 2)) - 1).astype(np.float64)
    dr = dr / np.linalg.norm(dr, axis=1, keepdims=True) / np.sqrt(2) * 2 * np.expand_dims(r_e, 1)
    
    potentials = []
    for i in range(N_samples):
        r1 = np.linalg.norm(coords + dr[i].reshape(2, -1), axis=0)
        r2 = np.linalg.norm(coords - dr[i].reshape(2, -1), axis=0)
        V12 = compute_EMO(r1, r_e[i], D[i], phi1[i]) + compute_EMO(r2, r_e[i], D[i], phi2[i])
        potentials.append(V12)
    potentials = np.array(potentials)

    eigenvectors = []
    eigenvalues = []
    for sample in range(N_samples):
        vals, vecs = compute_eigenvectors(L, potentials[sample], N_eig)
        vecs = np.array(vecs)
        vecs = np.moveaxis(vecs.reshape(N, N, -1), 2, 0)[:N_eig]
        vals = vals[:N_eig]
        vals = 1 / vals
        order = np.argsort(vals)
        vecs = vecs[order]
        vals = vals[order]
        eigenvectors.append(vecs[:N_eig])
        eigenvalues.append(vals[:N_eig])
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)

    data = {
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "potentials": potentials.reshape(-1, N, N),
        "coordinates": coords.reshape(-1, N, N),
        "true_features": np.concatenate([phi1, phi2, r_e.reshape(-1, 1), D.reshape(-1, 1), dr], axis=1)
    }
    return data

def EMO_2D_dataset_II(N_samples=1000):
    # Dynamics on the Double Morse Potential: A Paradigm for Roaming Reactions with no Saddle Points, https://arxiv.org/abs/1709.06973
    N_eig = 30
    N = 100
    R = 7
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    L = get_laplacian_2D(N, x[1] - x[0])
    coords = np.stack(np.meshgrid(x, y)).reshape(2, -1)
    ind = np.arange(0, N**2)
    
    apmlitude1 = 6
    amplitude2 = 6
    p_ord1 = 5
    p_ord2 = 5
    N_r = 100
    R = 10
    r_e_min = 1
    r_e_max = 5
    D_min = 10
    D_max = 40

    key = random.PRNGKey(33)
    keys = random.split(key, 5)
    mask = -np.array([(-1)**i for i in range(p_ord1-1)] + [1,]).reshape(1, -1)
    phi1 = np.array(random.uniform(keys[0], (N_samples, p_ord1)) * apmlitude1).astype(np.float64)*mask
    phi1[:, -1] = 10 + np.abs(phi1[:, -1])
    phi2 = np.array(random.uniform(keys[1], (N_samples, p_ord2)) * amplitude2).astype(np.float64)*mask
    phi2[:, -1] = 10 + np.abs(phi2[:, -1])
    r_e = np.array(random.uniform(keys[2], (N_samples,)) * (r_e_max - r_e_min) + r_e_min).astype(np.float64)
    D = np.array(random.uniform(keys[3], (N_samples,)) * (D_max - D_min) + D_min).astype(np.float64)
    dr = np.array(2*random.uniform(keys[4], (N_samples, 2)) - 1).astype(np.float64)
    dr = dr / np.linalg.norm(dr, axis=1, keepdims=True) / np.sqrt(2) * 1.5 * np.expand_dims(r_e, 1)
    
    potentials = []
    for i in range(N_samples):
        r1 = np.linalg.norm(coords + dr[i].reshape(2, -1), axis=0)
        r2 = np.linalg.norm(coords - dr[i].reshape(2, -1), axis=0)
        V12 = compute_EMO(r1, r_e[i], D[i], phi1[i]) + compute_EMO(r2, r_e[i], D[i], phi2[i])
        potentials.append(V12)
    potentials = np.array(potentials)

    eigenvectors = []
    eigenvalues = []
    for sample in range(N_samples):
        vals, vecs = compute_eigenvectors(L, potentials[sample], N_eig)
        vecs = np.array(vecs)
        vecs = np.moveaxis(vecs.reshape(N, N, -1), 2, 0)[:N_eig]
        vals = vals[:N_eig]
        vals = 1 / vals
        order = np.argsort(vals)
        vecs = vecs[order]
        vals = vals[order]
        eigenvectors.append(vecs[:N_eig])
        eigenvalues.append(vals[:N_eig])
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)

    data = {
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "potentials": potentials.reshape(-1, N, N),
        "coordinates": coords.reshape(-1, N, N),
        "true_features": np.concatenate([phi1, phi2, r_e.reshape(-1, 1), D.reshape(-1, 1), dr], axis=1)
    }
    return data

def EMO_2D_dataset_III(N_samples=1000):
    # Dynamics on the Double Morse Potential: A Paradigm for Roaming Reactions with no Saddle Points, https://arxiv.org/abs/1709.06973
    N_eig = 30
    N = 100
    R = 7
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    L = get_laplacian_2D(N, x[1] - x[0])
    coords = np.stack(np.meshgrid(x, y)).reshape(2, -1)
    ind = np.arange(0, N**2)
    
    apmlitude1 = 10
    amplitude2 = 10
    p_ord1 = 10
    p_ord2 = 10
    N_r = 100
    R = 10
    r_e_min = 1
    r_e_max = 5
    D_min = 10
    D_max = 40

    key = random.PRNGKey(33)
    keys = random.split(key, 5)
    mask = -np.array([(-1)**i for i in range(p_ord1-1)] + [1,]).reshape(1, -1)
    phi1 = np.array(random.uniform(keys[0], (N_samples, p_ord1)) * apmlitude1).astype(np.float64)*mask
    phi1[:, -1] = 10 + np.abs(phi1[:, -1])
    phi2 = np.array(random.uniform(keys[1], (N_samples, p_ord2)) * amplitude2).astype(np.float64)*mask
    phi2[:, -1] = 10 + np.abs(phi2[:, -1])
    r_e = np.array(random.uniform(keys[2], (N_samples,)) * (r_e_max - r_e_min) + r_e_min).astype(np.float64)
    D = np.array(random.uniform(keys[3], (N_samples,)) * (D_max - D_min) + D_min).astype(np.float64)
    dr = np.array(2*random.uniform(keys[4], (N_samples, 2)) - 1).astype(np.float64)
    dr = dr / np.linalg.norm(dr, axis=1, keepdims=True) / np.sqrt(2) * 1.5 * np.expand_dims(r_e, 1)
    
    potentials = []
    for i in range(N_samples):
        r1 = np.linalg.norm(coords + dr[i].reshape(2, -1), axis=0)
        r2 = np.linalg.norm(coords - dr[i].reshape(2, -1), axis=0)
        V12 = compute_EMO(r1, r_e[i], D[i], phi1[i]) + compute_EMO(r2, r_e[i], D[i], phi2[i])
        potentials.append(V12)
    potentials = np.array(potentials)

    eigenvectors = []
    eigenvalues = []
    for sample in range(N_samples):
        vals, vecs = compute_eigenvectors(L, potentials[sample], N_eig)
        vecs = np.array(vecs)
        vecs = np.moveaxis(vecs.reshape(N, N, -1), 2, 0)[:N_eig]
        vals = vals[:N_eig]
        vals = 1 / vals
        order = np.argsort(vals)
        vecs = vecs[order]
        vals = vals[order]
        eigenvectors.append(vecs[:N_eig])
        eigenvalues.append(vals[:N_eig])
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)

    data = {
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "potentials": potentials.reshape(-1, N, N),
        "coordinates": coords.reshape(-1, N, N),
        "true_features": np.concatenate([phi1, phi2, r_e.reshape(-1, 1), D.reshape(-1, 1), dr], axis=1)
    }
    return data

if __name__ == '__main__':
    '''
    eigenvectors (1000, 30, 100, 100)
    eigenvalues (1000, 30)
    potentials (1000, 100, 100)
    coordinates (100,)
    true_features (1000, 6) or true_features (1000, 14) or true_features (1000, 24) for I, II, III respectively
    '''
    dataset_name = sys.argv[1]
    if dataset_name == "I":
        data = EMO_2D_dataset_I()
    elif dataset_name == "II":
        data = EMO_2D_dataset_II()
    elif dataset_name == "III":
        data = EMO_2D_dataset_III()
    np.savez(f"EMO_2D_{dataset_name}.npz", **data)