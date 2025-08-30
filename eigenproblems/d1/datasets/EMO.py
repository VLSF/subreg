import sys
import numpy as np
from jax import random

def compute_p(r, r_e, phi):
    r_ = r / r_e
    n = np.expand_dims(np.arange(phi.shape[0]-1), 0)
    p = (1 - (r_ - 1) / (r_ + 1)) * np.sum((np.expand_dims((r_ - 1) / (r_ + 1), 1) ** n) * np.expand_dims(phi[:-1], 0), axis=1)
    p = p + (r_ - 1) / (r_ + 1) * phi[-1]
    return p

def compute_EMO(r, r_e, D, phi):
    # expanded Morse oscillator, doi: 10.1063/1.2354502
    r_ = r / r_e
    n = np.expand_dims(np.arange(phi.shape[0]), 0)
    V = D*(1 - np.exp(-(r_-1) / (r_ + 1) * compute_p(r, r_e, phi)))**2
    return V

def compute_mixed_EMO(r, r_e, D, phi1, phi2):
    V = np.concatenate([compute_EMO(r[r <= r_e], r_e, D, phi1), compute_EMO(r[r > r_e], r_e, D, phi2)], 0)
    return V

def get_laplacian(N_x, R):
    h = R / (N_x - 1)
    L = -2*np.eye(N_x) + np.eye(N_x, k=1) + np.eye(N_x, k=-1)
    return L / h**2

def compute_eigenvectors(L, V):
    A = - L + np.diag(V)
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvals, eigvecs

def mixed_EMO_dataset_I(N_samples=1000):
    N_eig = 30
    apmlitude1 = 1
    amplitude2 = 10
    p_ord1 = 3
    p_ord2 = 1
    N_r = 100
    R = 10
    r_e_min = 2
    r_e_max = 7
    D_min = 10
    D_max = 40
    L = get_laplacian(N_r, R)
    key = random.PRNGKey(33)
    r = np.linspace(0, R, N_r)

    keys = random.split(key, 4)
    mask = -np.array([(-1)**i for i in range(p_ord1-1)] + [1,]).reshape(1, -1)
    phi1 = np.array(random.uniform(keys[0], (N_samples, p_ord1)) * apmlitude1).astype(np.float64)*mask
    phi2 = np.array(random.uniform(keys[1], (N_samples, p_ord2)) * amplitude2).astype(np.float64)
    phi2[:, -1] = 1 + np.abs(phi2[:, -1])
    r_e = np.array(random.uniform(keys[2], (N_samples,)) * (r_e_max - r_e_min) + r_e_min).astype(np.float64)
    D = np.array(random.uniform(keys[3], (N_samples,)) * (D_max - D_min) + D_min).astype(np.float64)

    potentials = []
    for i in range(N_samples):
        V12 = compute_mixed_EMO(r, r_e[i], D[i], phi1[i], phi2[i])
        potentials.append(V12)
    potentials = np.array(potentials)

    eigenvectors = []
    eigenvalues = []
    for sample in range(N_samples):
        eigvals, eigvecs = compute_eigenvectors(L, potentials[sample])
        eigenvectors.append(eigvecs[:, :N_eig].T)
        eigenvalues.append(eigvals[:N_eig])
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)

    data = {
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "potentials": potentials,
        "coordinates": r,
        "true_features": np.concatenate([phi1, phi2, r_e.reshape(-1, 1), D.reshape(-1, 1)], axis=1)
    }
    return data

def mixed_EMO_dataset_II(N_samples=1000):
    N_eig = 30
    apmlitude1 = 1
    amplitude2 = 10
    p_ord1 = 4
    p_ord2 = 4
    N_r = 100
    R = 10
    r_e_min = 1
    r_e_max = 5
    D_min = 10
    D_max = 40
    L = get_laplacian(N_r, R)
    key = random.PRNGKey(33)
    r = np.linspace(0, R, N_r)

    keys = random.split(key, 4)
    mask = -np.array([(-1)**i for i in range(p_ord1-1)] + [1,]).reshape(1, -1)
    phi1 = np.array(random.uniform(keys[0], (N_samples, p_ord1)) * apmlitude1).astype(np.float64)*mask
    phi2 = np.array(random.uniform(keys[1], (N_samples, p_ord2)) * amplitude2).astype(np.float64)
    phi2[:, -1] = 1 + np.abs(phi2[:, -1])
    r_e = np.array(random.uniform(keys[2], (N_samples,)) * (r_e_max - r_e_min) + r_e_min).astype(np.float64)
    D = np.array(random.uniform(keys[3], (N_samples,)) * (D_max - D_min) + D_min).astype(np.float64)

    potentials = []
    for i in range(N_samples):
        V12 = compute_mixed_EMO(r, r_e[i], D[i], phi1[i], phi2[i])
        potentials.append(V12)
    potentials = np.array(potentials)

    eigenvectors = []
    eigenvalues = []
    for sample in range(N_samples):
        eigvals, eigvecs = compute_eigenvectors(L, potentials[sample])
        eigenvectors.append(eigvecs[:, :N_eig].T)
        eigenvalues.append(eigvals[:N_eig])
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)

    data = {
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "potentials": potentials,
        "coordinates": r,
        "true_features": np.concatenate([phi1, phi2, r_e.reshape(-1, 1), D.reshape(-1, 1)], axis=1)
    }
    return data

def mixed_EMO_dataset_III(N_samples=1000):
    N_eig = 30
    apmlitude1 = 5
    amplitude2 = 10
    p_ord1 = 10
    p_ord2 = 10
    N_r = 100
    R = 10
    r_e_min = 1
    r_e_max = 8
    D_min = 10
    D_max = 40
    L = get_laplacian(N_r, R)
    key = random.PRNGKey(33)
    r = np.linspace(0, R, N_r)

    keys = random.split(key, 4)
    mask = -np.array([(-1)**i for i in range(p_ord1-1)] + [1,]).reshape(1, -1)
    phi1 = np.array(random.uniform(keys[0], (N_samples, p_ord1)) * apmlitude1).astype(np.float64)*mask
    phi2 = np.array(random.uniform(keys[1], (N_samples, p_ord2)) * amplitude2).astype(np.float64)
    phi2[:, -1] = 1 + np.abs(phi2[:, -1])
    r_e = np.array(random.uniform(keys[2], (N_samples,)) * (r_e_max - r_e_min) + r_e_min).astype(np.float64)
    D = np.array(random.uniform(keys[3], (N_samples,)) * (D_max - D_min) + D_min).astype(np.float64)

    potentials = []
    for i in range(N_samples):
        V12 = compute_mixed_EMO(r, r_e[i], D[i], phi1[i], phi2[i])
        potentials.append(V12)
    potentials = np.array(potentials)

    eigenvectors = []
    eigenvalues = []
    for sample in range(N_samples):
        eigvals, eigvecs = compute_eigenvectors(L, potentials[sample])
        eigenvectors.append(eigvecs[:, :N_eig].T)
        eigenvalues.append(eigvals[:N_eig])
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)

    data = {
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "potentials": potentials,
        "coordinates": r,
        "true_features": np.concatenate([phi1, phi2, r_e.reshape(-1, 1), D.reshape(-1, 1)], axis=1)
    }
    return data

if __name__ == '__main__':
    '''
    eigenvectors (1000, 30, 100)
    eigenvalues (1000, 30)
    potentials (1000, 100)
    coordinates (100,)
    true_features (1000, 6) or true_features (1000, 10) or true_features (1000, 22) for I, II, III respectively
    '''
    dataset_name = sys.argv[1]
    if dataset_name == "I":
        data = mixed_EMO_dataset_I()
    elif dataset_name == "II":
        data = mixed_EMO_dataset_II()
    elif dataset_name == "III":
        data = mixed_EMO_dataset_III()
    np.savez(f"EMO_{dataset_name}.npz", **data)
