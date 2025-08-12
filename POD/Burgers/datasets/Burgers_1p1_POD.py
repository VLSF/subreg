import numpy as np
import sys

def get_coordinates(N):
    x = np.linspace(0, 1, N+2)[1:-1]
    x_extended = np.linspace(0, 1, 2*N + 3)[1:-1]
    return x, x_extended

def get_A(a_extended):
    A = np.diag(-a_extended[::2][:-1]-a_extended[::2][1:]) + np.diag(a_extended[::2][1:-1], k=-1) + np.diag(a_extended[::2][1:-1], k=+1)
    return A

def get_B(N):
    B = np.diag(np.ones((N-1,))/2, k=+1) - np.diag(np.ones((N-1,))/2, k=-1)
    B[0, :3] = [-3/2, 2, -1/2]
    B[-1, -3:] = [1/2, -2, 3/2]
    return B

def get_reduced_matrices(A, B, W):
    A_ = W.T @ A @ W
    B_ = W.T @ B
    return A_, B_

def integration_step_pod(u, f, A_, B_, W, N_steps):
    v = np.copy(u)
    for _ in range(N_steps):
        r = B_ @ (W @ v)**2 / 2 + A_ @ v - u + f
        v = v - np.linalg.solve(B_ @ np.diag(W @ v) @ W + A_, r)
    return v

def integrate_Burgers_pod(u, f, A, B, W, N_newton, N_t):
    u_ = W.T @ u
    f_ = W.T @ f
    A_, B_ = get_reduced_matrices(A, B, W)
    U = [u_,]
    for _ in range(N_t):
        u_ = integration_step_pod(u_, f_, A_, B_, W, N_newton)
        U.append(u_)
    U = np.array(U) @ W.T
    return U

def global_pod(dataset, N_train):
    pod_data = {
        "basis": None,
        "sigmas": None,
    }
    _, sigmas, Vt = np.linalg.svd(np.transpose(dataset["solutions"][:N_train], [1, 0, 2]).reshape(-1, dataset["solutions"].shape[2]))
    W = Vt.T
    pod_data['basis'] = np.expand_dims(W, 0)
    pod_data['sigmas'] = np.expand_dims(sigmas, 0)
    return pod_data

def global_pod_II(dataset, N_train):
    pod_data = {
        "basis": None,
        "sigmas": None,
    }
    _, sigmas, Vt = np.linalg.svd(dataset["solutions"][:N_train].reshape(N_train, -1))
    W = Vt.T
    pod_data['basis'] = np.expand_dims(W, 0)
    pod_data['sigmas'] = np.expand_dims(sigmas, 0)
    return pod_data

def local_pod(dataset):
    pod_data = {
        "basis": [],
        "sigmas": []
    }
    W = []
    sigmas = []
    errors_intrusive = []
    errors_non_intrusive = []
    for i in range(dataset["solutions"].shape[0]):
        _, sigma, Vt = np.linalg.svd(dataset["solutions"][i])
        pod_data['basis'].append(Vt.T)
        pod_data['sigmas'].append(sigma)

    for key in pod_data.keys():
        pod_data[key] = np.array(pod_data[key])
    return pod_data

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    data = np.load(f'Burgers_{dataset_name}.npz')
    N_train = 900
    global_pod_data = global_pod(data, N_train)
    np.savez(f'Burgers_{dataset_name}_global_POD.npz', **global_pod_data)
    local_pod_data = local_pod(data)
    np.savez(f'Burgers_{dataset_name}_local_POD.npz', **local_pod_data)