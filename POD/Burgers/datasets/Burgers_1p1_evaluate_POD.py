import jax.numpy as jnp
import sys, os

from jax.lax import scan, dot_general
from jax import vmap

def get_coordinates(N):
    x = jnp.linspace(0, 1, N+2)[1:-1]
    x_extended = jnp.linspace(0, 1, 2*N + 3)[1:-1]
    return x, x_extended

def get_A(a_extended):
    A = jnp.diag(-a_extended[::2][:-1]-a_extended[::2][1:]) + jnp.diag(a_extended[::2][1:-1], k=-1) + jnp.diag(a_extended[::2][1:-1], k=+1)
    return A

def get_B(N):
    B = jnp.diag(jnp.ones((N-1,))/2, k=+1) - jnp.diag(jnp.ones((N-1,))/2, k=-1)
    B = B.at[0, :3].set(jnp.array([-3/2, 2, -1/2]))
    B = B.at[-1, -3:].set(jnp.array([1/2, -2, 3/2]))
    return B

def get_reduced_matrices(A, B, W):
    A_ = W @ A @ W.T
    B_ = W @ B
    return A_, B_

def integration_step_pod(u, f, A_, B_, W, N_steps):
    v = jnp.copy(u)
    for _ in range(N_steps):
        r = B_ @ (W.T @ v)**2 / 2 + A_ @ v - u + f
        v = v - jnp.linalg.solve(B_ @ jnp.diag(W.T @ v) @ W.T + A_, r)
    return v

def integrate_Burgers_pod(u, f, a_extended, B, W, N_newton, t, h):
    u_ = W @ u
    f_ = W @ f
    A = get_A(a_extended)
    A = jnp.eye(B.shape[1]) - (t[1] - t[0]) * get_A(a_extended) / h**2
    A_, B_ = get_reduced_matrices(A, B, W)
    def integration_step_(u, t):
        u = integration_step_pod(u, f_, A_, B_, W, N_newton)
        return u, u

    _, U = scan(integration_step_, u_, t)
    U = U @ W
    U = jnp.concatenate([jnp.expand_dims(u, 0), U], axis=0)
    return U

def compute_metrics(u0, forcing, targets, coordinates, basis, a_extended, T_max=0.1, N_newton=3, global_basis=False):
    N_t = targets.shape[1] - 1
    N = basis.shape[-1]
    t = jnp.linspace(0, T_max, N_t)
    dt = t[1] - t[0]
    h = coordinates[0, 1] - coordinates[0, 0]
    B = dt * get_B(N) / h
    get_projected_error = lambda basis, target: jnp.linalg.norm((target @ basis.T) @ basis - target) / jnp.linalg.norm(target)
    if global_basis:
        res = vmap(integrate_Burgers_pod, in_axes=(0, 0, 0, None, None, None, None, None))(u0, forcing, a_extended, B, basis, N_newton, t, h)
        intrusive_errors = vmap(jnp.linalg.norm)(targets - res) / vmap(jnp.linalg.norm)(targets)
        non_intrusive_errors = vmap(get_projected_error, in_axes=(None, 0))(basis, targets)
    else:
        res = vmap(integrate_Burgers_pod, in_axes=(0, 0, 0, None, 0, None, None, None))(u0, forcing, a_extended, B, basis, N_newton, t, h)
        intrusive_errors = vmap(jnp.linalg.norm)(targets - res) / vmap(jnp.linalg.norm)(targets)
        non_intrusive_errors = vmap(get_projected_error, in_axes=(0, 0))(basis, targets)
    return intrusive_errors, non_intrusive_errors

if __name__ == "__main__":
    N_bases = [5, 10, 15, 20, 25]
    dataset_name = sys.argv[1]
    
    data = jnp.load(f'Burgers_{dataset_name}.npz')
    
    if dataset_name == "III":
        a_extended = jnp.array(data["diff"])
        forcing = jnp.array(data["forcing"])
    elif dataset_name == "II":
        forcing = jnp.concatenate([data["forcing"],]*data["solutions"].shape[0], axis=0)
        a_extended = jnp.array(data["diff"])
    elif dataset_name == "I":
        forcing = jnp.concatenate([data["forcing"],]*data["solutions"].shape[0], axis=0)
        a_extended = jnp.concatenate([data["diff"],]*data["solutions"].shape[0], axis=0)

    targets = jnp.array(data["solutions"])
    coordinates = jnp.expand_dims(jnp.array(data["coords"]), axis=0)
    u0 = jnp.array(data["solutions"][:, 0])

    statistics = dict()
    for pod_type in ["local", "global"]:
        subspace_data = jnp.load(f'Burgers_{dataset_name}_{pod_type}_POD.npz')
        basis = jnp.array(subspace_data["basis"])
        statistics[f"{pod_type} intrusive"] = []
        statistics[f"{pod_type} non-intrusive"] = []
        for n in N_bases:
            if basis.shape[0] == 1:
                basis_ =  basis[0, :, :n].T
                global_basis = True
            else:
                basis_ =  jnp.transpose(basis[:, :, :n], (0, 2, 1))
                global_basis = False
            intrusive_errors, non_intrusive_errors = compute_metrics(u0, forcing, targets, coordinates, basis_, a_extended, global_basis=global_basis)
            statistics[f"{pod_type} intrusive"].append(intrusive_errors)
            statistics[f"{pod_type} non-intrusive"].append(non_intrusive_errors)
    for key in statistics.keys():
        statistics[key] = jnp.array(statistics[key])

    jnp.savez(f"Burgers_1p1_POD_JAX/Burgers_{dataset_name}_POD_statistics.npz", **statistics)
    header = 'filename,N_basis,pod_type,error'
    if not os.path.isfile("Burgers_1p1_POD_JAX/results.csv"):
        with open("Burgers_1p1_POD_JAX/results.csv", "w") as f:
            f.write(header)
    for key in statistics.keys():
        mean_errors = jnp.mean(statistics[key], axis=1)
        for i in range(len(N_bases)):
            data = f"\nBurgers_{dataset_name}_POD_statistics.npz,{N_bases[i]},{key},{mean_errors[i]}"
            with open("Burgers_1p1_POD_JAX/results.csv", "a") as f:
                f.write(data)