import os
import jax.numpy as jnp

from jax.lax import scan
from jax import vmap

def diffusion_1D(dif_coef):
    h = 1 / dif_coef.shape[0]
    main_diag = -(dif_coef[:-1] + dif_coef[1:])
    off_diag = dif_coef[1:-1]
    A = (jnp.diag(main_diag) + jnp.diag(off_diag, k=-1) + jnp.diag(off_diag, k=1)) / h**2
    return A

def convection_1D(conv_coef):
    h = 1 / (conv_coef.shape[0] + 1)
    B = (jnp.diag(conv_coef) - jnp.diag(conv_coef[1:], k=-1)) / (2*h)
    return B

def get_operator(convection, diffusion, reaction, dt):
    A = diffusion_1D(diffusion)
    B = convection_1D(convection)
    S = jnp.diag(reaction)
    M = (jnp.eye(S.shape[0]) - dt * (A - B + S))
    return M

def integrate(carry, t):
    M, u, s = carry # supply s*dt
    u = jnp.linalg.solve(M, u + s)
    return [M, u, s], u

def get_sample(carry, ind):
    P, u0, diffusion, convection, reaction, source, t, dt = carry
    M = P[ind].T @ get_operator(convection[ind], diffusion[ind, ::2], reaction[ind], dt) @ P[ind]
    carry_ = [M, P[ind].T @ u0[ind], P[ind].T @ source[ind]*dt]
    
    carry_, U = scan(integrate, carry_, t[1:])
    U = jnp.concatenate([u0[ind].reshape(1, -1), U @ P[ind].T], axis=0)
    return carry, U

def get_projection(carry, ind):
    P, solutions = carry
    U = (solutions[ind] @ P[ind]) @ P[ind].T
    return carry, U

def get_errors(POD_basis, data, N_basis):
    t = data["t"]
    dt = t[1] - t[0]
    carry = [POD_basis[:, :, :N_basis], data["solutions"][:, 0, :], data["diffusion"], data["convection"], data["reaction"], data["source"], t, dt]
    _, sol_approximate = scan(get_sample, carry, jnp.arange(data["solutions"].shape[0]))
    errors_intrusive = vmap(jnp.linalg.norm)(data["solutions"] - sol_approximate) / vmap(jnp.linalg.norm)(data["solutions"])
    carry = [POD_basis[:, :, :N_basis], data["solutions"]]
    _, sol_approximate = scan(get_projection, carry, jnp.arange(data["solutions"].shape[0]))
    errors_non_intrusive = vmap(jnp.linalg.norm)(data["solutions"] - sol_approximate) / vmap(jnp.linalg.norm)(data["solutions"])
    return errors_intrusive, errors_non_intrusive


if __name__ == "__main__":

    data = jnp.load("CDR_dataset.npz")
    V_local = jnp.load("CDR_local_POD.npz")["basis"]
    V_global = jnp.load("CDR_global_POD.npz")["basis"]

    N_bases = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    statistics = dict()
    for pod_type in ["local", "global"]:
        statistics[f"{pod_type} intrusive"] = []
        statistics[f"{pod_type} non-intrusive"] = []
        for n in N_bases:
            intrusive_errors, non_intrusive_errors = get_errors(V_local, data, n) if pod_type == 'local' else get_errors(V_global, data, n)
            statistics[f"{pod_type} intrusive"].append(intrusive_errors)
            statistics[f"{pod_type} non-intrusive"].append(non_intrusive_errors)
    for key in statistics.keys():
        statistics[key] = jnp.array(statistics[key])
        
    jnp.savez(f"CDR_1p1_POD/CDR_POD_statistics.npz", **statistics)
    header = 'N_basis,pod_type,error'
    if not os.path.isfile("CDR_1p1_POD/results.csv"):
        with open("CDR_1p1_POD/results.csv", "w") as f:
            f.write(header)

    for key in statistics.keys():
        mean_errors = jnp.mean(statistics[key], axis=1)
        for i in range(len(N_bases)):
            data = f"\n{N_bases[i]},{key},{mean_errors[i]}"
            with open("CDR_1p1_POD/results.csv", "a") as f:
                f.write(data)