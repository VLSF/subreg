import sys
import os
import jax.numpy as jnp
from jax.lax import scan
from jax import random

def get_weights(N, n, alpha):
    w = jnp.fft.rfftfreq(N)
    w = 1 / (1 + alpha*(2*jnp.pi*w)**2)**n
    return w

def get_random_function(w, N, key):
    c = random.normal(key, w.shape[0], dtype=jnp.complex64) * w
    f = jnp.fft.irfft(c, n=N)
    return f

def get_initial_conditions(w, x, key):
    f = get_random_function(w, x.shape[0], key)
    f = f * jnp.sqrt(x.shape[0]) * 5 * jnp.sin(jnp.pi*x)
    return f

def get_diffusion_coefficient(w, x, key):
    f = get_random_function(w, x.shape[0], key)
    f = f * jnp.sqrt(x.shape[0])
    f = 0.0005 + (1 + jnp.tanh(5*f))/2 * 0.2
    return f

def get_b_coefficient(w, x, key):
    f = get_random_function(w, x.shape[0], key)
    f = f * jnp.sqrt(x.shape[0]) * 10
    return f

def get_random_subspace(w, x, rank, key):
    keys = random.split(key, rank)
    w = jnp.stack([get_random_function(w, x.shape[0], key) for key in keys])
    q = jnp.linalg.qr(w.T)[0]
    return q

def diffusion_1D(dif_coef):
    h = 1 / dif_coef.shape[0]
    main_diag = -(dif_coef[:-1] + dif_coef[1:])
    off_diag = dif_coef[1:-1]
    A = (jnp.diag(main_diag) + jnp.diag(off_diag, k=-1) + jnp.diag(off_diag, k=1)) / h**2
    return A

def integrate(carry, t):
    M, phi, b, dt = carry
    phi = jnp.linalg.solve(M, phi - dt * b)
    return [M, phi, b, dt], phi

def get_weight_matrix(A, dt):
    vals, vecs = jnp.linalg.eigh(A)
    weights11 = jnp.ones((vals.shape[0], vals.shape[0])) - dt * (vals.reshape(-1, 1) + vals.reshape(1, -1))
    weigths12 = jnp.ones((vals.shape[0], vals.shape[0])) - dt * vals.reshape(-1, 1)
    return weights11, weigths12, vecs

def integrate_C(carry, t):
    C11_, C12_, w11, w12, W_, U, l_c, dt = carry
    CW = C11_ @ W_
    C12_ = (C12_ - dt * C11_ - dt * CW @ (W_.T @ C12_) * l_c) / w12
    C11_ = (C11_ - dt * CW @ CW.T * l_c) / w11
    return [C11_, C12_, w11, w12, W_, U, l_c, dt], [U @ C11_ @ U.T, U @ C12_ @ U.T]

def get_optimal_control(psi, A, W, b, t, l_c):
    # output alpha, beta
    # closed-loop optimal control alpha @ phi + beta
    dt = t[1] - t[0]
    w11, w12, U = get_weight_matrix(A, dt)
    W_ = U.T @ W
    psi_ = U.T @ psi
    C11_ = psi_ @ psi_.T
    C12_ = C11_ * 0
    _, [C11, C12] = scan(integrate_C, [C11_, C12_, w11, w12, W_, U, l_c, dt], t[1:])
    C11 = jnp.concatenate([C11[::-1], jnp.expand_dims(U @ C11_ @ U.T, 0)], axis=0)
    C12 = jnp.concatenate([C12[::-1], jnp.expand_dims(C12_, 0)], axis=0)
    return -W.T @ C11 * l_c, -W.T @ C12 @ b * l_c

def integrate_closed_loop(carry, i):
    M, W, phi, b, alpha, beta, dt = carry
    phi = jnp.linalg.solve(M, phi - dt * b + dt * W @ (alpha[i] @ phi + beta[i]))
    return [M, W, phi, b, alpha, beta, dt], phi

def solve_sylvester_eig(A, R):
    # A X + X A.T = R
    vals, vecs = jnp.linalg.eigh(A)
    C = (vecs @ ((vecs.T @ (R @ vecs)) / (vals.reshape(-1, 1) + vals.reshape(1, -1)))) @ vecs.T
    return C

def get_gramians_eig(A, B, C):
    # A X + X A.T = - B B.T
    W_c = solve_sylvester_eig(A, -B @ B.T)
    W_o = solve_sylvester_eig(A, -C.T @ C)
    return W_c, W_o

def get_balanced_truncation_coords(A, B, C, eps=1e-5):
    W_c, W_o = get_gramians_eig(A, B, C)
    sigma_c, U_c = jnp.linalg.eigh(W_c)
    order = jnp.argsort(sigma_c)[::-1]
    sigma_c = sigma_c[order]
    U_c = U_c[:, order]
    T_1 = U_c * jnp.sqrt(jnp.abs(sigma_c) + eps).reshape(1, -1)
    T_1inv = jnp.linalg.inv(T_1)
    W_o_ = T_1.T @ W_o @ T_1
    sigma_o, U_o = jnp.linalg.eigh(W_o_)
    order = jnp.argsort(sigma_o)[::-1]
    sigma_o = sigma_o[order]
    U_o = U_o[:, order]
    T_2 = U_o * jnp.power(jnp.abs(sigma_o) + eps, -1/4).reshape(1, -1)
    T = T_1 @ T_2
    return T, sigma_o

def get_sample(key, control_rank, observations_rank, l_c = 15):
    N = 128
    Nt = 128
    n = 4
    
    x = jnp.linspace(0, 1, N+2)[1:-1]
    x_extended = jnp.linspace(0, 1, 2*N+3)[1:-1]
    
    alpha = 5
    w_1 = get_weights(N, n, alpha)
    w_1ex = get_weights(x_extended.shape[0], n, alpha)
    
    alpha = 6
    w_2 = get_weights(N, n, alpha)
    
    alpha = 5
    w_3 = get_weights(N, n, alpha)
    
    t = jnp.linspace(0, 5, Nt)
    dt = t[1] - t[0]
    
    keys = random.split(key, 7)
    W = get_random_subspace(w_3, x, control_rank, keys[0])
    psi = get_random_subspace(w_1, x, observations_rank, keys[1])
    
    phi = get_initial_conditions(w_1, x, keys[2])
    diffusion = get_diffusion_coefficient(w_1ex, x_extended, keys[3])
    A = diffusion_1D(diffusion[::2])
    b = get_b_coefficient(w_2, x, keys[4])
    
    alpha, beta = get_optimal_control(psi, A, W, b, t, l_c)
    M = (jnp.eye(A.shape[0]) - dt * A)
    carry_ = [M, phi, b, dt]
    carry_closed_loop = [M, W, phi, b, alpha, beta, dt]
    
    _, phi_0 = scan(integrate, carry_, jnp.arange(Nt-1))
    _, phi_opt = scan(integrate_closed_loop, carry_closed_loop, jnp.arange(Nt-1))
    
    W_c, W_o = get_gramians_eig(A, W, psi.T)
    T, sigma = get_balanced_truncation_coords(A, W, psi.T)

    data = {
        "coords": x,
        "coords_extended": x_extended,
        "t": t,
        "W": W,
        "diffusion": diffusion,
        "b": b,
        "psi": psi,
        "phi": phi,
        "solution": phi_0,
        "alpha_optimal": alpha,
        "beta_optimal": beta,
        "solution_controlled": phi_opt,
        "basis": T,
        "singular_values": sigma
    }
    
    return data

def compute_truncated_control(diffusion, W, psi, b, phi, T, t, N_truncate, l_c = 15):
    A = diffusion_1D(diffusion[::2])
    Q = jnp.linalg.qr(T[:, :N_truncate])[0]
    A_ = Q.T @ A @ Q
    W_ = Q.T @ W
    psi_ = Q.T @ psi
    M_ = (jnp.eye(A_.shape[0]) - (t[1] - t[0]) * A_)
    b_ = Q.T @ b
    phi_ = Q.T @ phi
    alpha_, beta_ = get_optimal_control(psi_, A_, W_, b_, t, l_c)
    carry_closed_loop_ = [M_, W_, phi_, b_, alpha_, beta_, t[1] - t[0]]
    _, phi_tr = scan(integrate_closed_loop, carry_closed_loop_, jnp.arange(t.shape[0]-1))
    phi_tr = phi_tr @ Q.T
    return alpha_, beta_, phi_tr

if __name__ == '__main__':
    '''
    shapes are
    coords (128,)
    coords_extended (257,)
    t (128,)
    W (1200, 128, 5)
    diffusion (1200, 257)
    b (1200, 128)
    psi (1200, 128, 5)
    phi (1200, 128)
    solution (1200, 127, 128)
    alpha_optimal (1200, 128, 5, 128)
    beta_optimal (1200, 128, 5)
    solution_controlled (1200, 127, 128)
    basis (1200, 128, 128)
    singular_values (1200, 128)

    solution and controlled solution are saved without initial value which is stored in phi
    '''
    control_rank, observations_rank = int(sys.argv[1]), int(sys.argv[2])
    N_samples = 1200
    key = random.PRNGKey(44)
    keys = random.split(key, N_samples)
    dataset_name = f"heat_control_{control_rank}_{observations_rank}"
    stack_keys = ["W", "diffusion", "b", "psi", "phi", "solution", "alpha_optimal", "beta_optimal", "solution_controlled", "basis", "singular_values"]
    data = get_sample(keys[1], control_rank, observations_rank)
    N_truncates = [5, 10, 15, 20, 25, 30]
    solution_errors = []
    observation_errors = []
    for k in stack_keys:
        data[k] = [data[k],]
    for key in keys[1:]:
        data_ = get_sample(key, control_rank, observations_rank)
        for k in stack_keys:
            data[k].append(data_[k])
        s_e, s_o = [], []
        for n in N_truncates:
            _, _, phi_tr = compute_truncated_control(data_['diffusion'], data_['W'], data_['psi'], data_['b'], data_['phi'], data_['basis'], data_['t'], n)
            s_o.append(jnp.linalg.norm(data_['psi'].T @ data_['solution_controlled'][-1] - data_['psi'].T @ phi_tr[-1]) / jnp.linalg.norm(data_['psi'].T @ data_['solution_controlled'][-1]))
            s_e.append(jnp.linalg.norm(data_['solution_controlled'][-1] - phi_tr[-1]) / jnp.linalg.norm(data_['solution_controlled'][-1]))
        solution_errors.append(jnp.array(s_e))
        observation_errors.append(jnp.array(s_o))
    
    for k in stack_keys:
        data[k] = jnp.stack(data[k])
    jnp.savez(dataset_name + ".npz", **data)

    solution_errors = jnp.array(solution_errors)
    observation_errors = jnp.array(observation_errors)
    jnp.savez("balanced_truncation/" + dataset_name + "_metrics.npz", solution_errors=solution_errors, observation_errors=observation_errors)
    header = "dataset_name,N_basis,solution_error,observation_error"
    if not os.path.isfile(f'balanced_truncation/results.csv'):
        with open(f'balanced_truncation/results.csv', "w") as f:
            f.write(header)

    write_data = ""
    for i in range(len(N_truncates)):
        write_data += f"\n{dataset_name},{N_truncates[i]},{jnp.mean(solution_errors[-100:, i])},{jnp.mean(observation_errors[-100:, i])}"
    with open(f'balanced_truncation/results.csv', "a") as f:
        f.write(write_data)