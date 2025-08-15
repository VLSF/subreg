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

def get_convection_coefficient(w, x, key):
    f = get_random_function(w, x.shape[0], key)
    f = f * jnp.sqrt(x.shape[0]) * 0.25
    return f

def get_source_coefficient(w, x, key):
    f = get_random_function(w, x.shape[0], key)
    f = f * jnp.sqrt(x.shape[0]) * 10
    return f   

def get_reaction_coefficient(w, x, key):
    f = get_random_function(w, x.shape[0], key)
    f = f * jnp.sqrt(x.shape[0])
    return f    

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

def get_sample(carry, key):
    w_1, w_1ex, w_2, x, x_extended, t = carry
    keys = random.split(key, 5)
    dt = t[1] - t[0]
    
    u0 = get_initial_conditions(w_1, x, keys[0])
    diffusion = get_diffusion_coefficient(w_1ex, x_extended, keys[1])
    convection = get_convection_coefficient(w_2, x, keys[2])
    source = get_source_coefficient(w_2, x, keys[3])
    reaction = get_reaction_coefficient(w_2, x, keys[4])
    M = get_operator(convection, diffusion[::2], reaction, dt)
    carry_ = [M, u0, source*dt]
    
    carry_, U = scan(integrate, carry_, t[1:])
    U = jnp.concatenate([u0.reshape(1, -1), U], axis=0)
    return carry, (U, diffusion, convection, source, reaction)

def get_dataset(N_samples, key):
    N = 128
    Nt = 64
    n = 4

    x = jnp.linspace(0, 1, N+2)[1:-1]
    x_extended = jnp.linspace(0, 1, 2*N+3)[1:-1]
    
    alpha = 5
    w_1 = get_weights(N, n, alpha)
    w_1ex = get_weights(x_extended.shape[0], n, alpha)

    alpha = 6
    w_2 = get_weights(N, n, alpha)

    t = jnp.linspace(0, 5, Nt)
    dt = t[1] - t[0]
    keys = random.split(key, N_samples)

    carry = [w_1, w_1ex, w_2, x, x_extended, t]
    _, (U, diffusion, convection, source, reaction) = scan(get_sample, carry, keys)
    data = {
        "solutions": U,
        "diffusion": diffusion,
        "convection": convection,
        "reaction": reaction,
        "source": source,
        "x": x,
        "t": t
    }
    return data

if __name__=="__main__":
    key = random.PRNGKey(44)
    N_samples = 1000
    data = get_dataset(N_samples, key)
    jnp.savez("CDR_dataset.npz", **data)