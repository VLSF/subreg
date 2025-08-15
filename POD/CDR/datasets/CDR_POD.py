import jax.numpy as jnp
from jax.lax import scan

def get_global_POD_basis(solutions):
    _, sigmas, Vt = jnp.linalg.svd(solutions.reshape(-1, solutions.shape[-1]))
    return Vt.T, sigmas

def get_local_POD_basis(solutions):
    _, (V, sigmas) = scan(lambda carry, i: (carry, get_global_POD_basis(carry[i])), solutions, jnp.arange(solutions.shape[0]))
    return V, sigmas

if __name__ == "__main__":
    N_train = 900
    data = jnp.load("CDR_dataset.npz")
    V_local, sigmas_local = get_local_POD_basis(data["solutions"])
    local_POD_data = {
        "basis": V_local,
        "sigmas": sigmas_local
    }
    jnp.savez("CDR_local_POD.npz", **local_POD_data)

    V, sigmas = get_global_POD_basis(data["solutions"][:N_train])
    global_POD_data = {
        "basis": jnp.expand_dims(V, 0),
        "sigmas": sigmas
    }
    jnp.savez("CDR_global_POD.npz", **global_POD_data)