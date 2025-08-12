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

def integration_step(u, f, A, B, N_steps):
    v = np.copy(u)
    for _ in range(N_steps):
        r = B @ v**2 / 2 + A @ v - u + f
        v = v - np.linalg.solve(B @ np.diag(v) + A, r)
    return v

def integrate_Burgers(u, f, A, B, N_newton, N_t):
    U = [u,]
    for _ in range(N_t):
        u = integration_step(u, f, A, B, N_newton)
        U.append(u)
    U = np.array(U)
    return U

def get_weights(N, n, alpha):
    w = np.fft.rfftfreq(N)
    w = 1 / (1 + alpha*(2*np.pi*w)**2)**n
    return w

def get_random_function(w, N):
    c = (np.random.randn(w.shape[0]) + np.random.randn(w.shape[0])*1j) * w
    f = np.fft.irfft(c, n=N)
    return f

def get_initial_conditions(w, N):
    f = get_random_function(w, N)
    return f * np.sqrt(N) * 5

def get_diffusion_coefficient(w, N):
    f = get_random_function(w, N)
    f = f * np.sqrt(N)
    f = 0.005 + (1 + np.tanh(30*f))/2 * 0.1
    return f

def get_Burgers_dataset_I(N_samples):
    N_x, N_t = 128, 64
    T_max = 0.1
    N_newton = 3

    x, x_extended = get_coordinates(N_x)
    project = np.sin(np.pi*x)
    t = np.linspace(0, T_max, N_t)[1:]
    dt = t[1] - t[0]
    h = x[1] - x[0]
    B = dt * get_B(N_x) / h

    n = 4
    alpha = 40
    w_diff = get_weights(N_x*2+1, n, alpha)

    n = 2
    alpha = 10
    w_u = get_weights(N_x, n, alpha)

    data = {
        "solutions": [],
        "diff": [],
        "forcing": [],
        "coords": x
    }

    a_extended = get_diffusion_coefficient(w_diff, N_x*2+1)*0 + 5e-2
    for _ in range(N_samples):
        u = get_initial_conditions(w_u, N_x) * project

        A = np.eye(N_x) - dt * get_A(a_extended) / h**2
        U = integrate_Burgers(u, u*0, A, B, N_newton, N_t)
        data["solutions"].append(U)

    data["diff"].append(a_extended)
    data["forcing"].append(np.zeros_like(u))
    for key in data.keys():
        data[key] = np.array(data[key])
    return data

def get_Burgers_dataset_II(N_samples):
    N_x, N_t = 128, 64
    T_max = 0.1
    N_newton = 3

    x, x_extended = get_coordinates(N_x)
    project = np.sin(np.pi*x)
    t = np.linspace(0, T_max, N_t)[1:]
    dt = t[1] - t[0]
    h = x[1] - x[0]
    B = dt * get_B(N_x) / h

    n = 4
    alpha = 40
    w_diff = get_weights(N_x*2+1, n, alpha)

    n = 2
    alpha = 10
    w_u = get_weights(N_x, n, alpha)

    data = {
        "solutions": [],
        "diff": [],
        "forcing": [],
        "coords": x
    }

    for _ in range(N_samples):
        a_extended = get_diffusion_coefficient(w_diff, N_x*2+1)
        u = get_initial_conditions(w_u, N_x) * project

        A = np.eye(N_x) - dt * get_A(a_extended) / h**2
        U = integrate_Burgers(u, u*0, A, B, N_newton, N_t)
        data["solutions"].append(U)
        data["diff"].append(a_extended)

    data["forcing"].append(np.zeros_like(u))
    for key in data.keys():
        data[key] = np.array(data[key])
    return data

def get_Burgers_dataset_III(N_samples):
    N_x, N_t = 128, 64
    T_max = 0.1
    N_newton = 3

    x, x_extended = get_coordinates(N_x)
    project = np.sin(np.pi*x)
    t = np.linspace(0, T_max, N_t)[1:]
    dt = t[1] - t[0]
    h = x[1] - x[0]
    B = dt * get_B(N_x) / h

    n = 4
    alpha = 40
    w_diff = get_weights(N_x*2+1, n, alpha)

    n = 2
    alpha = 10
    w_u = get_weights(N_x, n, alpha)

    n = 4
    alpha = 40
    w_f = get_weights(N_x, n, alpha)

    data = {
        "solutions": [],
        "diff": [],
        "forcing": [],
        "coords": x
    }

    for _ in range(N_samples):
        a_extended = get_diffusion_coefficient(w_diff, N_x*2+1)
        u = get_initial_conditions(w_u, N_x) * project
        f = 5e-2*get_initial_conditions(w_f, N_x)

        A = np.eye(N_x) - dt * get_A(a_extended) / h**2
        U = integrate_Burgers(u, f, A, B, N_newton, N_t)
        data["solutions"].append(U)
        data["diff"].append(a_extended)
        data["forcing"].append(f)

    for key in data.keys():
        data[key] = np.array(data[key])
    return data

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    N_samples = 1000
    if dataset_name == "I":
        data = get_Burgers_dataset_I(N_samples)
    elif dataset_name == "II":
        data = get_Burgers_dataset_II(N_samples)
    elif dataset_name == "III":
        data = get_Burgers_dataset_III(N_samples)
    np.savez(f'Burgers_{dataset_name}.npz', **data)