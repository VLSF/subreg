import jax.numpy as jnp

from jax import jacfwd
from jax.lax import dot_general

def compute_prediction(l, model, features1, features2, coords):
    features = features1*(1-l) + features2*l
    prediction = model(features, coords)
    n = prediction.reshape(prediction.shape[0], -1)
    return n

def Taylor_indicator(l, model, features1, features2, coords):
    prediction_0 = compute_prediction(l*0, model, features1, features2, coords)
    prediction_l = compute_prediction(l, model, features1, features2, coords)
    d_prediction_0 = jacfwd(compute_prediction)(l*0, model, features1, features2, coords)[..., 0]
    return jnp.mean(jnp.linalg.norm(prediction_0 + d_prediction_0*l - prediction_l, axis=1)/jnp.linalg.norm(prediction_l, axis=1))
        
def Taylor_indicator_scan(carry, ind):
    model, features, coordinates, l = carry
    smoothness_ind = Taylor_indicator(l, model, features[ind[0]], features[ind[1]], coordinates)
    return carry, smoothness_ind

def Frobenius_d_indicator(model, features1, features2, coordinates):
    return jnp.mean(jnp.linalg.norm(jacfwd(compute_prediction)(jnp.array([0.0,]), model, features1, features2, coordinates)[..., 0], axis=1))
    
def Frobenius_d_indicator_scan(carry, ind):
    model, features, coordinates = carry
    smoothness_ind = Frobenius_d_indicator(model, features[ind[0]], features[ind[1]], coordinates)
    return carry, smoothness_ind

def get_cosines(p1, p2):
    q1 = jnp.linalg.qr(p1.T)[0].T
    q2 = jnp.linalg.qr(p2.T)[0].T
    sigma = jnp.linalg.svdvals(dot_general(q1, q2, (((1,), (1,)), ((), ()))))
    return sigma

def mean_angle_indicator(l, model, features1, features2, coords):
    features = features1*(1-l) + features2*l
    prediction_l = model(features, coords)
    prediction_1 = model(features1, coords)
    return jnp.mean(get_cosines(prediction_1.reshape(prediction_1.shape[0], -1), prediction_l.reshape(prediction_l.shape[0], -1)))

def mean_angle_indicator_scan(carry, ind):
    model, features, coordinates, l = carry
    smoothness_ind = mean_angle_indicator(l, model, features[ind[0]], features[ind[1]], coordinates)
    return carry, smoothness_ind

def mean_angle_target_indicator(l, model, gt1, features1, features2, coords):
    features = features1*(1-l) + features2*l
    prediction_l = model(features, coords)
    return jnp.mean(get_cosines(gt1.reshape(gt1.shape[0], -1), prediction_l.reshape(prediction_l.shape[0], -1)))

def mean_angle_target_indicator_scan(carry, ind):
    model, features, targets, coordinates, l = carry
    smoothness_ind = mean_angle_target_indicator(l, model, targets[ind[0]], features[ind[0]], features[ind[1]], coordinates)
    return carry, smoothness_ind
