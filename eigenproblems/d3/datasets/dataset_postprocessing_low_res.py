import jax.numpy as jnp

if __name__ == "__main__":
    train_ind = range(45)
    test_ind = range(45, 50)
    train_data = {
        "features": [],
        "eigenvectors": [],
        "A_data": []
    }
    for i in train_ind:
        data_ = jnp.load(f"lone_flattop_low_res/lone_flattop_{i}.npz")
        for key in train_data.keys():
            train_data[key].append(data_[key])
    
    for key in train_data.keys():
        train_data[key] = jnp.concatenate(train_data[key], axis=0)

    test_data = {
        "features": [],
        "eigenvectors": [],
        "A_data": []
    }
    for i in test_ind:
        data_ = jnp.load(f"lone_flattop_low_res/lone_flattop_{i}.npz")
        for key in test_data.keys():
            test_data[key].append(data_[key])
    
    for key in test_data.keys():
        test_data[key] = jnp.concatenate(test_data[key], axis=0)

    global_data = jnp.load("lone_flattop_low_res/lone_flattop_global.npz")
    for key in global_data.keys():
        print(key, global_data[key].shape)

    jnp.savez("lone_flattop_low_res/lone_flattop_low_res_train_0.npz", **train_data)
    jnp.savez("lone_flattop_low_res/lone_flattop_low_res_test.npz", **test_data)
    jnp.savez("lone_flattop_low_res/lone_flattop_low_res_global.npz", **global_data)