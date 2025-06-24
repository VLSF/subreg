import numpy as np
import sys

if __name__ == "__main__":
    dataset = sys.argv[1]
    n_eig = int(sys.argv[2])

    train_ind = [range(14), range(14, 27), range(27, 40)]
    test_ind = range(40, 50)

    for i, train_i in enumerate(train_ind):
        data = {
            "eigenvalues": [],
            "features": [],
            "eigenvectors": [],
            "A_data": []
        }
        for j in train_i:
            data_ = np.load(f"{dataset}/{dataset}_{j}.npz")
            for key in data_.keys():
                if key == "eigenvectors" or key == "eigenvalues":
                    data[key].append(np.copy(data_[key][:, :n_eig]))
                else:
                    data[key].append(np.copy(data_[key]))
            del data_
            
        for key in data.keys():
            data[key] = np.concatenate(data[key], axis=0)
        np.savez(f"{dataset}/{dataset}_train_{i}.npz", **data)

    data = {
        "eigenvalues": [],
        "features": [],
        "eigenvectors": [],
        "A_data": []
    }
    for j in test_ind:
        data_ = np.load(f"{dataset}/{dataset}_{j}.npz")
        for key in data_.keys():
            if key == "eigenvectors" or key == "eigenvalues":
                data[key].append(np.copy(data_[key][:, :n_eig]))
            else:
                data[key].append(np.copy(data_[key]))
        del data_
        
    for key in data.keys():
        data[key] = np.concatenate(data[key], axis=0)
    np.savez(f"{dataset}/{dataset}_test.npz", **data)
