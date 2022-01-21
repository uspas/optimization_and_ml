import numpy as np


# generate a lot of data for fitting
def generate_data():
    dim = 2000
    x = np.random.rand(100000, dim)
    w = (np.random.rand(dim) * 2.0 - 1.0)

    f = w @ x.T
    f = f / np.max(f)
    f = f + np.random.randn(*f.shape) * 0.1

    #add f column
    data = np.hstack((x, f.reshape(-1, 1)))
    np.save('complex_dataset_large.npy', data)
    np.save('complex_dataset_weights_large.npy', w)


if __name__ == '__main__':
    generate_data()
