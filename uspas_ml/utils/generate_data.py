import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    n = 5
    x = np.linspace(0, 1, n)
    xx = np.meshgrid(x, x, x)

    pts = np.vstack([ele.ravel() for ele in xx]).T
    f = np.sin(2 * np.pi * pts[:, 0] / 1.0) * np.sin(2 * np.pi * pts[:, 1] / 2.0) + pts[:, 2]
    f = f.reshape(-1, 1)

    data = np.hstack([pts, f])
    print(pts.shape)
    print(f.shape)
    print(data.shape)

    np.save('../lab_05_bayesian_fitting/bayesian_fitting_data.npy', data)


if __name__ == '__main__':
    generate_data()
