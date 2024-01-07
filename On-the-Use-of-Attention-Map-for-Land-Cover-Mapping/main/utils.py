# Author: Suesarn Wilainuch

"""Helper functions for On the Use of Attention Map for Land Cover Mapping."""

import os
import h5py
import numpy as np

def load_h5(path):
    """
    Load parameters from .h5 file (h5py).
    Parameters
    ----------
    path: str
        File path.
    Examples
    --------
    >>>
    Returns
    -------

    """
    dict_output = dict()
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            dict_output[key] = f[key][()]
    return dict_output


def load_and_preprocess(filename):
    data = load_h5(filename)

    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    y_train_U = data['y_unet_train']
    y_test_U = data['y_unet_test']

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return [x_train, y_train_U, x_test, y_test_U, y_train, y_test]


def select_supervised_samples(dataset, n_samples=None):
    X, y = dataset
    X_list, y_list = list(), list()
    for i in range(n_samples):
      ix = np.random.randint(0, len(X))
      X_list.append(X[ix])
      y_list.append(y[ix])

    print("random image:", len(X_list))
    print("random tag:", len(y_list))
    return [np.asarray(X_list), np.asarray(y_list)]
