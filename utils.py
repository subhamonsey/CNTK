import os
import pickle

import numpy as np


def load_cifar(path="cifar-10-batches-py"):
    train_batches, train_labels = [], []

    for i in range(1, 6):
        print(os.path.join(path, f"data_batch_{i}"))
        with open(os.path.join(path, f"data_batch_{i}"), "rb") as f:
            # use latin1 to be compatible with Python2 pickles
            cifar_out = pickle.load(f, encoding="bytes")

        train_batches.append(cifar_out[b"data"])
        train_labels.extend(cifar_out[b"labels"])

    X_train = np.vstack(train_batches).reshape(-1, 3, 32, 32)
    y_train = np.array(train_labels)

    # load test batch
    with open(os.path.join(path, "test_batch"), "rb") as f:
        cifar_out = pickle.load(f, encoding="bytes")

    X_test = cifar_out[b"data"].reshape(-1, 3, 32, 32)
    y_test = np.array(cifar_out[b"labels"])

    # normalize
    X_train = (X_train / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)

    mean = X_train.mean(axis=(0, 2, 3))
    std = X_train.std(axis=(0, 2, 3))

    X_train = (X_train - mean[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]

    return (X_train, y_train), (X_test, y_test)
