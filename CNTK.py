import argparse

import cupy as cp
import numpy as np
import scipy.linalg
from tqdm import trange

from cuda import conv3, trans
from utils import load_cifar

parser = argparse.ArgumentParser(
    description="Convolutional Neural Tangent Kernel (CNTK) for CIFAR-10"
)
parser.add_argument(
    "--depth", default=21, type=int, help="depth of CNTK (#conv layers + 1)"
)
parser.add_argument(
    "--gap",
    default="yes",
    type=str,
    help="whether GAP (global average pooling) is used",
)
parser.add_argument(
    "--fix",
    default="yes",
    type=str,
    help="whether first layer and last layer are fixed (or trained) (see Section 4.2 in our paper)",
)
args = parser.parse_args()

d = args.depth
gap = args.gap == "yes"
fix = args.fix == "yes"


# Calculate diagonal entries of $\Sigma^{(h)}(x, x)$ and their reciprocals. See Section 4.3 in our paper.
def xx(x):
    RL, iRL = [1.0], [1.0]

    S = cp.matmul(x.T, x).reshape(32, 32, 32, 32)
    conv3((S, S))
    T = cp.zeros((32, 32, 32, 32), dtype=cp.float32)

    if not fix:
        T += S

    for i in range(1, d - 1):
        L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
        iL = 1.0 / L

        RL.append(L)
        iRL.append(iL)

        trans((S, T, L, L, iL, iL))
        conv3((S, S))
        conv3((T, T))

    L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
    iL = 1.0 / L

    RL.append(L)
    iRL.append(iL)

    trans((S, T, L, L, iL, iL))

    if fix:
        T -= S

    return RL, iRL


# Calculate the kernel value of x and z.
# Lx and Lz are diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$.
# iLx and iLz are reciprocals of diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$.
def xz(x, z, Lx, Lz, iLx, iLz):
    S = cp.matmul(x.T, z).reshape(32, 32, 32, 32)
    conv3((S, S))
    T = cp.zeros((32, 32, 32, 32), dtype=cp.float32)

    if not fix:
        T += S

    for i in range(1, d - 1):
        trans((S, T, Lx[i], Lz[i], iLx[i], iLz[i]))
        conv3((S, S))
        conv3((T, T))

    trans((S, T, Lx[-1], Lz[-1], iLx[-1], iLz[-1]))

    if fix:
        T -= S

    return cp.mean(T) if gap else cp.trace(T.reshape(1024, 1024))


# Load CIFAR-10.
print("Loading CIFAR-10...")

(X_train, y_train), (X_test, y_test) = load_cifar()
X = np.concatenate((X_train, X_test), axis=0)
N = X.shape[0]

N_train = X_train.shape[0]
N_test = X_test.shape[0]
X = cp.asarray(X).reshape(-1, 3, 1024)

# Calculate diagonal entries.
print("\nCalculating diagonal entries...")

L, iL = [], []
for i in trange(N):
    Lx, iLx = xx(X[i])
    L.append(Lx)
    iL.append(iLx)

# Calculate kernel values.
## Below we provide a naive implementation using for-loops.
## Parallelize this part according to your specific computing environment to utilize multiple GPUs.
print("\nCalculating kernel values...")

H = np.zeros((N, N), dtype=np.float32)
for i in trange(N):
    for j in trange(N, leave=False):
        H[i][j] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j])
##

# Solve kernel regression.
print("\nSolving kernel regression...")

Y_train = np.ones((N_train, 10)) * -0.1
for i in trange(N_train):
    Y_train[i][y_train[i]] = 0.9

u = H[N_train:, :N_train].dot(scipy.linalg.solve(H[:N_train, :N_train], Y_train))
print("\nTest Accuracy:", 1.0 * np.sum(np.argmax(u, axis=1) == y_test) / N_test)
