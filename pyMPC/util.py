import numpy as np


def lagrange_matrices(Ad, Bd, Np, Nc=None, A_lag=None, B_lag=None):

    nx = Bd.shape[0]
    nu = Bd.shape[1]

    if Nc is None:
        Nc = Np

    if A_lag is None:
        A_lag = np.empty((Np * nx, nx))

    if B_lag is None:
        B_lag = np.zeros((Np * nx, Nc * nu))
    else:
        B_lag[:, (Nc-1)*nu:(Nc)*nu] = 0.0  # this part of the matrix must be emptied

    A_kp1 = Ad
    for k in range(Np):
        A_lag[k*nx:(k+1)*nx, :] = A_kp1
        A_kp1 = np.matmul(A_kp1, Ad)

    # First "column" of B_lag
    B_lag[:nx, :nu] = Bd
    B_lag[nx:, :nu] = A_lag[:-nx, :] @ Bd

    # Repeat on the diagonal
    for k in range(Np):
        B_tmp = B_lag[k * nx:(k + 1) * nx, :nu]
        # copy on the diagonal
        n_diag = min((Np - k, Nc-1))
        for p in range(1, n_diag):
            B_lag[(k+p)*nx:(k+p+1)*nx, p*nu:(p+1)*nu] = B_tmp
        for p in range(n_diag, Np - k):
            B_lag[(k+p)*nx:(k+p+1)*nx, (Nc-1)*nu:(Nc)*nu] += B_tmp

    return A_lag, B_lag


if __name__ == "__main__":

    # Constants #
    Ts = 0.2 # sampling time (s)
    M = 2    # mass (Kg)
    b = 0.3  # friction coefficient (N*s/m)


    #Ad = 2*np.eye(2)
    #Bd = np.array([
    #    [1.0],
    #    [2.0]])
    Ad = np.array([
        [1.0, Ts],
        [0, 1.0 - b / M * Ts]
    ])
    Bd = np.array([
        [0.11],
        [Ts / M]])
    nx = Bd.shape[0]
    nu = Bd.shape[1]

    Np = 20
    Nc = 10
    A_lag, B_lag = lagrange_matrices(Ad, Bd, Np, Nc)
