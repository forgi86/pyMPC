# conda install -c conda-forge slycot

import control
import numpy as np
import scipy as sp


def __first_dim__(X):
    if sp.size(X) == 1:
        m = 1
    else:
        m = sp.size(X,0)
    return m


def __second_dim__(X):
    if sp.size(X) == 1:
        m = 1
    else:
        m = sp.size(X,1)
    return  m

def kalman_filter_simple(A, B, C, D, Qn, Rn):
    nx = __first_dim__(A)
    nw = nx  # number of uncontrolled inputs
    nu = __second_dim__(B) # number of controlled inputs
    ny = __first_dim__(C)

    P,W,K, = control.dare(np.transpose(A), np.transpose(C), Qn, Rn)
    L = np.transpose(K) # Kalman gain
    return L,P,W


if __name__ == '__main__':

    # Constants #
    Ts = 0.2 # sampling time (s)
    M = 2    # mass (Kg)
    b = 0.3  # friction coefficient (N*s/m)

    Ad = np.array([
        [1.0, Ts],
        [0,  1.0 -b/M*Ts]
    ])

    Bd = np.array([
      [0.0],
      [Ts/M]])

    Cd = np.array([[1, 0]])
    Dd = np.array([[0]])

    [nx, nu] = Bd.shape # number of states and number or inputs
    ny = np.shape(Cd)[0]

    ## General design ##
    Bd_kal = np.hstack([Bd, Bd])
    Dd_kal = np.array([[0, 0]])
    Q_kal = np.array([[100]]) # nw x nw matrix, w general (here, nw = nu)
    R_kal = np.eye(ny) # ny x ny)
    L_general,P_general,W_general = kalman_filter(Ad, Bd_kal, Cd, Dd_kal, Q_kal, R_kal)

    # Simple design
    Q_kal = 10 * np.eye(nx)
    R_kal = np.eye(ny)
    L_simple,P_simple,W_simple  = kalman_filter_simple(Ad, Bd, Cd, Dd, Q_kal, R_kal)

    # Simple design written in general form
    Bd_kal = np.hstack([Bd, np.eye(nx)])
    Dd_kal = np.hstack([Dd, np.zeros((ny, nx))])
    Q_kal = 10 * np.eye(nx)#np.eye(nx) * 100
    R_kal = np.eye(ny) * 1
    L_gensim,P_gensim,W_gensim  = kalman_filter_simple(Ad, Bd, Cd, Dd, Q_kal, R_kal)

    assert(np.isclose(L_gensim[0], L_simple[0]))