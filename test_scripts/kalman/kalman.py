import control
import numpy as np
import scipy as sp

# conda install -c conda-forge slycot


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

def kalman_filter(A, B, C, D, Qn, Rn, Nn=None):
    """ Design a Kalman filter for the discrete-time system
     x_{k+1} = Ax_{k} + Bu_{k} + Gw_{k}
     y_{k} = Cx_{k} + Du_{k} + Hw_{k} + v_{k}
     with known inputs u and sctochastic disturbances v, w.
     In particular, v and w are zero mean, white Gaussian noise sources with
     E[vv'] = Qn, E[ww'] = Rn, E['wv'] = Nn

    The Kalman filter has structure
     \hat x_{k+1} = Ax_{k} + Bu_{k} + L(y_{k} - C\hat x{k} - Du_{k})
     \hat y_{k}   = Cx_k + Du_k
    """
    nx = np.shape(A)[0]
    nw = np.shape(Qn)[0] # number of uncontrolled inputs
    nu = np.shape(B)[1] - nw # number of controlled inputs
    ny = np.shape(C)[0]

    if Nn is None:
        Nn = np.zeros((nw, ny))

    E = np.eye(nx)
    Bu = B[:, 0:nu]
    Du = D[:, 0:nu]
    Bw = B[:, nu:]
    Dw = D[:, nu:]

    Hn = Dw @ Nn
    Rb = Rn + Hn + np.transpose(Hn) + Dw @ Qn @ np.transpose(Dw)
    Qb = Bw @ Qn @ np.transpose(Bw)
    Nb = Bw @ (Qn @ np.transpose(Dw) + Nn)

    # Enforce symmetry
    Qb = (Qb + np.transpose(Qb))/2
    Rb = (Rb+np.transpose(Rb))/2

    P,W,K, = control.dare(np.transpose(A), np.transpose(C), Qb, Rb, Nb, np.transpose(E))

    L = np.transpose(K) # Kalman gain
    return L,P,W

def kalman_filter_simple(A, B, C, D, Qn, Rn):
    r"""Design a Kalman filter for the discrete-time system

    .. math::
        \begin{split}
         x_{k+1} &= Ax_{k} + Bu_{k} + Iw_{k}\\
         y_{k}  &= Cx_{k} + Du_{k} + I v_{k}
        \end{split}

    with known inputs u and stochastic disturbances w and v.
    In particular, w and v are zero mean, white Gaussian noise sources with
    E[vv'] = Qn, E[ww'] = Rn, E['wv'] = 0

    The Kalman filter has structure

    .. math::
        \begin{split}
        \hat x_{k+1} &= Ax_{k} + Bu_{k} + L(y_{k} - C\hat x{k} - Du_{k})\\
        \hat y_{k}   &= Cx_k + Du_k
        \end{split}
    """

    nx = __first_dim__(A)
    nw = nx  # number of uncontrolled inputs
    nu = __second_dim__(B) # number of controlled inputs
    ny = __first_dim__(C)

    P,W,K, = control.dare(np.transpose(A), np.transpose(C), Qn, Rn)
    L = np.transpose(K) # Kalman gain
    return L,P,W


class LinearStateEstimator:
    def __init__(self, x0, A, B, C, D, L):

        self.x = x0
        self.y = C @ x0
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.L = L

        self.nx = __first_dim__(A)
        self.nu = __second_dim__(B) # number of controlled inputs
        self.ny = __first_dim__(C)

    def out_y(self,u):
        return self.y

    def predict(self, u):
        self.x = self.A @ self.x + self.B @u  # x[k+1|k]
        self.y = self.C @ self.x + self.D @u
        return self.x

    def update(self, y_meas):
        self.x = self.x + self.L @ (y_meas - self.y)  # x[k+1|k+1]
        return self.x

    def sim(self, u_seq, x=None):

        if x is None:
            x = self.x
        Np = __first_dim__(u_seq)
        nu = __second_dim__(u_seq)
        assert(nu == self.nu)

        y = np.zeros((Np,self.ny))
        x_tmp = x
        for i in range(Np):
            u_tmp = u_seq[i]
            y[i,:] = self.C @ x_tmp + self.D @ u_tmp
            x_tmp = self.A @x_tmp + self.B @ u_tmp

        #y[Np] = self.C @ x_tmp + self.D @ u_tmp # not really true for D. Here it is 0 anyways
        return y

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
