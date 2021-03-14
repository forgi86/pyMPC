import numpy as np
import scipy.linalg
from pyMPC.util import lagrange_matrices


class MPCController:
    """ This class implements an MPC controller

    Attributes
    ----------
    Ad : 2D array_like. Size: (nx, nx)
         Discrete-time system matrix Ad.
    Bd : 2D array-like. Size: (nx, nu)
         Discrete-time system matrix Bd.
    n_p : int
        Prediction horizon. Default value: 20.
    n_c : int
        Control horizon. Default value: Np.
    x_0 : 1D array_like. Size: (nx,)
         System state at time instant 0. If None, it is set to np.zeros(nx)
    x_ref : 1D array-like. Size: (nx,)
           System state reference (aka target, set-point).
    u_ref : 1D array-like. Size: (nu, ).
           System input reference. If None, it is set to np.zeros(nx)
    u_minus1 : 1D array_like
             Input value assumed at time instant -1. If None, it is set to uref.
    Q_x : 2D array_like
         State weight matrix. If None, it is set to eye(nx).
    Q_u : 2D array_like
         Input weight matrix. If None, it is set to zeros((nu,nu)).
    Q_du : 2D array_like
    """

    def __init__(self, Ad, Bd, n_p=20, n_c=None, x_0=None, x_ref=None, u_ref=None, u_minus1=None,
                 Q_x=None, Q_u=None, Q_du=None):
        self.Ad = Ad
        self.Bd = Bd
        self.n_x, self.n_u = self.Bd.shape # number of states and number or inputs
        self.n_p = n_p  # assert

        if n_c is not None:
            if n_c <= n_p:
                self.n_c = n_c
            else:
                raise ValueError("Nc should be <= Np!")
        else:
            self.n_c = self.n_p

        # x0 handling
        if x_0 is not None:
            self.x_0 = x_0
        else:
            self.x_0 = np.zeros(self.n_x)
        # reference handing
        if x_ref is not None:
            self.x_ref = x_ref  # assert...
        else:
            self.x_ref = np.zeros(self.n_x)
        self.x_ref_all = np.kron(np.ones((n_p, 1)), self.x_ref.reshape(-1, 1))

        if u_ref is not None:
            self.u_ref = u_ref  # assert...
        else:
            self.u_ref = np.zeros(self.n_u)
        self.u_ref_all = np.kron(np.ones((n_c, 1)), self.u_ref.reshape(-1, 1))

        if u_minus1 is not None:
            self.u_minus1 = u_minus1.reshape(-1, 1)

        else:
            self.u_minus1 = self.u_ref

        # weights handling
        if Q_x is not None:
            self.Q_x = Q_x
        else:
            self.Q_x = np.zeros((self.n_x, self.n_x))

        if Q_u is not None:
            self.Q_u = Q_u
        else:
            self.Q_u = np.zeros((self.n_u, self.n_u))

        if Q_du is not None:
            self.Q_du = Q_du
        else:
            self.Q_du = np.zeros((self.n_u, self.n_u))

        self.u_failure = self.u_ref  # value provided when the MPC solver fails.

        self.x_0_rh = None
        self.u_minus1_rh = None

        self.A_lag = None
        self.B_lag = None

        self.Q_cal_x = None
        self.Q_cal_u = None
        self.Q_cal_du = None

        self.P = None
        self.P_inv = None

        # Linear terms
        self.p_x_0 = None
        self.p_x_ref_all = None
        self.p_u_minus1 = None
        self.p_u_ref_all = None

        # Linear gains
        self.k_x_0 = None
        self.k_x_ref_all = None
        self.k_u_minus1 = None
        self.k_u_ref_all = None

    def setup(self, solve=True):
        """ Set-up the QP problem.

        Parameters
        ----------
        solve : bool
               If True, also solve the QP problem.

        """

        n_p = self.n_p
        n_c = self.n_c
        Q_x = self.Q_x
        n_u = self.n_u
        Q_du = self.Q_du

        self.x_0_rh = np.copy(self.x_0)
        self.u_minus1_rh = np.copy(self.u_minus1)

        # MPC solution
        self.u_MPC = None
        self.u_MPC_all = None

        # From term Q_x
        self.A_lag, self.B_lag = lagrange_matrices(self.Ad, self.Bd, self.n_p, self.n_c)
        self.Q_cal_x = np.kron(np.eye(n_p), Q_x)  # x1,...,x_Np

        # From term Q_u
        s_Q_cal_u = np.eye(n_c)
        s_Q_cal_u[-1, -1] = n_p - n_c + 1
        self.Q_cal_u = np.kron(s_Q_cal_u, Q_u)  # u0,...,u_Nc-1

        # From term Q_du
        s_Q_cal_du = 2 * np.eye(n_c) - np.eye(n_c, k=1) - np.eye(n_c, k=-1)
        s_Q_cal_du[n_c - 1, n_c - 1] = 1
        self.Q_cal_du = np.kron(s_Q_cal_du, Q_du)

        # Quadratic term
        self.P = np.transpose(self.B_lag) @ self.Q_cal_x @ self.B_lag + self.Q_cal_u + self.Q_cal_du
        self.P_inv = np.linalg.inv(self.P)

        # Linear terms
        self.p_x_0 = np.transpose(self.B_lag) @ self.Q_cal_x @ self.A_lag
        self.p_x_ref_all = -np.transpose(self.B_lag) @ self.Q_cal_x
        self.p_u_minus1 = np.vstack([-Q_du, np.zeros(((n_c - 1)*n_u, n_u)) ])
        self.p_u_ref_all = - self.Q_cal_u

        # Linear gains
        self.k_x_0 = -self.P_inv @ self.p_x_0
        self.k_x_ref_all = -self.P_inv @ self.p_x_ref_all
        self.k_u_minus1 = -self.P_inv @ self.p_u_minus1
        self.k_u_ref_all = -self.P_inv @ self.p_u_ref_all

        if solve:
            self.solve()

    def output(self):
        """ Return the MPC controller output uMPC, i.e., the first element
         of the optimal input sequence and assign is to self.uminus1_rh. """

        self.u_minus1_rh = self.u_MPC
        return self.u_MPC

    def update(self, x, u=None, x_ref=None, solve=True):
        """ Update the QP problem.

        Parameters
        ----------
        x : array_like. Size: (nx,)
            The new value of x0.

        u : array_like. Size: (nu,)
            The new value of uminus1. If none, it is set to the previously computed u.

        x_ref : array_like. Size: (nx,)
            The new value of xref. If none, it is not changed

        solve : bool
               If True, also solve the QP problem.

        """

        self.x_0_rh = x # previous x0
        if u is not None:
            self.u_minus1_rh = u # otherwise it is just the uMPC updated from the previous step() call

        if x_ref is not None:
            self.x_ref = x_ref
            self.x_ref_all = np.kron(np.ones((n_p, 1)), self.x_ref.reshape(-1, 1))

        self._update_QP_matrices_()
        if solve:
            self.solve()

    def solve(self):

        n_u = self.n_u
        x_0 = self.x_0_rh
        u_minus1 = self.u_minus1_rh

        self.u_MPC_all = (self.k_x_0 @ x_0).ravel() + \
                         (self.k_x_ref_all @ self.x_ref_all).ravel() + \
                         (self.k_u_ref_all @ self.u_ref_all).ravel() + \
                         (self.k_u_minus1 @ u_minus1).ravel()

        self.u_MPC_all = self.u_MPC_all.reshape(-1, n_u)
        self.u_MPC = self.u_MPC_all[0, :]

    def __controller_function__(self, x, u, xref=None):
        """ This function is meant to be used for debug only.
        """

        self.update(x, u, x_ref=xref, solve=True)
        u_MPC = self.output()

        return u_MPC

    def _update_QP_matrices_(self):
        self.p_x_0 = np.transpose(self.B_lag) @ self.Q_cal_x @ self.A_lag
        self.p_x_ref_all = -np.transpose(self.B_lag) @ self.Q_cal_x
        self.p_u_ref_all = - self.Q_cal_u

        self.k_x_0 = -self.P_inv @ self.p_x_0
        self.k_x_ref_all = -self.P_inv @ self.p_x_ref_all
        self.k_u_minus1 = -self.P_inv @ self.p_u_minus1
        self.k_u_ref_all = -self.P_inv @ self.p_u_ref_all


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    # Constants #
    Ts = 0.2 # sampling time (s)
    M = 2    # mass (Kg)
    b = 0.3  # friction coefficient (N*s/m)

    Ad = np.array([
        [1.0, Ts],
        [0,  1.0 - b/M*Ts]
    ])
    Bd = np.array([
      [0.0],
      [Ts/M]])

    # Reference input and states
    p_ref = 4.0
    v_ref = 0.0
    x_ref = np.array([p_ref, v_ref]) # reference state
    u_ref = np.array([0.0])    # reference input
    u_minus1 = np.array([0.0])  # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Objective function
    Q_x = np.diag([1.0, 0.1])    # Quadratic cost for states x0, x1, ..., x_N-1
    Q_u = 1.0 * np.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    Q_du = 0.0 * np.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    x_0 = np.array([0.1, 0.2]) # initial state

    # Prediction horizon
    n_p = 20
    n_c = 20

    K = MPCController(Ad, Bd, n_p=n_p, n_c=n_c, x_0=x_0, x_ref=x_ref, u_minus1=u_minus1,
                      Q_x=Q_x, Q_u=Q_u, Q_du=Q_du)

    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape  # number of states and number or inputs
    len_sim = 200  # simulation length (s)
    n_sim = int(len_sim / Ts)  # simulation length(timesteps)
    x_sim = np.zeros((n_sim, nx))
    u_sim = np.zeros((n_sim, nu))
    t_sim = np.arange(0, n_sim) * Ts

    time_start = time.time()
    x_step = x_0
    for i in range(n_sim):
        u_MPC = K.output()
        x_step = Ad.dot(x_step) + Bd.dot(u_MPC)  # system step
        K.update(x=x_step, u=u_MPC)  # update with measurement
        K.solve()
        x_sim[i, :] = x_step
        u_sim[i, :] = u_MPC

    time_sim = time.time() - time_start

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(t_sim, x_sim[:, 0], "k", label='p')
    axes[0].plot(t_sim, x_ref[0] * np.ones(np.shape(t_sim)), "r--", label="p_ref")
    axes[0].set_title("Position (m)")

    axes[1].plot(t_sim, x_sim[:, 1], label="v")
    axes[1].plot(t_sim, x_ref[1] * np.ones(np.shape(t_sim)), "r--", label="v_ref")
    axes[1].set_title("Velocity (m/s)")

    axes[2].plot(t_sim, u_sim[:, 0], label="u")
    axes[2].plot(t_sim, u_ref * np.ones(np.shape(t_sim)), "r--", label="u_ref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()
