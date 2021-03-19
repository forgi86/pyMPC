import numpy as np
import scipy.linalg
from qpoases import PySQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from pyMPC.util import lagrange_matrices
import warnings


def __is_vector__(vec):
    if vec.ndim == 1:
        return True
    else:
        if vec.ndim == 2:
            if vec.shape[0] == 1 or vec.shape[1] == 0:
                return True
        else:
            return False
        return False


def __is_matrix__(mat):
    if mat.ndim == 2:
        return True
    else:
        return False


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
         Input delta weight matrix. If None, it is set to zeros((nu,nu)).
    x_min : 1D array_like
           State minimum value. If None, it is set to -np.inf*ones(nx).
    x_max : 1D array_like
           State maximum value. If None, it is set to np.inf*ones(nx).
    u_min : 1D array_like
           Input minimum value. If None, it is set to -np.inf*ones(nx).
    u_max : 1D array_like
           Input maximum value. If None, it is set to np.inf*ones(nx).
    du_min : 1D array_like
           Input variation minimum value. If None, it is set to np.inf*ones(nx).
    du_max : 1D array_like
           Input variation maximum value. If None, it is set to np.inf*ones(nx).
    """

    def __init__(self, Ad, Bd, n_p=20, n_c=None, x_0=None, x_ref=None, u_ref=None, u_minus1=None,
                 Q_x=None, Q_u=None, Q_du=None,
                 x_min=None, x_max=None, u_min=None, u_max=None, du_min=None, du_max=None):

        # options
        self.SOFT_ON = True  # soft constraints with slack variables ON

        self.Ad = Ad
        self.Bd = Bd
        self.n_x, self.n_u = self.Bd.shape # number of states and number or inputs
        self.n_p = n_p  # assert

        if n_c is not None:
            if n_c <= n_p:
                self.n_c = n_c
            else:
                raise ValueError("n_c should be <= n_p!")
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
        self.x_ref_all = np.kron(np.ones(self.n_p), self.x_ref)

        if u_ref is not None:
            self.u_ref = u_ref  # assert...
        else:
            self.u_ref = np.zeros(self.n_u)
        self.u_ref_all = np.kron(np.ones(self.n_c), self.u_ref)

        if u_minus1 is not None:
            self.u_minus1 = u_minus1

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

        # constraints handling
        if x_min is not None:
            if __is_vector__(x_min) and x_min.size == self.n_x:
                self.x_min = x_min.ravel()
            else:
                raise ValueError("xmin should be a vector of shape (nx,)!")
        else:
            self.x_min = -np.ones(self.n_x) * np.inf
        self.x_min_all = np.kron(np.ones(self.n_p), self.x_min)

        if x_max is not None:
            if __is_vector__(x_max) and x_max.size == self.n_x:
                self.x_max = x_max
            else:
                raise ValueError("xmax should be a vector of shape (nx,)!")
        else:
            self.x_max = np.ones(self.n_x) * np.inf
        self.x_max_all = np.kron(np.ones(self.n_p), self.x_max)

        if u_min is not None:
            if __is_vector__(u_min) and u_min.size == self.n_u:
                self.u_min = u_min
            else:
                raise ValueError("umin should be a vector of shape (nu,)!")
        else:
            self.u_min = -np.ones(self.n_u) * np.inf
        self.u_min_all = np.kron(np.ones(self.n_c), self.u_min)

        if u_max is not None:
            if __is_vector__(u_max) and u_max.size == self.n_u:
                self.u_max = u_max
            else:
                raise ValueError("umax should be a vector of shape (nu,)!")
        else:
            self.u_max = np.ones(self.n_u) * np.inf
        self.u_max_all = np.kron(np.ones(self.n_c), self.u_max)

        self.Q_eps = 1e4

        self.u_failure = self.u_ref  # value provided when the MPC solver fails.

        self.x_0_rh = None
        self.u_minus1_rh = None

        self.problem = None
        self.res = None
        self.n_vars = None
        self.n_cnst = None
        self.n_slack = None

        self.A_lag = None
        self.B_lag = None

        self.Q_cal_x = None
        self.Q_cal_u = None
        self.Q_cal_du = None

        self.P_QP = None
        self.p_QP = None
        self.P_x = None
        self.p_x = None
        self.P_u = None
        self.p_u = None
        self.P_u = None
        self.P_du = None
        self.p_du = None

        self.var_min_all = None
        self.var_max_all = None
        self.A_cnst = None
        self.lbA = None
        self.ubA = None

        # cnst_name = ["x", "du"]
        # cnst_size_val = np.array([
        #     self.n_x * self.n_p,  # linear dynamics constraints
        #     self.n_u * self.n_c   # interval constraints on Du
        # ])
        #
        # cnst_idx = np.r_[0, np.cumsum(cnst_size_val)[:-1]]
        # self.cnst_size = dict(zip(cnst_name, cnst_size_val))  # dictionary constraint name -> constraint size
        # self.cnst_idx = dict(zip(cnst_name, cnst_idx))  # dictionary constraint name -> constraint idx

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

        if self.SOFT_ON:
            self.n_slack = 1  # a single slack
        else:
            self.n_slack = 0

        # From term Q_x
        self.A_lag, self.B_lag = lagrange_matrices(self.Ad, self.Bd, self.n_p, self.n_c)
        self.Q_cal_x = np.kron(np.eye(n_p), Q_x)  # x1,...,x_Np
        self.P_x = np.transpose(self.B_lag) @ self.Q_cal_x @ self.B_lag
        self.p_x = np.transpose(self.B_lag) @ self.Q_cal_x @ (self.A_lag @ self.x_0_rh - self.x_ref_all)

        # From term Q_u
        s_Q_cal_u = np.eye(n_c)
        s_Q_cal_u[-1, -1] = n_p - n_c + 1
        self.Q_cal_u = np.kron(s_Q_cal_u, Q_u)  # u0,...,u_Nc-1
        self.P_u = self.Q_cal_u
        self.p_u = -self.Q_cal_u @ self.u_ref_all

        # From term Q_du
        s_Q_cal_du = 2 * np.eye(n_c) - np.eye(n_c, k=1) - np.eye(n_c, k=-1)
        s_Q_cal_du[n_c - 1, n_c - 1] = 1
        self.Q_cal_du = np.kron(s_Q_cal_du, Q_du)
        self.P_du = self.Q_cal_du
        self.p_du = np.r_[(-1 * self.Q_du.dot(self.u_minus1_rh)), np.zeros((self.n_c - 1) * self.n_u)]

        self.P_QP = self.P_x + self.P_u + self.P_du
        self.p_QP = self.p_x + self.p_u + self.p_du

        if self.SOFT_ON:
            self.P_QP = scipy.linalg.block_diag(self.P_QP, self.Q_eps)
            self.p_QP = np.r_[self.p_QP, self.Q_eps]

        # Constraints
        if self.SOFT_ON:
            self.var_min_all = np.r_[self.u_min_all, np.zeros(self.n_slack)]
            self.var_max_all = np.r_[self.u_max_all, np.inf * np.ones(self.n_slack)]
        else:
            self.var_min_all = self.u_min_all
            self.var_max_all = self.u_max_all


        if self.SOFT_ON:
            A_cnst_1 = np.c_[self.B_lag, np.ones((self.n_p * self.n_x, 1))]
            A_cnst_2 = np.c_[-self.B_lag, np.ones((self.n_p * self.n_x, 1))]
        else:
            A_cnst_1 = self.B_lag
            A_cnst_2 = -self.B_lag

        lbA_1 = self.x_min_all - self.A_lag.dot(self.x_0_rh)   # lower bound x constraint
        lbA_2 = -self.x_max_all + self.A_lag.dot(self.x_0_rh)  # upper bound x constraint

        self.A_cnst = np.r_[A_cnst_1, A_cnst_2]
        self.lbA = np.r_[lbA_1, lbA_2]
        self.ubA = np.inf*np.ones_like(self.lbA)

        self.n_vars = self.n_u * self.n_c + self.n_slack
        self.n_cnst = 2 * self.n_x * self.n_p

        self.problem = SQProblem(self.n_vars, self.n_cnst)  # n_vars, n_cnst

        options = Options()
        options.setToMPC()
        options.printLevel = PrintLevel.NONE
        self.problem.setOptions(options)

        # solve_first
        nWSR = np.array([1000])
        self.problem.init(self.P_QP, self.p_QP, self.A_cnst, self.var_min_all, self.var_max_all, self.lbA, self.ubA, nWSR)

        if solve:
            self.solve()

    def update(self, x_zero, u_minus1=None, x_ref=None, solve=True):
        """ Update the QP problem.

        Parameters
        ----------
        x_zero : array_like. Size: (nx,)
            The new value of x0.

        u_minus1 : array_like. Size: (nu,)
            The new value of uminus1. If none, it is set to the previously computed u.

        x_ref : array_like. Size: (nx,)
            The new value of xref. If none, it is not changed

        solve : bool
               If True, also solve the QP problem.

        """

        self.x_0_rh = np.copy(x_zero)  # previous x0

        if u_minus1 is not None:
            self.u_minus1_rh = u_minus1  # otherwise it is just the uMPC updated from the previous step() call

        if x_ref is not None:
            self.x_ref = x_ref
            self.x_ref_all = np.kron(np.ones(n_p), self.x_ref)

        self.p_x = np.transpose(self.B_lag) @ self.Q_cal_x @ (self.A_lag @ self.x_0_rh - self.x_ref_all)
        self.p_du = np.r_[(-1 * self.Q_du.dot(self.u_minus1_rh)), np.zeros((self.n_c - 1) * self.n_u)]

        self.P_QP[:self.n_c*self.n_u, :self.n_c*self.n_u] = self.P_x + self.P_u + self.P_du
        self.p_QP[:self.n_c*self.n_u] = self.p_x + self.p_u + self.p_du

        # constraints
        lbA_1 = self.x_min_all - self.A_lag.dot(self.x_0_rh)   # lower bound constraint
        lbA_2 = -self.x_max_all + self.A_lag.dot(self.x_0_rh)  # upper bound constraint
        self.lbA = np.r_[lbA_1, lbA_2]

        if solve:
            self.solve()

    def solve(self):
        nWSR = np.array([1000])
        self.problem.hotstart(self.P_QP, self.p_QP, self.A_cnst, self.var_min_all, self.var_max_all, self.lbA, self.ubA,
                              nWSR)

        res = np.zeros(self.n_vars)  # to store the solution
        self.problem.getPrimalSolution(res)
        self.res = np.copy(res)
        #self.res = np.linalg.solve(self.P_QP, -self.p_QP)

    def output(self, update_u_minus1=True):
        u_MPC = self.res[:self.n_u]

        if update_u_minus1:
            self.u_minus1 = u_MPC
        return np.copy(u_MPC)

    def full_output(self):
        u_seq = np.copy(self.res[: (self.n_c * self.n_u)])
        x_seq = self.A_lag @ self.x_0_rh + self.B_lag @ u_seq
        u_seq = u_seq.reshape(self.n_c, self.n_u)
        x_seq = x_seq.reshape(self.n_p, self.n_x)
        if self.SOFT_ON:
            gamma = self.res[-1]
        else:
            gamma = np.nan
        return u_seq, x_seq, gamma


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
    u_minus1 = np.array([0.0])

    # Objective function
    Q_x = np.diag([1.0, 0.1])    # Quadratic cost for states x0, x1, ..., x_N-1
    Q_u = 1.0 * np.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    Q_du = 0.0 * np.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    x_0 = np.array([0.0, 1.0])  # initial state

    # Prediction horizon
    n_p = 20
    n_c = 10

    # Constraints
    x_min = np.array([-10, -0.6])
    x_max = np.array([7.0, 0.6])

    u_min = np.array([-1.2])
    u_max = np.array([1.2])

    du_min = np.array([-2e-1])
    du_max = np.array([2e-1])

    K = MPCController(Ad, Bd, n_p=n_p, n_c=n_c, x_0=x_0, x_ref=x_ref, u_minus1=u_minus1,
                      Q_x=Q_x, Q_u=Q_u, Q_du=Q_du,
                      x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max, du_min=du_min, du_max=du_max)
    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape  # number of states and number or inputs
    len_sim = 40  # simulation length (s)
    n_sim = int(len_sim / Ts)  # simulation length(timesteps)
    x_sim = np.zeros((n_sim, nx))
    u_sim = np.zeros((n_sim, nu))
    t_sim = np.arange(0, n_sim) * Ts

    time_start = time.time()
    x_step = x_0
    for i in range(n_sim):
        u_MPC = K.output()
        x_step = Ad.dot(x_step) + Bd.dot(u_MPC)  # system step
        K.update(x_step, x_ref=x_ref) # update with measurement
        K.solve()
        x_sim[i, :] = x_step
        u_sim[i, :] = u_MPC

    time_sim = time.time() - time_start

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(t_sim, x_sim[:, 0], "k", label='p')
    axes[0].plot(t_sim, x_ref[0] * np.ones(np.shape(t_sim)), "r--", label="pref")
    axes[0].set_title("Position (m)")

    axes[1].plot(t_sim, x_sim[:, 1], label="v")
    axes[1].plot(t_sim, x_ref[1] * np.ones(np.shape(t_sim)), "r--", label="vref")
    axes[1].set_title("Velocity (m/s)")

    axes[2].plot(t_sim, u_sim[:, 0], label="u")
    axes[2].plot(t_sim, u_ref * np.ones(np.shape(t_sim)), "r--", label="uref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()
