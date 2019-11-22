import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse as sparse
import osqp
import warnings

class MPCController:
    """ This class implements an MPC controller

    Attributes
    ----------
    Ad : 2D array_like. Size: (nx, nx)
         Discrete-time system matrix Ad.
    Bd : 2D array-like. Size: (nx, nu)
         Discrete-time system matrix Bd.
    Np : int
        Prediction horizon. Default value: 20.
    Nc : int
        Control horizon. It must be lower or equal to Np. If None, it is set equal to Np.
    x0 : 1D array_like. Size: (nx,)
         System state at time instant 0. If None, it is set to np.zeros(nx)
    xref : 1D array-like. Size: (nx,)
           System state reference (aka target, set-point).
    uref : 1D array-like. Size: (nu, ).
           System input reference. If None, it is set to np.zeros(nx)
    uminus1 : 1D array_like
             Input value assumed at time instant -1. If None, it is set to uref.
    Qx : 2D array_like
         State weight matrix. If None, it is set to eye(nx).
    Qu : 2D array_like
         Input weight matrix. If None, it is set to zeros((nu,nu)).
    QDu : 2D array_like
         Input delta weight matrix. If None, it is set to zeros((nu,nu)).
    xmin : 1D array_like
           State minimum value. If None, it is set to -np.inf*ones(nx).
    xmax : 1D array_like
           State maximum value. If None, it is set to np.inf*ones(nx).
    umin : 1D array_like
           Input minimum value. If None, it is set to -np.inf*ones(nx).
    umax : 1D array_like
           Input maximum value. If None, it is set to np.inf*ones(nx).
    Dumin : 1D array_like
           Input variation minimum value. If None, it is set to np.inf*ones(nx).
    Dumax : 1D array_like
           Input variation maximum value. If None, it is set to np.inf*ones(nx).
    eps_feas : float
               Scale factor for the matrix Q_eps. Q_eps = eps_feas*eye(nx).
    eps_rel : float
              Relative tolerance of the QP solver. Default value: 1e-3.
    eps_abs : float
              Absolute tolerance of the QP solver. Default value: 1e-3.
    """

    def __init__(self, Ad, Bd, Np=20, Nc=None,
                 x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None,umax=None,Dumin=None,Dumax=None,
                 eps_feas = 1e6, eps_rel=1e-3, eps_abs=1e-3):
        self.Ad = Ad
        self.Bd = Bd
        self.nx, self.nu = self.Bd.shape # number of states and number or inputs
        self.Np = Np # assert

        if Nc is not None:
            assert(Nc <= Np)
            self.Nc = Nc
        else:
            self.Nc = self.Np

        # x0 handling
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.nx)
        # reference handing
        if xref is not None:
            self.xref = xref # assert...
        else:
            self.xref = np.zeros(self.nx)

        self.Xref = np.kron(np.ones((Np, 1)), self.xref.reshape(-1, 1))

        if uref is not None:
            self.uref = uref # assert...
        else:
            self.uref = np.zeros(self.nu)
        self.Uref = np.kron(np.ones((Np, 1)), self.uref.reshape(-1, 1))

        if uminus1 is not None:
            self.uminus1 = uminus1.reshape(-1,1)

        else:
            self.uminus1 = self.uref

        # weights handling
        if Qx is not None:
            self.Qx = Qx
        else:
            self.Qx = np.zeros((self.nx, self.nx)) # sparse

        if QxN is not None:
            self.QxN = QxN
        else:
            self.QxN = self.Qx # sparse

        if Qu is not None:
            self.Qu = Qu
        else:
            self.Qu = np.zeros((self.nu, self.nu))

        if QDu is not None:
            self.QDu = QDu
        else:
            self.QDu = np.zeros((self.nu, self.nu))

        self.u_failure = self.uref # value provided when the MPC solver fails.

    def setup(self, solve=True):
        """ Set-up the QP problem.

        Parameters
        ----------
        solve : bool
               If True, also solve the QP problem.

        """

        Np = self.Np
        Qx = self.Qx
        QxN = self.QxN
        nx = self.nx
        nu = self.nu
        Ad = self.Ad
        Bd = self.Bd
        QDu = self.QDu

        self.x0_rh = self.x0
        self.uminus1_rh = self.uminus1

        self.A_cal = np.empty((Np * nx, nx))
        self.B_cal = np.empty((Np * nx, Np * nu))

        # build A_cal
        for k in range(0, Np):
            if k == 0:
                A_km1 = np.eye(nx)
            else:
                A_km1 = self.A_cal[(k - 1) * nx:(k) * nx, 0:nx]
            self.A_cal[k * nx:(k + 1) * nx, 0:nx] = Ad @ A_km1

        # build B_cal
        for k in range(Np):
            if k == 0:
                A_k = np.eye(nx)
            else:
                A_k = self.A_cal[(k - 1) * nx:(k) * nx, 0:nx]
            A_kB = A_k @ Bd
            for p in range(Np - k):
                self.B_cal[(k + p) * nx:(k + p + 1) * nx, p * nu:(p + 1) * nu] = A_kB

        self.Q_cal_x = scipy.linalg.block_diag(np.kron(np.eye(Np-1), Qx),   # x0...x_N-1
                                          QxN)                              # xN
        self.Q_cal_u = scipy.linalg.block_diag(np.kron(np.eye(Np), Qu))   # x0...x_N-1

        iDu = 2 * np.eye(Np) - np.eye(Np, k=1) - np.eye(Np, k=-1)
        iDu[Np - 1, Np - 1] = 1
        self.Q_cal_Du = np.kron(iDu, QDu)

        self.P = np.transpose(self.B_cal) @ self.Q_cal_x @ self.B_cal + self.Q_cal_u + self.Q_cal_Du
        self.P_inv = np.linalg.inv(self.P)
        self.p_uminus1 = np.vstack([-QDu,  # u0
                                    np.zeros(((Np - 1)*nu, nu))  # u1..uN-1
                                    ])

        self.p_x0 = np.transpose(self.B_cal) @ self.Q_cal_x @ self.A_cal
        self.p_Xref = -np.transpose(self.B_cal) @ self.Q_cal_x
        self.p_Uref = - self.Q_cal_u

        self.k_x0 = -self.P_inv @ self.p_x0
        self.k_Xref = -self.P_inv @ self.p_Xref
        self.k_uminus1 = -self.P_inv @ self.p_uminus1
        self.k_Uref = -self.P_inv @ self.p_Uref

        if solve:
            self.solve()

    def output(self, return_x_seq=False, return_u_seq=False, return_eps_seq=False, return_status=False, return_obj_val=False):
        """ Return the MPC controller output uMPC, i.e., the first element of the optimal input sequence and assign is to self.uminus1_rh. """
        self.uminus1_rh = self.u_MPC
        return self.u_MPC

    def update(self,x,u=None,xref=None,solve=True):
        """ Update the QP problem.

        Parameters
        ----------
        x : array_like. Size: (nx,)
            The new value of x0.

        u : array_like. Size: (nu,)
            The new value of uminus1. If none, it is set to the previously computed u.

        xref : array_like. Size: (nx,)
            The new value of xref. If none, it is not changed

        solve : bool
               If True, also solve the QP problem.

        """

        self.x0_rh = x # previous x0
        if u is not None:
            self.uminus1_rh = u # otherwise it is just the uMPC updated from the previous step() call
        if xref is not None:
            self.xref = xref # TODO: check that new reference is != old reference, do a minimal update of the QP matrices to improve speed
            self.Xref = np.kron(np.ones((Np, 1)), self.xref.reshape(-1, 1))

        self._update_QP_matrices_()
        if solve:
            self.solve()

    def solve(self):

        nu = self.nu
        x0 = self.x0_rh
        uminus1 = self.uminus1_rh

        self.u_MPC_all = (self.k_x0 @ x0).ravel() +\
                         (self.k_Xref @ self.Xref).ravel() + \
                         (self.k_Uref @ self.Uref).ravel() +\
                         (self.k_uminus1 @ uminus1).ravel()

        self.u_MPC_all = self.u_MPC_all.reshape(-1, nu)
        self.u_MPC = self.u_MPC_all[0, :]

    def __controller_function__(self, x, u, xref=None):
        """ This function is meant to be used for debug only.
        """

        self.update(x,u,xref=xref,solve=True)
        uMPC = self.output()

        return uMPC


    def _update_QP_matrices_(self):
        self.p_x0 = np.transpose(self.B_cal) @ self.Q_cal_x @ self.A_cal
        self.p_Xref = -np.transpose(self.B_cal) @ self.Q_cal_x
        self.p_Uref = - self.Q_cal_u

        self.k_x0 = -self.P_inv @ self.p_x0
        self.k_Xref = -self.P_inv @ self.p_Xref
        self.k_uminus1 = -self.P_inv @ self.p_uminus1
        self.k_Uref = -self.P_inv @ self.p_Uref

    def _compute_QP_matrices_(self):
        pass


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
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

    # Reference input and states
    pref = 7.0
    vref = 0.0
    xref = np.array([pref, vref]) # reference state
    uref = np.array([0.0])    # reference input
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Objective function
    Qx = np.diag([0.5, 0.1])   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = np.diag([0.5, 0.1])  # Quadratic cost for xN
    Qu = 2 * np.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 10 * np.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    x0 = np.array([0.1, 0.2]) # initial state

    # Prediction horizon
    #Np = 25
    Np = 30
    Nc = None

    K = MPCController(Ad,Bd,Np=Np,Nc=Nc,x0=x0,xref=xref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu)

    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape # number of states and number or inputs
    len_sim = 40 # simulation length (s)
    nsim = int(len_sim/Ts) # simulation length(timesteps)
    xsim = np.zeros((nsim,nx))
    usim = np.zeros((nsim,nu))
    tsim = np.arange(0,nsim)*Ts

    time_start = time.time()
    xstep = x0
    for i in range(nsim):
        uMPC = K.output()
        xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # system step
        K.update(x=xstep,u=uMPC) # update with measurement
        K.solve()
        xsim[i,:] = xstep
        usim[i,:] = uMPC

    time_sim = time.time() - time_start

    #K.__controller_function__(np.array([0,0]), np.array([0]))

    fig,axes = plt.subplots(3,1, figsize=(10,10))
    axes[0].plot(tsim, xsim[:,0], "k", label='p')
    axes[0].plot(tsim, xref[0]*np.ones(np.shape(tsim)), "r--", label="pref")
    axes[0].set_title("Position (m)")

    axes[1].plot(tsim, xsim[:,1], label="v")
    axes[1].plot(tsim, xref[1]*np.ones(np.shape(tsim)), "r--", label="vref")
    axes[1].set_title("Velocity (m/s)")

    axes[2].plot(tsim, usim[:,0], label="u")
    axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="uref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()
