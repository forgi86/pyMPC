import numpy as np
import scipy as sp
import scipy.sparse as sparse
import osqp


class MPCController:
    """ This class implements an MPC controller

    Attributes
    ----------
    x0 : array_like
        Initial system state.
    Ad : matrix_like
        Discrete-time system matrix A.
    Bd : matrix_like
        Discrete-time system matrix B.
    ...
    """
    def __init__(self, Ad, Bd, Np=10,
                 x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None,umax=None,Dumin=None,Dumax=None):
        self.Ad = Ad
        self.Bd = Bd
        self.nx, self.nu = self.Bd.shape # number of states and number or inputs
        self.Np = Np # assert

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

        if uref is not None:
            self.uref = uref # assert...
        else:
            self.uref = np.zeros(self.nu)

        if uminus1 is not None:
            self.uminus1 = uminus1
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

        # constraints handling
        if xmin is not None:
            self.xmin = xmin # assert...
        else:
            self.xmin = -np.ones(self.nx)*np.inf

        if xmax is not None:
            self.xmax = xmax # assert...
        else:
            self.xmax = np.ones(self.nx)*np.inf

        if umin is not None:
            self.umin = umin # assert...
        else:
            self.umin = -np.ones(self.nu)*np.inf

        if umax is not None:
            self.umax = umax # assert...
        else:
            self.umax = np.ones(self.nu)*np.inf

        if Dumin is not None:
            self.Dumin = Dumin # assert...
        else:
            self.Dumin = -np.ones(self.nu)*np.inf

        if Dumax is not None:
            self.Dumax = Dumax # assert...
        else:
            self.Dumax = np.ones(self.nu)*np.inf

        self.JX_ON = True
        self.JU_ON = True
        self.JDU_ON = True

        self.prob = osqp.OSQP()
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None
        self.x0_rh = None
        self.uminus1_rh = None

    def setup(self):
        self.x0_rh = self.x0
        self.uminus1_rh = self.uminus1
        self._compute_QP_matrices_()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, verbose=False, eps_abs=1e-4, eps_rel=1e-4)

    def step(self):
        # Solve
        res = self.prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Extract first control input to the plant
        uMPC = res.x[-self.Np*self.nu:-(self.Np - 1)*self.nu]

        self.uminus1_rh = uMPC
        return uMPC

    def update(self,x,u=None):
        self.x0_rh = x
        if u is not None:
            self.uminus1_rh = u # otherwise it is just the uMPC updated in the step function!
        self._update_QP_matrices_()

    def __controller_function__(self, x, u):
        """ This function is meant to be used for debug only.
        """
        self.x0_rh = x
        self.uminus1_rh = u
        self._update_QP_matrices_()
        # Check solver status

        res = self.prob.solve()

        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Extract first control input to the plant
        uMPC = res.x[-self.Np*self.nu:-(self.Np - 1)*self.nu]

        return uMPC


    def _update_QP_matrices_(self):
        x0_rh = self.x0_rh
        uminus1_rh = self.uminus1_rh
        Np = self.Np
        nx = self.nx
        nu = self.nu
        Dumin = self.Dumin
        Dumax = self.Dumax
        QDu = self.QDu
        uref = self.uref
        #Qeps = self.Qeps
        Qx = self.Qx
        QxN = self.QxN
        Qu = self.Qu
        xref = self.xref

        self.l[:nx] = -x0_rh
        self.u[:nx] = -x0_rh

        self.l[(Np+1)*nx + (Np+1)*nx + (Np)*nu:(Np+1)*nx + (Np+1)*nx + (Np)*nu + nu] = Dumin + uminus1_rh[0:nu]
        self.u[(Np+1)*nx + (Np+1)*nx + (Np)*nu:(Np+1)*nx + (Np+1)*nx + (Np)*nu + nu] = Dumax + uminus1_rh[0:nu]


        # Update the linear term q. This part could be further optimized in case of constant xref...
        q_X = np.zeros((Np + 1) * nx)  # x_N
        if self.JX_ON:
            q_X += np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),       # x0... x_N-1
                           -QxN.dot(xref)])                             # x_N
        else:
            pass

        q_U = np.zeros(Np*nu)
        if self.JU_ON:
            q_U =  np.kron(np.ones(Np), -Qu.dot(uref))
        # Filling P and q for J_DU
        if self.JDU_ON:
            q_U += np.hstack([-QDu.dot(uminus1_rh),           # u0
                              np.zeros((Np - 1) * nu)])     # u1..uN-1
        else:
            pass
        self.q = np.hstack([q_X, q_U])

        self.prob.update(l=self.l, u=self.u, q=self.q)

    def _compute_QP_matrices_(self):
        Np = self.Np
        nx = self.nx
        nu = self.nu
        Qx = self.Qx
        QxN = self.QxN
        Qu = self.Qu
        QDu = self.QDu
        xref = self.xref
        uref = self.uref
        uminus1 = self.uminus1
        Ad = self.Ad
        Bd = self.Bd
        x0 = self.x0
        xmin = self.xmin
        xmax = self.xmax
        umin = self.umin
        umax = self.umax
        Dumin = self.Dumin
        Dumax = self.Dumax

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective

        # Filling P and q for J_X
        P_X = sparse.kron(sparse.eye((Np + 1) * nx), 0)  # x0...xN
        q_X = np.zeros((Np + 1) * nx)  # x_N
        if self.JX_ON:
            P_X += sparse.block_diag([sparse.kron(sparse.eye(Np), Qx),   # x0...x_N-1
                                     QxN])                              # xN
            q_X += np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),       # x0... x_N-1
                           -QxN.dot(xref)])                             # x_N
        else:
            pass

        # Filling P and q for J_U
        P_U = sparse.kron(sparse.eye((Np)*nu),0)
        q_U = np.zeros(Np*nu)
        if self.JU_ON:
            P_U += sparse.kron(sparse.eye(Np), Qu)
            q_U =  np.kron(np.ones(Np), -Qu.dot(uref))

        # Filling P and q for J_DU
        if self.JDU_ON:
            iDu = 2 * np.eye(Np) - np.eye(Np, k=1) - np.eye(Np, k=-1)
            iDu[Np - 1, Np - 1] = 1
            P_U += sparse.kron(iDu, QDu)
            q_U += np.hstack([-QDu.dot(uminus1),           # u0
                              np.zeros((Np - 1) * nu)])     # u1..uN-1
        else:
            pass

        # Linear constraints

        # - linear dynamics x_k+1 = Ax_k + Bu_k
        Ax = sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Np + 1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Np)), sparse.eye(Np)]), Bd)
        Aeq_dyn = sparse.hstack([Ax, Bu])
        leq_dyn = np.hstack([-x0, np.zeros(Np * nx)])
        ueq_dyn = leq_dyn # for equality constraints -> upper bound  = lower bound!

        # - input and state constraints
        Aineq_xu = sparse.eye((Np + 1) * nx + Np * nu)
        lineq_xu = np.hstack([np.kron(np.ones(Np + 1), xmin), np.kron(np.ones(Np), umin)]) # lower bound of inequalities
        uineq_xu = np.hstack([np.kron(np.ones(Np + 1), xmax), np.kron(np.ones(Np), umax)]) # upper bound of inequalities

        Aineq_du = sparse.vstack([sparse.hstack([np.zeros((Np + 1) * nx), np.ones(nu), np.zeros((Np - 1) * nu)]),  # for u0 - u-1
                                  sparse.hstack([np.zeros((Np * nu, (Np+1) * nx)), -sparse.eye(Np * nu) + sparse.eye(Np * nu, k=1)])  # for uk - uk-1, k=1...Np
                                  ]
                                 )

        uineq_du = np.ones((Np+1) * nu)*Dumax
        uineq_du[0:nu] += self.uminus1[0:nu]

        lineq_du = np.ones((Np+1) * nu)*Dumin
        lineq_du[0:nu] += self.uminus1[0:nu] # works for nonscalar u?

        # - OSQP constraints
        A = sparse.vstack([Aeq_dyn, Aineq_xu, Aineq_du]).tocsc()
        l = np.hstack([leq_dyn, lineq_xu, lineq_du])
        u = np.hstack([ueq_dyn, uineq_xu, uineq_du])

        # assign all
        self.P = sparse.block_diag([P_X, P_U])
        self.q = np.hstack([q_X, q_U])
        self.A = A
        self.l = l
        self.u = u

import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    # Constants #
    Ts = 0.2 # sampling time (s)
    M = 2    # mass (Kg)
    b = 0.3  # friction coefficient (N*s/m)

    Ad = sparse.csc_matrix([
        [1.0, Ts],
        [0,  1.0 -b/M*Ts]
    ])
    Bd = sparse.csc_matrix([
      [0.0],
      [Ts/M]])

    # Continous-time matrices (just for reference)
    Ac = np.array([
        [0.0, 1.0],
        [0, -b/M]]
    )
    Bc = np.array([
        [0.0],
        [1/M]
    ])

    # Reference input and states
    pref = 7.0
    vref = 0.0
    xref = np.array([pref, vref]) # reference state
    uref = np.array([0.0])    # reference input
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-10, -10.0])
    xmax = np.array([7.0,   10.0])

    umin = np.array([-1.2])
    umax = np.array([1.2])

    Dumin = np.array([-2e-1])
    Dumax = np.array([2e-1])

    # Objective function
    Qx = sparse.diags([0.5, 0.1])   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = sparse.diags([0.5, 0.1])  # Quadratic cost for xN
    Qu = 2.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 10.0 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    x0 = np.array([0.1, 0.2]) # initial state

    # Prediction horizon
    Np = 20

    K = MPCController(Ad,Bd,Np=20, x0=x0,xref=xref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                      xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax)
    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape # number of states and number or inputs
    len_sim = 20 # simulation length (s)
    nsim = int(len_sim/Ts) # simulation length(timesteps)
    xsim = np.zeros((nsim,nx))
    usim = np.zeros((nsim,nu))
    tsim = np.arange(0,nsim)*Ts

    time_start = time.time()
    xstep = x0
    for i in range(nsim):
        uMPC = K.step()
        xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # system step
        K.update(xstep) # update with measurement
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