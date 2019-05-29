import numpy as np
import scipy as sp
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
    x0 : 1D array_like. Size: (nx,)
         System state at time instant 0.
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
    Np : int
        Prediction horizon. Default value: 10.
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
    eps_feas : Scale factor for the matrix Q_eps. Q_eps = eps_feas*eye(nx)
    """

    def __init__(self, Ad, Bd, Np=10,
                 x0=None, xref=None, uref=None, uminus1=None,
                 Qx=None, QxN=None, Qu=None, QDu=None,
                 xmin=None, xmax=None, umin=None,umax=None,Dumin=None,Dumax=None,
                 eps_feas = 1e6):
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

        self.eps_feas = eps_feas
        self.Qeps = eps_feas * sparse.eye(self.nx)

        self.raise_error = False
        self.u_failure = self.uref # value provided when the MPC solver fails.

        self.JX_ON = True
        self.JU_ON = True
        self.JDU_ON = True
        self.SOFT_ON = True

        self.prob = osqp.OSQP()
        self.res = None
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None
        self.x0_rh = None
        self.uminus1_rh = None

    def setup(self, solve=True):
        """ Set-up the QP problem.

        Parameters
        ----------
        solve : bool
               If True, also solve the QP problem.

        """
        self.x0_rh = self.x0
        self.uminus1_rh = self.uminus1
        self._compute_QP_matrices_()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, verbose=False, eps_abs=1e-3, eps_rel=1e-3)

        if solve:
            self.solve()


    def output(self, return_x_seq=False, return_status=False):
        """ Return the MPC controller output uMPC, i.e., the first element of the optimal input sequence.
            Assign uMPC to self.uminus1_rh.

        Parameters
        ----------
        return_x_seq : bool
                       If true, the method also returns the optimal sequence of states

        Returns
        -------
        array_like (nu,)
            The first element of the optimal input sequence uMPC to be applied to the system.
        """

        Np = self.Np
        nx = self.nx
        nu = self.nu

        # Extract first control input to the plant
        if self.res.info.status == 'solved':
            uMPC = self.res.x[(Np+1)*nx:(Np+1)*nx + nu]
        else:
            uMPC = self.u_failure

        # Return additional info?
        info = {}
        if return_x_seq:
            seq_X = self.res.x[0:(Np+1)*nx]
            seq_X = seq_X.reshape(-1,nx)
            info['x_seq'] = seq_X

        if return_status:
            info['status'] = self.res.info.status

        self.uminus1_rh = uMPC

        if len(info) == 0:
            return uMPC

        else:
            return uMPC, info

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
        self._update_QP_matrices_()
        if solve:
            self.solve()

    def solve(self):
        """ Solve the QP problem. """

        self.res = self.prob.solve()

        # Check solver status
        if self.res.info.status != 'solved':
            warnings.warn('OSQP did not solve the problem!')
            if self.raise_error:
                raise ValueError('OSQP did not solve the problem!')

    def __controller_function__(self, x, u):
        """ This function is meant to be used for debug only.
        """

        self.update(x,u)
        self.solve()
        uMPC = self.output()

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
        Qeps = self.Qeps
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

        if self.SOFT_ON:
            q_eps = np.zeros((Np+1)*nx)
            self.q = np.hstack([q_X, q_U, q_eps])
        else:
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
        Qeps = self.Qeps

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective

        # Filling P and q for J_X
        #P_X = sparse.kron(sparse.eye((Np + 1) * nx), 0)  # x0...xN
        P_X = sparse.csc_matrix(((Np+1)*nx, (Np+1)*nx))
        q_X = np.zeros((Np + 1) * nx)  # x_N
        if self.JX_ON:
            P_X += sparse.block_diag([sparse.kron(sparse.eye(Np), Qx),   # x0...x_N-1
                                     QxN])                              # xN
            q_X += np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),       # x0... x_N-1
                           -QxN.dot(xref)])                             # x_N
        else:
            pass

        # Filling P and q for J_U
        #P_U = sparse.kron(sparse.eye((Np)*nu),0)
        P_U = sparse.csc_matrix((Np*nu, Np*nu))
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

        if self.SOFT_ON:
            P_eps = sparse.kron(np.eye((Np+1)), Qeps)
            q_eps = np.zeros((Np+1)*nx)

        # Linear constraints

        # - linear dynamics x_k+1 = Ax_k + Bu_k
        Ax = sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Np + 1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Np)), sparse.eye(Np)]), Bd)

        n_eps = (Np + 1) * nx
        Aeq_dyn = sparse.hstack([Ax, Bu])
        if self.SOFT_ON:
            Aeq_dyn = sparse.hstack([Aeq_dyn, sparse.csc_matrix((Aeq_dyn.shape[0], n_eps))]) # For soft constraints slack variables

        leq_dyn = np.hstack([-x0, np.zeros(Np * nx)])
        ueq_dyn = leq_dyn # for equality constraints -> upper bound  = lower bound!

        # - bounds on x
        Aineq_x = sparse.hstack([sparse.eye((Np + 1) * nx), sparse.csc_matrix(((Np+1)*nx, Np*nu))])
        if self.SOFT_ON:
            Aineq_x = sparse.hstack([Aineq_x, sparse.eye(n_eps)]) # For soft constraints slack variables
        lineq_x = np.kron(np.ones(Np + 1), xmin) # lower bound of inequalities
        uineq_x = np.kron(np.ones(Np + 1), xmax) # upper bound of inequalities

        Aineq_u = sparse.hstack([sparse.csc_matrix((Np*nu, (Np+1)*nx)), sparse.eye(Np * nu)])
        if self.SOFT_ON:
            Aineq_u = sparse.hstack([Aineq_u, sparse.csc_matrix((Aineq_u.shape[0], n_eps))]) # For soft constraints slack variables
        lineq_u = np.kron(np.ones(Np), umin)     # lower bound of inequalities
        uineq_u = np.kron(np.ones(Np), umax)     # upper bound of inequalities


        # - bounds on \Delta u
        Aineq_du = sparse.vstack([sparse.hstack([np.zeros((Np + 1) * nx), np.ones(nu), np.zeros((Np - 1) * nu)]),  # for u0 - u-1
                                  sparse.hstack([np.zeros((Np * nu, (Np+1) * nx)), -sparse.eye(Np * nu) + sparse.eye(Np * nu, k=1)])  # for uk - uk-1, k=1...Np
                                  ]
                                 )
        if self.SOFT_ON:
            Aineq_du = sparse.hstack([Aineq_du, sparse.csc_matrix((Aineq_du.shape[0], n_eps))])

        uineq_du = np.ones((Np+1) * nu)*Dumax
        uineq_du[0:nu] += self.uminus1[0:nu]

        lineq_du = np.ones((Np+1) * nu)*Dumin
        lineq_du[0:nu] += self.uminus1[0:nu] # works for nonscalar u?

        # Positivity of slack variables (not necessary!)
        #Aineq_eps_pos = sparse.hstack([sparse.coo_matrix((n_eps,(Np+1)*nx)), sparse.coo_matrix((n_eps, Np*nu)), sparse.eye(n_eps)])
        #lineq_eps_pos = np.zeros(n_eps)
        #uineq_eps_pos = np.ones(n_eps)*np.inf

        # - OSQP constraints
        #A = sparse.vstack([Aeq_dyn, Aineq_x, Aineq_u, Aineq_du, Aineq_eps_pos]).tocsc()
        #l = np.hstack([leq_dyn, lineq_x, lineq_u, lineq_du, lineq_eps_pos])
        #u = np.hstack([ueq_dyn, uineq_x, uineq_u, uineq_du, uineq_eps_pos])

        A = sparse.vstack([Aeq_dyn, Aineq_x, Aineq_u, Aineq_du]).tocsc()
        l = np.hstack([leq_dyn, lineq_x, lineq_u, lineq_du])
        u = np.hstack([ueq_dyn, uineq_x, uineq_u, uineq_du])

        # assign all
        if self.SOFT_ON:
            self.P = sparse.block_diag([P_X, P_U, P_eps])
            self.q = np.hstack([q_X, q_U, q_eps])
        else:
            self.P = sparse.block_diag([P_X, P_U])
            self.q = np.hstack([q_X, q_U])

        self.A = A
        self.l = l
        self.u = u

        # Debug assignments

#        self.P_x = P_X
#        self.P_U = P_U
#        self.P_eps = P_eps
        #self.Aineq_du = Aineq_du
        #self.leq_dyn = leq_dyn
        #self.lineq_du = lineq_du

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
        uMPC = K.output()
        xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # system step
        K.update(xstep) # update with measurement
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
