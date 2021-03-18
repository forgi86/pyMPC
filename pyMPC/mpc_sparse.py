import numpy as np
import scipy.sparse as sparse
import osqp
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
    """ This class implements a linear constrained MPC controller

    Attributes
    ----------
    Ad : 2D array_like. Size: (nx, nx)
         Discrete-time system matrix Ad.
    Bd : 2D array-like. Size: (nx, nu)
         Discrete-time system matrix Bd.
    n_p : int
        Prediction horizon. Default value: 20.
    n_c : int
        Control horizon. It must be lower or equal to Np. If None, it is set equal to Np.
    x_0 : 1D array_like. Size: (nx,)
         System state at time instant 0. If None, it is set to np.zeros(nx)
    x_ref : 1D array-like. Size: (nx,) or (Np, nx)
           System state reference (aka target, set-point). If size is (Np, nx), reference is time-dependent.
    u_ref : 1D array-like. Size: (nu, )
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
    eps_feas : float
               Scale factor for the matrix Q_eps. Q_eps = eps_feas*eye(nx).
    eps_rel : float
              Relative tolerance of the QP solver. Default value: 1e-3.
    eps_abs : float
              Absolute tolerance of the QP solver. Default value: 1e-3.
    """

    def __init__(self, Ad, Bd, n_p=20, n_c=None,
                 x_0=None, x_ref=None, u_ref=None, u_minus1=None,
                 Q_x=None, Q_u=None, Q_du=None,
                 x_min=None, x_max=None, u_min=None, u_max=None, du_min=None, du_max=None,
                 eps_feas=1e6, eps_rel=1e-3, eps_abs=1e-3):

        if __is_matrix__(Ad) and (Ad.shape[0] == Ad.shape[1]):
            self.Ad = Ad
            self.n_x = Ad.shape[0]  # number of states
        else:
            raise ValueError("Ad should be a square matrix of dimension (nx,nx)!")

        if __is_matrix__(Bd) and Bd.shape[0] == self.n_x:
            self.Bd = Bd
            self.n_u = Bd.shape[1]  # number of inputs
        else:
            raise ValueError("Bd should be a matrix of dimension (nx, nu)!")

        if n_p > 1:
            self.n_p = n_p  # assert
        else:
            raise ValueError("Np should be > 1!")

        if n_c is not None:
            if n_c <= n_p:
                self.n_c = n_c
            else:
                raise ValueError("Nc should be <= Np!")
        else:
            self.n_c = self.n_p

        # x0 handling
        if x_0 is not None:
            if __is_vector__(x_0) and x_0.size == self.n_x:
                self.x_0 = x_0.ravel()
            else:
                raise ValueError("x0 should be an array of dimension (nx,)!")
        else:
            self.x_0 = np.zeros(self.n_x)

        # reference handing
        if x_ref is not None:
            if __is_vector__(x_ref) and x_ref.size == self.n_x:
                self.x_ref = x_ref.ravel()
            elif __is_matrix__(x_ref) and x_ref.shape[1] == self.n_x and x_ref.shape[0] == self.n_p:
                self.x_ref = x_ref
            else:
                raise ValueError("xref should be either a vector of shape (nx,) or a matrix of shape (Np+1, nx)!")
        else:
            self.x_ref = np.zeros(self.n_x)

        if u_ref is not None:
            if __is_vector__(u_ref) and u_ref.size == self.n_u:
                self.u_ref = u_ref.ravel()  # assert...
            else:
                raise ValueError("uref should be a vector of shape (nu,)!")
        else:
            self.u_ref = np.zeros(self.n_u)

        if u_minus1 is not None:
            if __is_vector__(u_minus1) and u_minus1.size == self.n_u:
                self.u_minus1 = u_minus1
            else:
                raise ValueError("uminus1 should be a vector of shape (nu,)!")
        else:
            self.u_minus1 = self.u_ref

        # weights handling
        if Q_x is not None:
            if __is_matrix__(Q_x) and Q_x.shape[0] == self.n_x and Q_x.shape[1] == self.n_x:
                self.Q_x = Q_x
            else:
                raise ValueError("Qx should be a matrix of shape (nx, nx)!")
        else:
            self.Q_x = sparse.zeros((self.n_x, self.n_x))

        if Q_u is not None:
            if __is_matrix__(Q_u) and Q_u.shape[0] == self.n_u and Q_u.shape[1] == self.n_u:
                self.Q_u = Q_u
            else:
                raise ValueError("Qu should be a square matrix of shape (nu, nu)!")
        else:
            self.Q_u = np.zeros((self.n_u, self.n_u))

        if Q_du is not None:
            if __is_matrix__(Q_du) and Q_du.shape[0] == self.n_u and Q_du.shape[1] == self.n_u:
                self.Q_du = Q_du
            else:
                raise ValueError("QDu should be a square matrix of shape (nu, nu)!")
        else:
            self.Q_du = sparse.zeros((self.n_u, self.n_u))

        # constraints handling
        if x_min is not None:
            if __is_vector__(x_min) and x_min.size == self.n_x:
                self.x_min = x_min.ravel()
            else:
                raise ValueError("xmin should be a vector of shape (nx,)!")
        else:
            self.x_min = -np.ones(self.n_x) * np.inf

        if x_max is not None:
            if __is_vector__(x_max) and x_max.size == self.n_x:
                self.x_max = x_max
            else:
                raise ValueError("xmax should be a vector of shape (nx,)!")
        else:
            self.x_max = np.ones(self.n_x) * np.inf

        if u_min is not None:
            if __is_vector__(u_min) and u_min.size == self.n_u:
                self.u_min = u_min
            else:
                raise ValueError("umin should be a vector of shape (nu,)!")
        else:
            self.u_min = -np.ones(self.n_u) * np.inf

        if u_max is not None:
            if __is_vector__(u_max) and u_max.size == self.n_u:
                self.u_max = u_max
            else:
                raise ValueError("umax should be a vector of shape (nu,)!")
        else:
            self.u_max = np.ones(self.n_u) * np.inf

        if du_min is not None:
            if __is_vector__(du_min) and du_min.size == self.n_u:
                self.du_min = du_min
            else:
                raise ValueError("Dumin should be a vector of shape (nu,)!")
        else:
            self.du_min = -np.ones(self.n_u) * np.inf

        if du_max is not None:
            if __is_vector__(du_max) and du_max.size == self.n_u:
                self.du_max = du_max
            else:
                raise ValueError("Dumax should be a vector of shape (nu,)!")
        else:
            self.du_max = np.ones(self.n_u) * np.inf

        self.eps_feas = eps_feas
        self.Q_eps = eps_feas * sparse.eye(self.n_x)

        self.eps_rel = eps_rel
        self.eps_abs = eps_abs
        self.u_failure = self.u_ref  # value provided when the MPC solver fails.

        # Hidden settings (for debug purpose)
        self.raise_error = False  # Raise an error when MPC optimization fails
        self.JX_ON = True  # Cost function terms in X active
        self.JU_ON = True  # Cost function terms in U active
        self.JDU_ON = True  # Cost function terms in Delta U active
        self.SOFT_ON = True  # Soft constraints active
        self.COMPUTE_J_CNST = False  # Compute the constant term of the MPC QP problem

        # QP problem instance
        self.prob = osqp.OSQP()

        # Variables initialized by the setup() method
        self.res = None
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None
        self.x0_rh = None
        self.u_minus1_rh = None
        self.J_CNST = None  # Constant term of the cost function

        # QP problem variable indexes
        var_name = ["x", "u", "slack"]
        var_size_val = np.array([
            self.n_x * self.n_p,  # state variables
            self.n_u * self.n_c,  # command variables
            self.n_x * self.n_p   # slack variables
        ])
        var_idx = np.r_[0, np.cumsum(var_size_val)[:-1]]
        self.var_size = dict(zip(var_name, var_size_val))  # dictionary variable name -> variable size
        self.var_idx = dict(zip(var_name, var_idx))  # dictionary variable name -> variable idx

        cnst_name = ["dyn", "x", "u", "Du"]
        cnst_size_val = np.array([
            self.n_x * self.n_p,  # linear dynamics constraints
            self.n_x * self.n_p,  # interval constraints on x
            self.n_u * self.n_c,  # interval constraints on u
            self.n_u * self.n_c   # interval constraints on Du
        ])
        cnst_idx = np.r_[0, np.cumsum(cnst_size_val)[:-1]]
        self.cnst_size = dict(zip(cnst_name, cnst_size_val))  # dictionary constraint name -> constraint size
        self.cnst_idx = dict(zip(cnst_name, cnst_idx))  # dictionary constraint name -> constraint idx

    def setup(self, solve=True):
        """ Set-up the QP problem.

        Parameters
        ----------
        solve : bool
               If True, also solve the QP problem.

        """
        self.x0_rh = np.copy(self.x_0)
        self.u_minus1_rh = np.copy(self.u_minus1)
        self._compute_QP_matrices_()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, verbose=False, eps_abs=self.eps_rel, eps_rel=self.eps_abs)

        if solve:
            self.solve()

    def output(self, return_x_seq=False, return_u_seq=False, return_eps_seq=False, return_status=False, return_obj_val=False):
        """ Return the MPC controller output u_MPC, i.e., the first element of the optimal input sequence and assign is to self.uminus1_rh.


        Parameters
        ----------
        return_x_seq : bool
                       If True, the method also returns the optimal sequence of states in the info dictionary
        return_u_seq : bool
                       If True, the method also returns the optimal sequence of inputs in the info dictionary
        return_eps_seq : bool
                       If True, the method also returns the optimal sequence of epsilon in the info dictionary
        return_status : bool
                       If True, the method also returns the optimizer status in the info dictionary
        return_obj_val : bool
                       If True, the method also returns the objective function value in the info dictionary

        Returns
        -------
        array_like (n_u,)
            The first element of the optimal input sequence u_MPC to be applied to the system.
        dict
            A dictionary with additional infos. It is returned only if one of the input flags return_* is set to True
        """
        n_x = self.n_x
        n_u = self.n_u

        # Extract first control input to the plant
        if self.res.info.status == 'solved':
            u_MPC = self.res.x[self.var_idx["u"]:self.var_idx["u"] + n_u]
        else:
            u_MPC = self.u_failure

        # Return additional info?
        info = {}
        if return_x_seq:
            seq_X = self.res.x[self.var_idx["x"]:self.var_size["x"]]
            seq_X = seq_X.reshape(-1, n_x)
            info['x_seq'] = seq_X

        if return_u_seq:
            seq_U = self.res.x[self.var_idx["u"]:self.var_idx["u"] + self.var_size["u"]]
            seq_U = seq_U.reshape(-1, n_u)
            info['u_seq'] = seq_U

        if return_eps_seq:
            seq_eps = self.res.x[self.var_idx["slack"]:self.var_size["slack"]]
            seq_eps = seq_eps.reshape(-1, n_x)
            info['eps_seq'] = seq_eps

        if return_status:
            info['status'] = self.res.info.status

        if return_obj_val:
            obj_val = self.res.info.obj_val + self.J_CNST
            info['obj_val'] = obj_val

        self.u_minus1_rh = u_MPC

        if len(info) == 0:
            return u_MPC

        else:
            return u_MPC, info

    def update(self, x, u=None, xref=None, solve=True):
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
            self.u_minus1_rh = u  # otherwise it is just the uMPC updated from the previous step() call
        if xref is not None:
            # TODO: check that new reference is != old reference, do a minimal update of the QP matrices to improve speed
            self.x_ref = xref
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

    def __controller_function__(self, x, u, xref=None):
        """ This function is meant to be used for debug only.
        """

        self.update(x, u, xref=xref, solve=True)
        uMPC = self.output()

        return uMPC

    def _update_QP_matrices_(self):
        x0_rh = self.x0_rh
        u_minus1_rh = self.u_minus1_rh
        n_p = self.n_p
        n_c = self.n_c
        n_x = self.n_x
        n_u = self.n_u
        du_min = self.du_min
        du_max = self.du_max
        Q_du = self.Q_du
        u_ref = self.u_ref
        Q_x = self.Q_x
        Q_u = self.Q_u
        x_ref = self.x_ref
        P_X = self.P_X

        # update linear dynamics constraint with x_1 = A x_0 + B u_0
        x1_tmp = self.Ad@x0_rh
        self.l[self.cnst_idx["dyn"]:self.cnst_idx["dyn"]+n_x] = -x1_tmp
        self.u[self.cnst_idx["dyn"]:self.cnst_idx["dyn"]+n_x] = -x1_tmp

        # update constraint on \Delta u0: du_min <= u0 - u_{-1} <= du_max
        self.l[self.cnst_idx["Du"]:self.cnst_idx["Du"] + n_u] = du_min + u_minus1_rh[0:n_u]
        self.u[self.cnst_idx["Du"]:self.cnst_idx["Du"] + n_u] = du_max + u_minus1_rh[0:n_u]

        # update the linear term q. This part could be further optimized in case of constant x_ref...
        q_X = np.zeros(n_p * n_x)  # x_N
        self.J_CNST = 0.0
        if self.JX_ON:
            if x_ref.ndim == 2 and x_ref.shape[0] == n_p:  # x_ref is a matrix n_p x n_x
                q_X += (-x_ref.reshape(1, -1) @ P_X).ravel()

                if self.COMPUTE_J_CNST:
                    self.J_CNST += -1/2 * q_X @ x_ref.ravel()
            else:
                q_X += np.kron(np.ones(n_p), -Q_x.dot(x_ref))

        else:
            pass

        q_U = np.zeros(n_c*n_u)
        if self.JU_ON:
            self.J_CNST += 1/2 * n_p * (u_ref.dot(Q_u.dot(u_ref)))
            if self.n_c == self.n_p:
                q_U += np.kron(np.ones(n_c), -Q_u.dot(u_ref))
            else:
                iU = np.ones(n_c)
                iU[n_c-1] = (n_p - n_c + 1)
                q_U += np.kron(iU, -Q_u.dot(u_ref))

        # Filling P and q for J_DU
        if self.JDU_ON:
            self.J_CNST += 1/2*u_minus1_rh.dot(Q_du).dot(u_minus1_rh)
            q_U += np.hstack([-Q_du.dot(u_minus1_rh),           # u0
                              np.zeros((n_c - 1) * n_u)])     # u1..uN-1
        else:
            pass

        if self.SOFT_ON:
            q_eps = np.zeros(n_p*n_x)
            self.q = np.hstack([q_X, q_U, q_eps])
        else:
            self.q = np.hstack([q_X, q_U])

        self.prob.update(l=self.l, u=self.u, q=self.q)

    def _compute_QP_matrices_(self):
        n_p = self.n_p
        n_c = self.n_c
        n_x = self.n_x
        n_u = self.n_u
        Q_x = self.Q_x
        Q_u = self.Q_u
        Q_du = self.Q_du
        x_ref = self.x_ref
        u_ref = self.u_ref
        u_minus1 = self.u_minus1
        Ad = self.Ad
        Bd = self.Bd
        x_0 = self.x_0
        x_min = self.x_min
        x_max = self.x_max
        u_min = self.u_min
        u_max = self.u_max
        du_min = self.du_min
        du_max = self.du_max
        Q_eps = self.Q_eps

        # Cast MPC problem to a QP: x = (x(1),x(2),...,x(n_p),u(0),u(1)...,u(n_c-1))
        # - quadratic objective

        P_X = sparse.csc_matrix((n_p*n_x, n_p*n_x))
        q_X = np.zeros(n_p*n_x)  # x_N
        self.J_CNST = 0.0
        if self.JX_ON:
            P_X += sparse.kron(sparse.eye(n_p), Q_x)   # x(1) ... x(n_p)

            if x_ref.ndim == 2 and x_ref.shape[0] >= n_p: # x_ref is a reference trajectory
                q_X += (-x_ref.reshape(1, -1) @ P_X).ravel()
                if self.COMPUTE_J_CNST:
                    self.J_CNST += -1/2 * q_X @ x_ref.ravel()
            else:
                q_X += np.kron(np.ones(n_p), -Q_x.dot(x_ref))    # x(1) ... x(n_p)

        else:
            pass

        # Filling P and q for J_U
        P_U = sparse.csc_matrix((n_c*n_u, n_c*n_u))
        q_U = np.zeros(n_c*n_u)
        if self.JU_ON:
            self.J_CNST += 1/2*n_p*(u_ref.dot(Q_u.dot(u_ref)))

            if self.n_c == self.n_p:
                P_U += sparse.kron(sparse.eye(n_c), Q_u)
                q_U += np.kron(np.ones(n_c), -Q_u.dot(u_ref))
            else: # n_c < n_p. This formula is more general and could handle the case n_c = n_p either. TODO: test
                s_Q_u = np.ones(n_c)
                s_Q_u[n_c-1] = (n_p - n_c + 1)
                P_U += sparse.kron(sparse.diags(s_Q_u), Q_u)
                q_U += np.kron(s_Q_u, -Q_u.dot(u_ref))

        # Filling P and q for J_DU
        if self.JDU_ON:
            self.J_CNST += 1/2*u_minus1.dot((Q_du).dot(u_minus1))
            s_Q_du = 2 * np.eye(n_c) - np.eye(n_c, k=1) - np.eye(n_c, k=-1)
            s_Q_du[n_c - 1, n_c - 1] = 1
            P_U += sparse.kron(s_Q_du, Q_du)
            q_U += np.hstack([-Q_du.dot(u_minus1),            # u(0)
                              np.zeros((n_c - 1) * n_u)])     # u(1)..u(n_c-1)
        else:
            pass

        if self.SOFT_ON:
            P_eps = sparse.kron(np.eye(n_p), Q_eps)
            q_eps = np.zeros(n_p*n_x)

        # Linear constraints

        # - linear dynamics x_k+1 = Ax_k + Bu_k
        A_cal = sparse.kron(sparse.eye(n_p, k=-1), Ad)

        s_B_cal = sparse.eye(n_c)
        if self.n_c < self.n_p:  # expand B matrix if n_c < Nu (see notes)
            s_B_cal = sparse.vstack([s_B_cal,
                                 sparse.hstack([sparse.csc_matrix((n_p - n_c, n_c - 1)), np.ones((n_p - n_c, 1))])
                                ])
        B_cal = sparse.kron(s_B_cal, Bd)

        Aeq_dyn = sparse.hstack([A_cal - sparse.eye(self.n_p * self.n_x), B_cal])

        if self.SOFT_ON:
            n_eps = n_p * n_x
            Aeq_dyn = sparse.hstack([Aeq_dyn, sparse.csc_matrix((Aeq_dyn.shape[0], n_eps))])

        x1_tmp = Ad@x_0
        leq_dyn = np.hstack([-x1_tmp, np.zeros((n_p-1) * n_x)])
        ueq_dyn = leq_dyn  # for equality constraints -> upper bound  = lower bound!

        # - bounds on x
        Aineq_x = sparse.hstack([sparse.eye(n_p * n_x), sparse.csc_matrix((n_p*n_x, n_c*n_u))])
        if self.SOFT_ON:
            Aineq_x = sparse.hstack([Aineq_x, sparse.eye(n_eps)]) # For soft constraints slack variables
        lineq_x = np.kron(np.ones(n_p), x_min)  # lower bound of inequalities
        uineq_x = np.kron(np.ones(n_p), x_max)  # upper bound of inequalities

        Aineq_u = sparse.hstack([sparse.csc_matrix((n_c*n_u, n_p*n_x)), sparse.eye(n_c * n_u)])
        if self.SOFT_ON:
            Aineq_u = sparse.hstack([Aineq_u, sparse.csc_matrix((Aineq_u.shape[0], n_eps))])

        lineq_u = np.kron(np.ones(n_c), u_min)     # lower bound of inequalities
        uineq_u = np.kron(np.ones(n_c), u_max)     # upper bound of inequalities

        # - bounds on \Delta u
        # TODO check for non-scalar u
        Aineq_du = sparse.vstack([sparse.hstack([np.zeros((n_u, n_p * n_x)), sparse.eye(n_u), np.zeros((n_u, (n_c - 1) * n_u))]),  # for u0 - u-1
                                  sparse.hstack([np.zeros((n_c * n_u, n_p * n_x)), -sparse.eye(n_c * n_u) + sparse.eye(n_c * n_u, k=1)])  # for uk - uk-1, k=1...n_p
                                  ]
                                 )
        if self.SOFT_ON:
            Aineq_du = sparse.hstack([Aineq_du, sparse.csc_matrix((Aineq_du.shape[0], n_eps))])

        uineq_du = np.kron(np.ones(n_c+1), du_max)
        uineq_du[0:n_u] += self.u_minus1[0:n_u]

        lineq_du = np.kron(np.ones(n_c+1), du_min)
        lineq_du[0:n_u] += self.u_minus1[0:n_u]

        A = sparse.vstack([Aeq_dyn, Aineq_x, Aineq_u, Aineq_du]).tocsc()
        l = np.hstack([leq_dyn, lineq_x, lineq_u, lineq_du])
        u = np.hstack([ueq_dyn, uineq_x, uineq_u, uineq_du])

        # assign all
        if self.SOFT_ON:
            self.P = sparse.block_diag([P_X, P_U, P_eps], format='csc')
            self.q = np.hstack([q_X, q_U, q_eps])
        else:
            self.P = sparse.block_diag([P_X, P_U], format='csc')
            self.q = np.hstack([q_X, q_U])

        self.A = A
        self.l = l
        self.u = u

        self.P_X = P_X

        # Debug assignments
#        self.P_U = P_U
#        self.P_eps = P_eps
#        self.Aineq_du = Aineq_du
#        self.leq_dyn = leq_dyn
#        self.lineq_du = lineq_du


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    # Constants #
    Ts = 0.2 # sampling time (s)
    M = 2    # mass (Kg)
    b = 0.3  # friction coefficient (N*s/m)

    # Continuous-time system matrices
    Ac = np.array([
        [0.0, 1.0],
        [0, -b/M]]
    )
    Bc = np.array([
        [0.0],
        [1/M]
    ])

    [nx, nu] = Bc.shape # number of states and number or inputs

    # Forward euler discretization
    Ad = np.eye(nx) + Ac*Ts
    Bd = Bc*Ts

    # Reference input and states
    pref = 7.0
    vref = 0.0
    xref = np.array([pref, vref])  # reference state
    uref = np.array([0.0])   # reference input
    uminus1 = np.array([0.0])  # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-10, -10.0])
    xmax = np.array([7.0,   10.0])

    umin = np.array([-1.2])
    umax = np.array([1.2])

    Dumin = np.array([-2e-1])
    Dumax = np.array([2e-1])

    # Objective function
    Qx = sparse.diags([0.5, 0.1])   # Quadratic cost for states x0, x1, ..., x_N-1
    Qu = 2.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 10.0 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    x0 = np.array([0.1, 0.2])  # initial state

    # Prediction horizon
    Np = 25
    Nc = 10

    Xref = np.kron(np.ones((Np, 1)), xref)
    K = MPCController(Ad, Bd, n_p=Np, n_c=Nc, x_0=x0, x_ref=Xref, u_minus1=uminus1,
                      Q_x=Qx, Q_u=Qu, Q_du=QDu,
                      x_min=xmin, x_max=xmax, u_min=umin, u_max=umax, du_min=Dumin, du_max=Dumax)
    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape # number of states and number or inputs
    len_sim = 40 # simulation length (s)
    nsim = int(len_sim/Ts) # simulation length(timesteps)
    xsim = np.zeros((nsim, nx))
    usim = np.zeros((nsim, nu))
    tsim = np.arange(0, nsim)*Ts

    time_start = time.time()
    xstep = x0
    for i in range(nsim):
        uMPC, info = K.output(return_u_seq=True, return_x_seq=True, return_eps_seq=True, return_status=True)
        xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # system step
        K.update(xstep, xref=Xref) # update with measurement
        K.solve()
        xsim[i, :] = xstep
        usim[i, :] = uMPC

    time_sim = time.time() - time_start

    fig, axes = plt.subplots(3,1, figsize=(10,10))
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
