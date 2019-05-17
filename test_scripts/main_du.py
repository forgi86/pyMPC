import osqp
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import time

# Discrete time model of the system (mass point with input force and friction)

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

[nx, nu] = Bd.shape # number of states and number or inputs

# Reference input and states
pref = 7.0
vref = 0.0
xref = np.array([pref, vref]) # reference state
uref = np.array([0.0])    # reference input
uinit = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

# Constraints
xmin = np.array([-100.0, -100.0])
xmax = np.array([100.0,   100.0])

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

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P_xu = sparse.block_diag([sparse.kron(sparse.eye(Np), Qx),     # x0... x_N-1
                       QxN,                                 # x_N
                       sparse.kron(sparse.eye(Np), Qu)]     # u0... u_N-1
                      ).tocsc()

iDu = 2 * np.eye(Np) - np.eye(Np,k=1) - np.eye(Np, k=-1)
iDu[Np-1,Np-1] = 1
P_du = sparse.block_diag([sparse.kron(sparse.eye((Np+1)*nx), 0), # zeros for x
                          sparse.kron(iDu, QDu)]
                         ).tocsc()

P = P_xu + P_du

# - linear objective
q_x = np.hstack([np.kron(np.ones(Np), -Qx.dot(xref)),       # x0... x_N-1
               -QxN.dot(xref),                              # x_N
               np.zeros(Np * nu)])                          # u0... u_N-1

q_u = np.hstack([np.zeros((Np+1) * nx),
               np.kron(np.ones(Np), -Qu.dot(uref))]
               )

q_du = np.hstack([np.zeros((Np+1) * nx),     # x0... x_N
    -QDu.dot(uinit),                         # u0
    np.zeros((Np-1) * nu)])                  # u1... u_N-1

q = q_x + q_u + q_du

# - linear dynamics
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
uineq_du[0:nu] += uinit[0:nu]

lineq_du = np.ones((Np+1) * nu)*Dumin
lineq_du[0:nu] += uinit[0:nu] # works for nonscalar u?

# - OSQP constraints
A = sparse.vstack([Aeq_dyn, Aineq_xu, Aineq_du]).tocsc()
l = np.hstack([leq_dyn, lineq_xu, lineq_du])
u = np.hstack([ueq_dyn, uineq_xu, uineq_du])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True, verbose=False, eps_abs=1e-10, eps_rel=1e-10)

# Simulate in closed loop
len_sim = 15 # simulation length (s)
nsim = int(len_sim/Ts) # simulation length(timesteps)
xsim = np.zeros((nsim,nx))
usim = np.zeros((nsim,nu))
tsim = np.arange(0,nsim)*Ts


time_start = time.time()
uminus1_val = uinit # initial previous measured input is the input at time instant -1.

for i in range(nsim):

    # Update optimization problem
    l[:nx] = -x0
    u[:nx] = -x0

    l[(Np+1)*nx + (Np+1)*nx + (Np)*nu:(Np+1)*nx + (Np+1)*nx + (Np)*nu + nu] = Dumin + uminus1_val[0:nu]
    u[(Np+1)*nx + (Np+1)*nx + (Np)*nu:(Np+1)*nx + (Np+1)*nx + (Np)*nu + nu] = Dumax + uminus1_val[0:nu]

    q_du = np.hstack([np.zeros((Np+1) * nx),                # x0... x_N
                   -QDu.dot(uminus1_val),                   # u0
                   np.zeros((Np-1) * nu)])                  # u1... u_N-1
    q = q_x + q_u + q_du
    prob.update(l=l, u=u, q=q)

    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    uMPC = res.x[-Np * nu:-(Np - 1) * nu]
    x0 = Ad.dot(x0) + Bd.dot(uMPC)  # system step
    xsim[i,:] = x0
    usim[i,:] = uMPC
    uminus1_val = uMPC

    if i == 1:
        print(res.x[-Np * nu:])
time_sim = time.time() - time_start

# In [1]
import matplotlib.pyplot as plt
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
