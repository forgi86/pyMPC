import osqp
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import time

# Discrete time model of a quadcopter
Ts = 0.2
M = 2
Ad = sparse.csc_matrix([
    [1.0, Ts],
    [0,  1.0]
])
Bd = sparse.csc_matrix([
  [0.0],
  [Ts/M]])

[nx, nu] = Bd.shape # number of states and number or inputs

# Constraints
uref = 0
uinit = 0 # not used here
umin = np.array([-1000.0]) - uref
umax = np.array([1000.0]) - uref

xmin = np.array([-100.0, -100.0])
xmax = np.array([100.0,   100.0])

# Objective function
Q = sparse.diags([0.2, 0.3])
QN = sparse.diags([0.4, 0.5]) # final cost
R = 0.1*sparse.eye(1)

# Initial and reference states
x0 = np.array([0.1, 0.2]) # initial state
# Reference input and states
pref = 7.0
vref = 0
xref = np.array([pref, vref]) # reference state

# Prediction horizon
Np = 20

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(Np), Q), QN,
                       sparse.kron(sparse.eye(Np), R)]).tocsc()
# - linear objective
q = np.hstack([np.kron(np.ones(Np), -Q.dot(xref)), -QN.dot(xref),
               np.zeros(Np * nu)])

# - linear dynamics
Ax = sparse.kron(sparse.eye(Np + 1), -sparse.eye(nx)) + sparse.kron(sparse.eye(Np + 1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, Np)), sparse.eye(Np)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(Np * nx)])
ueq = leq # for equality constraints -> upper bound  = lower bound!
# - input and state constraints
Aineq = sparse.eye((Np + 1) * nx + Np * nu)
lineq = np.hstack([np.kron(np.ones(Np + 1), xmin), np.kron(np.ones(Np), umin)]) # lower bound of inequalities
uineq = np.hstack([np.kron(np.ones(Np + 1), xmax), np.kron(np.ones(Np), umax)]) # upper bound of inequalities
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq]).tocsc()
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
len_sim = 15 # simulation length (s)
nsim = int(len_sim/Ts) # simulation length(timesteps)
xsim = np.zeros((nsim,nx))
usim = np.zeros((nsim,nu))
tsim = np.arange(0,nsim)*Ts

#uminus1_val = uinit # initial previous measured input is the input at time instant -1.
time_start = time.time()
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    uMPC = res.x[-Np * nu:-(Np - 1) * nu]
    x0 = Ad.dot(x0) + Bd.dot(uMPC)
    xsim[i,:] = x0
    usim[i,:] = uMPC

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)
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
