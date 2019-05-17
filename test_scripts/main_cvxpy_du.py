from cvxpy import *
import numpy as np
import scipy as sp
import scipy.sparse as sparse

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
uref = np.array([0.0])      # reference input
uinit =  np.array([0.0])  # input at time step negative one - used to penalize the first delta 0. Could be the same as uref.

# Constraints
xmin = np.array([-100.0, -100.0])
xmax = np.array([100.0,   100.0])

umin = np.array([-1.5])*100
umax = np.array([1.5])*100

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

# Define the optimization problem
u = Variable((nu, Np)) # u0, u1, ... u_N-1
x = Variable((nx, Np + 1)) # x0, x1, ... x_N
x_init = Parameter(nx)
uminus1 = Parameter(nu) # input at time instant negative one (from previous MPC window or uinit in the first MPC window)

objective = 0
constraints = [x[:,0] == x_init]
for k in range(Np):
    objective += quad_form(x[:,k] - xref, Qx) + quad_form(u[:, k] - uref, Qu) # cost function J: \sum_{k=0}^{N_p-1} (xk - x_r)'Qx(xk - x_r) + (u_k)'Qx(u_k)
    if k > 0:
        objective += quad_form(u[:,k] - u[:,k-1], QDu)               # \sum_{k=1}^{N_p-1} (uk - u_k-1)'QDu(uk - u_k-1)
    else:
        objective += quad_form(u[:,k] - uminus1, QDu)                # ... penalize the variation of u0 with respect to uold

    constraints += [x[:,k+1] == Ad*x[:,k] + Bd*u[:,k]]               # model dynamics constraints

    constraints += [xmin <= x[:,k], x[:,k] <= xmax]                  # state constraints
    constraints += [umin <= u[:,k], u[:,k] <= umax]                  # input constraints


    if k > 0:
        constraints += [Dumin <= u[:,k] - u[:,k-1] , u[:,k] - u[:,k-1] <= Dumax]
    else: # at k = 0...
        constraints += [Dumin <= u[:,k] - uminus1 , u[:, k] - uminus1 <= Dumax]

objective += quad_form(x[:, Np] - xref, QxN)                          # add final cost for xN
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
len_sim = 15 # simulation length (s)
nsim = int(len_sim/Ts) # simulation length(timesteps)
xsim = np.zeros((nsim,nx))
usim = np.zeros((nsim,nu))
tsim = np.arange(0,nsim)*Ts

uminus1_val = uinit # initial previous measured input is the input at time instant -1.
for i in range(nsim):
    x_init.value = x0
    uminus1.value = uminus1_val
    #xminus1_val = xminus1
    prob.solve(solver=OSQP, warm_start=True, verbose=False, eps_abs=1e-10, eps_rel=1e-10) # solve MPC problem
    uMPC = u[:,0].value
    usim[i,:] = uMPC
    x0 = Ad.dot(x0) + Bd.dot(uMPC)
    xsim[i,:] = x0

    uminus1_val = uMPC # or a measurement if the input is affected by noise

    if i == 1:
        print(u.value)
    #print(u0)

# In [1]

import matplotlib.pyplot as plt
fig,axes = plt.subplots(3,1, figsize=(10,10))
axes[0].plot(tsim, xsim[:,0], "k", label='p')
axes[0].plot(tsim, pref*np.ones(np.shape(tsim)), "r--", label="pref")
axes[0].set_title("Position (m)")

axes[1].plot(tsim, xsim[:,1], label="v")
axes[1].plot(tsim, vref*np.ones(np.shape(tsim)), "r--", label="vref")
axes[1].set_title("Velocity (m/s)")

axes[2].plot(tsim, usim[:,0], label="u")
axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="uref")
axes[2].set_title("Force (N)")


for ax in axes:
    ax.grid(True)
    ax.legend()
