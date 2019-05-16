from cvxpy import *
import numpy as np
import scipy as sp
import scipy.sparse as sparse

# Discrete time model of the system (mass point with input force and friction)

# Constants #
Ts = 0.1 # sampling time (s)
M = 2 # mass (Kg)
b = 0.3 # friction coefficient (N*s/m)

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
pref = 10.0
vref = 0
xref = np.array([pref, vref]) # reference state
uref = 0      # reference input
un1 = np.nan  # input at time step negative one - used to penalize the first delta 0. Could be the same as uref.

# Constraints
umin = np.array([-1000.0])
umax = np.array([1000.0])

xmin = np.array([-100.0, -100.0])
xmax = np.array([100.0,   100.0])

# Objective function
Qx = sparse.diags([0.2, 0.3])   # Quadratic cost for states x0, x1, ..., x_N-1
QxN = sparse.diags([0.4, 0.5])  # Quadratic cost for xN
Qu = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
QDu = 2.0 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

# Initial state
x0 = np.array([0.1, 0.2]) # initial state

# Prediction horizon
Np = 10

# Define the optimization problem
u = Variable((nu, Np))
x = Variable((nx, Np + 1))
x_init = Parameter(nx)
objective = 0
constraints = [x[:,0] == x_init]
for k in range(Np):
    objective += quad_form(x[:,k] - xref, Qx) + quad_form(u[:, k] - uref, Qu) # cost function J: \sum_{k=0}^{N_p-1} (xk - x_r)'Qx(xk - x_r) + (u_k)'Qx(u_k)
    if k > 0:
        objective += quad_form(u[:,k] - u[:,k-1], QDu)               # \sum_{k=1}^{N_p-1} (uk - u_k-1)'QDu(uk - u_k-1)
    else: # at k = 0...
        if un1 is not np.nan:  # if there is an un1 to be considered
            objective += quad_form(u[:,k] - un1, QDu) # ... penalize the variation of u0 with respect to un1

    constraints += [x[:,k+1] == Ad*x[:,k] + Bd*u[:,k]]               # model dynamics constraints
    constraints += [xmin <= x[:,k], x[:,k] <= xmax]                  # state constraints
    constraints += [umin <= u[:,k], u[:,k] <= umax]                  # input constraints

objective += quad_form(x[:, Np] - xref, QxN)                          # add final cost for xN
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
nsim = 400
xsim = np.zeros((nsim,nx))
usim = np.zeros((nsim,nu))
tsim = np.arange(0,nsim)*Ts
for i in range(nsim):
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True) # solve MPC problem
    usim[i,:] = u[:,0].value
    x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)
    xsim[i,:] = x0
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
