from cvxpy import *
import numpy as np
import scipy as sp
import scipy.sparse as sparse

# Discrete time model of a quadcopter
Ts = 0.1
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
umin = np.array([-1000.0]) - uref
umax = np.array([1000.0]) - uref

xmin = np.array([-100.0, -100.0])
xmax = np.array([100.0,   100.0])

# Objective function
Qx = sparse.diags([0.2, 0.3])  # Quadratic weight for states x0, x1, ..., x_N-1
QxN = sparse.diags([0.4, 0.5]) # Quadratic cost for xN
Qu = 0.1 * sparse.eye(1)         # Quadratic cost for u0, u1, ...., u_N-1
QDu = 1 * sparse.eye(1)         # Quadratic cost for Du0, Du1, ...., Du_N-1

# Initial and reference states
x0 = np.array([0.1, 0.2]) # initial state
xr = np.array([50.0, 50.0]) # reference state

# Prediction horizon
Np = 10


# Define problem
u = Variable((nu, Np))
x = Variable((nx, Np + 1))
x_init = Parameter(nx)
objective = 0
constraints = [x[:,0] == x_init]
for k in range(Np):
    objective += quad_form(x[:,k] - xr, Qx) + quad_form(u[:, k], Qu) # cost function J: \sum_{k=0}^{N_p-1} (xk - x_r)'Qx(xk - x_r) + (u_k)'Qx(u_k)
    if k > 0:
        objective += quad_form(u[:,k] - u[:,k-1], QDu)               # \sum_{k=1}^{N_p-1} (uk - u_k-1)'QDu(uk - u_k-1)
    else:
        if uref is not np.nan:  # if there is an uref to be considered
            objective += quad_form(u[:,k] - uref, QDu)

    constraints += [x[:,k+1] == Ad*x[:,k] + Bd*u[:,k]]               # model dynamics constraints
    constraints += [xmin <= x[:,k], x[:,k] <= xmax]                  # state constraints
    constraints += [umin <= u[:,k], u[:,k] <= umax]                  # input constraints

objective += quad_form(x[:, Np] - xr, QxN)                           # add final cost for xN
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
nsim = 10
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

#In[1]
import matplotlib.pyplot as plt
fig,ax = plt.subplots(3,1)
ax[0].plot(tsim, xsim[:,0])
ax[1].plot(tsim, xsim[:,1])
ax[2].plot(tsim, usim[:,0])
