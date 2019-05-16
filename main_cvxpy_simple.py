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
Q = sparse.diags([0.2, 0.3])
QN = sparse.diags([0.4, 0.5]) # final cost
R = 0.1*sparse.eye(1)

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
    objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k], R)
    constraints += [x[:,k+1] == Ad*x[:,k] + Bd*u[:,k]]
    constraints += [xmin <= x[:,k], x[:,k] <= xmax]
    constraints += [umin <= u[:,k], u[:,k] <= umax]
objective += quad_form(x[:, Np] - xr, QN)
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
nsim = 100
for i in range(nsim):
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True)
    x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)
    u0 = u[:,0].value

