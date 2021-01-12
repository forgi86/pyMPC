from cvxpy import Variable, Parameter, Minimize, Problem, OSQP, quad_form
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import time


if __name__ == "__main__":

    # Discrete time model of a quadcopter
    Ts = 0.2
    M = 2.0

    Ad = sparse.csc_matrix([
        [1.0, Ts],
        [0,  1.0]
    ])
    Bd = sparse.csc_matrix([
      [0.0],
      [Ts/M]])

    [nx, nu] = Bd.shape  # number of states and number or inputs

    # Constraints
    uref = 0
    uinit = 0  # not used here
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

    # Define problem
    u = Variable((nu, Np))
    x = Variable((nx, Np + 1))
    x_init = Parameter(nx)
    objective = 0
    constraints = [x[:,0] == x_init]
    for k in range(Np):
        objective += quad_form(x[:, k] - xref, Q) + quad_form(u[:, k], R)
        constraints += [x[:, k+1] == Ad*x[:, k] + Bd*u[:, k]]
        constraints += [xmin <= x[:, k], x[:, k] <= xmax]
        constraints += [umin <= u[:, k], u[:, k] <= umax]
    objective += quad_form(x[:, Np] - xref, QN)
    prob = Problem(Minimize(objective), constraints)


    # Simulate in closed loop
    # Simulate in closed loop
    len_sim = 15 # simulation length (s)
    nsim = int(len_sim/Ts) # simulation length(timesteps)
    xsim = np.zeros((nsim,nx))
    usim = np.zeros((nsim,nu))
    tsim = np.arange(0,nsim)*Ts

    uminus1_val = uinit # initial previous measured input is the input at time instant -1.
    time_start = time.time()
    for i in range(nsim):
        x_init.value = x0
        #uminus1.value = uminus1_val
        prob.solve(solver=OSQP, warm_start=True)
        uMPC = u[:,0].value
        usim[i,:] = uMPC
        x0 = Ad.dot(x0) + Bd.dot(uMPC)
        xsim[i,:] = x0

        uminus1_val = uMPC # or a measurement if the input is affected by noise
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
