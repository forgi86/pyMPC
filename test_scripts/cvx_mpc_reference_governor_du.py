import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import time
import control
from cvxpy import Variable, Parameter, Minimize, Problem, OSQP, quad_form


if __name__ == "__main__":

    len_sim = 120  # simulation length (s)

    # Discrete time model of a frictionless mass (pure integrator)
    Ts = 1.0
    r_den = 0.9  # magnitude of poles
    wo_den = 0.2  # phase of poles (approx 2.26 kHz)

    # Build a second-order discrete-time dynamics with dcgain=1 (inner loop model)
    H_sys = control.TransferFunction([1], [1, -2 * r_den * np.cos(wo_den), r_den ** 2], Ts)
    H_sys = H_sys / control.dcgain(H_sys)
    H_ss = control.ss(H_sys)

    Ad = np.array(H_ss.A)
    Bd = np.c_[Bd, 1 / 2 * Bd]
    Cd = np.array(H_ss.C)
    Dd = np.array(H_ss.D)

    Ad = scipy.linalg.block_diag(Ad, Ad)
    Bd = scipy.linalg.block_diag(Bd, Bd)

    [nx, nu] = Bd.shape  # number of states and number or inputs
    [ny, _] = Cd.shape  # number of outputs

    # Constraints
    gref = 0
    ginit = np.array([0.0])  #
    gmin = np.array([-1000.0]) - gref
    gmax = np.array([1000.0]) - gref

    ymin = np.array([-100.0])
    ymax = np.array([100.0])

    Dgmin = np.array([-2e-1])
    Dgmax = np.array([2e-1])


    # Objective function
    Qy = np.diag([20])   # or sparse.diags([])
    QyN = np.diag([20])  # final cost
    #Qg = 0.01 * np.eye(1)
    QDg = 0.5 * sparse.eye(1)  # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial and reference
    x0 = np.array([0.0, 0.0])  # initial state
    r = 1.0  # Reference output

    # Prediction horizon
    Np = 40

    # Define problem
    u = Variable((nu, Np))
    x = Variable((nx, Np + 1))
    x_init = Parameter(nx)
    uminus1 = Parameter(nu)  # input at time instant negative one (from previous MPC window or uinit in the first MPC window)

    objective = 0
    constraints = [x[:, 0] == x_init]
    y = Cd @ x
    for k in range(Np):
        if k > 0:
            objective += quad_form(u[:, k] - u[:, k - 1], QDg)  # \sum_{k=1}^{N_p-1} (uk - u_k-1)'QDu(uk - u_k-1)
        else:  # at k = 0...
#            if uminus1[0] is not np.nan:  # if there is an uold to be considered
            objective += quad_form(u[:, k] - uminus1, QDg)  # ... penalize the variation of u0 with respect to uold

        objective += quad_form(y[:, k] - r, Qy)
        #objective += quad_form(u[:, k], Qg)  # objective function

        constraints += [x[:, k+1] == Ad@x[:, k] + Bd@u[:, k]]  # system dynamics constraint
        constraints += [ymin <= x[:, k], x[:, k] <= ymax]  # state interval constraint
        constraints += [gmin <= u[:, k], u[:, k] <= gmax]  # input interval constraint

        if k > 0:
            constraints += [Dgmin <= u[:, k] - u[:, k - 1], u[:, k] - u[:, k - 1] <= Dgmax]
        else:  # at k = 0...
#            if uminus1[0] is not np.nan:
            constraints += [Dgmin <= u[:, k] - uminus1, u[:, k] - uminus1 <= Dgmax]


    objective += quad_form(y[:, Np] - r, QyN)
    prob = Problem(Minimize(objective), constraints)

    # Simulate in closed loop
    nsim = int(len_sim/Ts)  # simulation length(timesteps)
    xsim = np.zeros((nsim, nx))
    ysim = np.zeros((nsim, ny))
    usim = np.zeros((nsim, nu))
    tsol = np.zeros((nsim, 1))
    tsim = np.arange(0, nsim)*Ts

    uMPC = ginit  # initial previous measured input is the input at time instant -1.
    time_start = time.time()
    for i in range(nsim):

        ysim[i, :] = Cd @ x0
        x_init.value = x0  # set value to the x_init cvx parameter to x0
        uminus1.value = uMPC

        time_start = time.time()
        prob.solve(solver=OSQP, warm_start=True)
        tsol[i] = 1000*(time.time() - time_start)

        uMPC = u[:, 0].value
        usim[i, :] = uMPC
        x0 = Ad.dot(x0) + Bd.dot(uMPC)
        xsim[i, :] = x0

    time_sim = time.time() - time_start

    # In[Plot time traces]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(tsim, ysim[:, 0], "k", label='p')
    axes[0].plot(tsim, r * np.ones(np.shape(tsim)), "r--", label="pref")
    axes[0].set_title("Output (-)")


    axes[2].plot(tsim, usim[:, 0], label="u")
    axes[2].plot(tsim, gref * np.ones(np.shape(tsim)), "r--", label="uref")
    axes[2].set_title("Input (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Timing]
    plt.figure()
    plt.hist(tsol[1:])
    plt.xlabel("MPC solution time (ms)")

    print(f"First MPC execution takes {tsol[0, 0]:.0f} ms")