import numpy as np
import matplotlib.pyplot as plt
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
    H_noise = control.TransferFunction([1], [1, -2 * r_den * np.cos(wo_den), r_den ** 2], Ts)
    H_noise = H_noise / control.dcgain(H_noise)
    H_ss = control.ss(H_noise)

    Ad = np.array(H_ss.A)
    Bd = np.array(H_ss.B)
    Cd = np.array(H_ss.C)
    Dd = np.array(H_ss.D)
    [nx, nu] = Bd.shape  # number of states and number or inputs
    [ny, _] = Cd.shape  # number of outputs

    # Constraints
    uref = 0
    uinit = 0  # not used here
    umin = np.array([-1000.0]) - uref
    umax = np.array([1000.0]) - uref

    ymin = np.array([-100.0])
    ymax = np.array([100.0])

    # Objective function
    Qy = np.diag([20])   # or sparse.diags([])
    QyN = np.diag([20])  # final cost
    Qu = 0.1 * np.eye(1)

    # Initial and reference
    x0 = np.array([0.0, 0.0])  # initial state
    r = 1.0  # Reference output

    # Prediction horizon
    Np = 40

    # Define problem
    u = Variable((nu, Np))
    x = Variable((nx, Np + 1))
    x_init = Parameter(nx)
    objective = 0
    constraints = [x[:, 0] == x_init]
    y = Cd @ x
    for k in range(Np):
        objective += quad_form(y[:, k] - r, Qy) \
                     + quad_form(u[:, k], Qu)  # objective function
        constraints += [x[:, k+1] == Ad@x[:, k] + Bd@u[:, k]]  # system dynamics constraint
        constraints += [ymin <= x[:, k], x[:, k] <= ymax]  # state interval constraint
        constraints += [umin <= u[:, k], u[:, k] <= umax]  # input interval constraint
    objective += quad_form(y[:, Np] - r, QyN)
    prob = Problem(Minimize(objective), constraints)

    # Simulate in closed loop
    nsim = int(len_sim/Ts)  # simulation length(timesteps)
    xsim = np.zeros((nsim, nx))
    ysim = np.zeros((nsim, ny))
    usim = np.zeros((nsim, nu))
    tsol = np.zeros((nsim, 1))
    tsim = np.arange(0, nsim)*Ts

#    uminus1_val = uinit  # initial previous measured input is the input at time instant -1.
    time_start = time.time()
    for i in range(nsim):

        ysim[i, :] = Cd @ x0
        x_init.value = x0  # set value to the x_init cvx parameter to x0

        time_start = time.time()
        prob.solve(solver=OSQP, warm_start=True)
        tsol[i] = 1000*(time.time() - time_start)

        uMPC = u[:, 0].value
        usim[i, :] = uMPC
        x0 = Ad.dot(x0) + Bd.dot(uMPC)
        xsim[i, :] = x0

#        uminus1_val = uMPC # or a measurement if the input is affected by noise
    time_sim = time.time() - time_start

    # In[Plot time traces]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(tsim, ysim[:, 0], "k", label='p')
    axes[0].plot(tsim, r * np.ones(np.shape(tsim)), "r--", label="pref")
    axes[0].set_title("Output (-)")


    axes[2].plot(tsim, usim[:, 0], label="u")
    axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="uref")
    axes[2].set_title("Input (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Timing]
    plt.figure()
    plt.hist(tsol[1:])
    plt.xlabel("MPC solution time (ms)")

    print(f"First MPC execution takes {tsol[0, 0]:.0f} ms")