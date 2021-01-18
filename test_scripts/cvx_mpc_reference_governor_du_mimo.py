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

    # SISO ABCD
    Ad = np.array(H_ss.A)
    Bd = np.array(H_ss.B)
    Cd = np.array(H_ss.C)
    Dd = np.array(H_ss.D)

    # MIMO ABCD
    Ad = scipy.linalg.block_diag(Ad, Ad)
    Bd = scipy.linalg.block_diag(Bd, Bd)
    Cd = scipy.linalg.block_diag(Cd, 1.5*Cd)
    Dd = scipy.linalg.block_diag(Dd, Dd)

    [nx, ng] = Bd.shape  # number of states and number or inputs
    [ny, _] = Cd.shape  # number of outputs

    # Constraints
    ginit = np.array(2*[0.0])  #
    gmin = np.array(2*[-1000.0]) #- gref
    gmax = np.array(2*[1000.0]) #- gref

    ymin = np.array(2*[-100.0])
    ymax = np.array(2*[100.0])

    Dgmin = np.array(2*[-2e-1])
    Dgmax = np.array(2*[2e-1])


    # Objective function
    Qy = np.diag(2*[20])   # or sparse.diags([])
    #QyN = np.diag(2*[20])  # final cost
    QDy = np.eye(ny)
    Qrg = 100*np.eye(ny)
    QDg = 0.5 * sparse.eye(ny)  # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial and reference
    x0 = np.array(2*[0.0, 0.0])  # initial state

    # Prediction horizon
    Np = 40

    # Define problem
    g = Variable((ng, Np))
    x = Variable((nx, Np))
    x_init = Parameter(nx)
    gminus1 = Parameter(ny)  # input at time instant negative one (from previous MPC window or uinit in the first MPC window)
    yminus1 = Parameter(ny)  # input at time instant negative one (from previous MPC window or uinit in the first MPC window)
    r = Parameter(ny)

    objective = 0.0
    constraints = [x[:, 0] == x_init]
    y = Cd @ x + Dd @g

    for k in range(Np):
        objective += quad_form(y[:, k] - r, Qy)   # tracking cost
        objective += quad_form(g[:, k] - r, Qrg)  # reference governor cost
        if k > 0:
            objective += quad_form(g[:, k] - g[:, k - 1], QDg)
            objective += quad_form(y[:, k] - y[:, k - 1], QDy)
        else:  # at k = 0...
            objective += quad_form(g[:, k] - gminus1, QDg)  # ... penalize the variation of u0 with respect to uold
            objective += quad_form(y[:, k] - yminus1, QDy)  # ... penalize the variation of u0 with respect to uold

        #objective += quad_form(u[:, k], Qg)  # objective function

        if k < Np - 1:
            constraints += [x[:, k+1] == Ad @x[:, k] + Bd @ g[:, k]]  # system dynamics constraint
        constraints += [ymin <= y[:, k], y[:, k] <= ymax]  # state interval constraint
        constraints += [gmin <= g[:, k], g[:, k] <= gmax]  # input interval constraint

        if k > 0:
            constraints += [Dgmin <= g[:, k] - g[:, k - 1], g[:, k] - g[:, k - 1] <= Dgmax]
        else:  # at k = 0...
#            if uminus1[0] is not np.nan:
            constraints += [Dgmin <= g[:, k] - gminus1, g[:, k] - gminus1 <= Dgmax]

    #objective += quad_form(y[:, Np] - r, QyN)

    prob = Problem(Minimize(objective), constraints)

    # Simulate in closed loop
    nsim = int(len_sim/Ts)  # simulation length(timesteps)
    xsim = np.zeros((nsim, nx))
    ysim = np.zeros((nsim, ny))
    gsim = np.zeros((nsim, ng))
    tsol = np.zeros((nsim, 1))
    tsim = np.arange(0, nsim)*Ts

    gMPC = ginit  # initial previous measured input is the input at time instant -1.
    time_start = time.time()
    for i in range(nsim):

        yold = Cd @ x0 + Dd @ gMPC
        ysim[i, :] = yold

        x_init.value = x0  # set value to the x_init cvx parameter to x0
        gminus1.value = gMPC
        yminus1.value = yold
        r.value = np.array(2*[1.0])  # Reference output

        time_start = time.time()
        prob.solve(solver=OSQP, warm_start=True)
        tsol[i] = 1000*(time.time() - time_start)

        gMPC = g[:, 0].value
        gsim[i, :] = gMPC
        x0 = Ad.dot(x0) + Bd.dot(gMPC)
        xsim[i, :] = x0

    time_sim = time.time() - time_start

    # In[Plot time traces]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(tsim, ysim[:, 0], "k", label='p')
    #axes[0].plot(tsim, r * np.ones(np.shape(tsim)), "r--", label="pref")
    axes[0].set_title("Output (-)")

    axes[1].plot(tsim, ysim[:, 1], "k", label='p')
    #axes[0].plot(tsim, r * np.ones(np.shape(tsim)), "r--", label="pref")
    axes[1].set_title("Output (-)")

    axes[2].plot(tsim, gsim[:, 0], label="u")
    axes[2].plot(tsim, gsim[:, 1], label="u")
    #axes[2].plot(tsim, gref * np.ones(np.shape(tsim)), "r--", label="uref")
    axes[2].set_title("Input (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Timing]
    plt.figure()
    plt.hist(tsol[1:])
    plt.xlabel("MPC solution time (ms)")

    print(f"First MPC execution takes {tsol[0, 0]:.0f} ms")