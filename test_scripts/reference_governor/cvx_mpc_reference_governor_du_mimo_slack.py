import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import time
from cvxpy import Variable, Parameter, Minimize, Problem, OSQP, quad_form
from system_dynamics import Ad, Bd, Cd, Dd


if __name__ == "__main__":

    # In[Constants]
    Ts = 1.0  # MPC sampling time
    len_sim = 120  # simulation length (s)

    [nx, ng] = Bd.shape  # number of states and number or inputs
    [ny, nx_2] = Cd.shape  # number of outputs
    assert(nx == nx_2)

    # In[MPC Settings]

    # Prediction horizon
    Np = 40

    # Constraints
    gmin = np.array(2*[-1000.0])
    gmax = np.array(2*[1000.0])

    ymin = np.array(2*[-100.0])
    ymax = np.array(2*[100.0])

    Dgmin = np.array(2*[-2e-1])
    Dgmax = np.array(2*[2e-1])

    # Objective function
    Qy = np.diag(2*[20])  # penalty on y - r
    QDy = 10 * np.eye(ny)  # penalty on Delta y
    Qrg = 10 * np.eye(ny)
    QDg = 100 * sparse.eye(ny)  # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state, reference, command
    x0 = np.array(2*[0.0, 0.0])  # initial state
    y0 = Cd @ x0  # initial state
    gm1 = np.array(2 * [0.0])  # g at time -1, used for the constraint on Delta g

    # In[MPC Problem setup]
    g = Variable((ng, Np))
    x = Variable((nx, Np))
    eps_slack = Variable(ny)

    x_init = Parameter(nx)
    gminus1 = Parameter(ny)  # MPC command at time -1 (from previous MPC window or g_step_old for the first instant)
    yminus1 = Parameter(ny)  # system output at time -1 (from previous MPC window or y_step_old for the first instant)
    r = Parameter(ny)

    y = Cd @ x + Dd @ g  # system output definition
    objective = 0.0
    objective += quad_form(eps_slack, 1e4*np.eye(ny))  # constraint violation penalty on slack
    constraints = [x[:, 0] == x_init]  # initial state constraint
    constraints += [eps_slack >= 0.0]  # slack positive constraint

    for k in range(Np):

        # Objective function
        objective += quad_form(r - y[:, k], Qy)   # tracking cost
        objective += quad_form(r - g[:, k], Qrg)  # reference governor cost
        if k > 0:
            objective += quad_form(g[:, k] - g[:, k - 1], QDg)  # MPC command variation cost
            objective += quad_form(y[:, k] - y[:, k - 1], QDy)  # system output variation cost
        else:  # at k = 0...
            objective += quad_form(g[:, k] - gminus1, QDg)  # MPC command variation cost k=0
            objective += quad_form(y[:, k] - yminus1, QDy)  # system output variation cost k=0

        # Constraints
        if k < Np - 1:
            constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ g[:, k]]  # system dynamics constraint

        constraints += [ymin - eps_slack <= y[:, k], y[:, k] <= ymax + eps_slack]  # system output interval constraint
        constraints += [gmin <= g[:, k], g[:, k] <= gmax]  # MPC command interval constraint

        # MPC command variation constraint
        if k > 0:
            constraints += [Dgmin <= g[:, k] - g[:, k - 1], g[:, k] - g[:, k - 1] <= Dgmax]
        else:  # at k = 0...
            constraints += [Dgmin <= g[:, k] - gminus1, g[:, k] - gminus1 <= Dgmax]

    prob = Problem(Minimize(objective), constraints)

    # Simulate in closed loop
    n_sim = int(len_sim / Ts)  # simulation length(timesteps)
    x_sim = np.zeros((n_sim, nx))
    y_sim = np.zeros((n_sim, ny))
    g_sim = np.zeros((n_sim, ng))
    t_MPC = np.zeros((n_sim, 1))
    t_sim = np.arange(0, n_sim) * Ts

    g_step_old = gm1  # initial previous measured input is the input at time instant -1.
    x_step = x0
    y_step_old = y0
    time_start = time.time()
    for i in range(n_sim):

        x_sim[i, :] = x_step

        # MPC Control law computation
        time_start = time.time()
        x_init.value = x_step  # set value to the x_init cvx parameter to x0
        gminus1.value = g_step_old
        yminus1.value = y_step_old
        r.value = np.array(2*[1.0])  # Reference output
        prob.solve(solver=OSQP, warm_start=True)
        g_step = g[:, 0].value
        time_MPC = 1000*(time.time() - time_start)
        t_MPC[i] = time_MPC   # MPC control law computation time

        y_step = Cd @ x_step + Dd @ g_step
        y_sim[i, :] = y_step
        g_sim[i, :] = g_step

        # System update
        x_step = Ad @ x_step + Bd @ g_step
        y_step_old = y_step  # like an additional state
        g_step_old = g_step  # like an additional state

    time_sim = time.time() - time_start

    # In[Plot time traces]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(t_sim, y_sim[:, 0], "k", label='y1')
    axes[0].plot(t_sim, y_sim[:, 1], "b", label='y2')
    axes[0].set_title("Output (-)")

    axes[1].plot(t_sim, g_sim[:, 0], "k", label="g1")
    axes[1].plot(t_sim, g_sim[:, 1], "b", label="g2")
    axes[1].set_title("Input (-)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[Timing]
    plt.figure()
    plt.hist(t_MPC[1:])
    plt.xlabel("MPC solution time (ms)")

    print(f"First MPC execution takes {t_MPC[0, 0]:.0f} ms")
    print(f"Following MPC execution take {np.max(t_MPC[1:, 0]):.0f} ms in the worst case")