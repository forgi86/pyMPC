import numpy as np
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from pyMPC.mpc import MPCController
from scipy.integrate import ode
from pyMPC.kalman import kalman_design_simple, LinearStateEstimator

if __name__ == '__main__':

    # Constants #
    M = 0.5
    m = 0.2
    b = 0.1
    ftheta = 0.1
    l = 0.3
    g = 9.81

    Ts = 5e-3

    Ac =np.array([[0,       1,          0,                  0],
                  [0,       -b/M,       -(g*m)/M,           (ftheta*m)/M],
                  [0,       0,          0,                  1],
                  [0,       b/(M*l),    (M*g + g*m)/(M*l),  -(M*ftheta + ftheta*m)/(M*l)]])

    Bc = np.array([
        [0.0],
        [1.0/M],
        [0.0],
        [-1/(M*l)]
    ])

    Cc = np.array([[1., 0., 0., 0.],
                   [0., 0., 1., 0.]])

    Dc = np.zeros((2, 1))

    [nx, nu] = Bc.shape  # number of states and number or inputs
    ny = np.shape(Cc)[0]

    # Nonlinear dynamics ODE
    def f_ODE(t,x,u):
        #print(x)
        F = u
        v = x[1]
        theta = x[2]
        omega = x[3]
        der = np.zeros(nx)
        der[0] = v
        der[1] = (m * l * np.sin(theta) * omega ** 2 - m * g * np.sin(theta) * np.cos(theta) + m * ftheta * np.cos(
            theta) * omega + F - b * v) / (M + m * (1 - np.cos(theta) ** 2))
        der[2] = omega
        der[3] = ((M + m) * (g * np.sin(theta) - ftheta * omega) - m * l * omega ** 2 * np.sin(theta) * np.cos(
            theta) - (
                          F - b * v) * np.cos(theta)) / (l * (M + m * (1 - np.cos(theta) ** 2)))
        return der

    # Simple forward euler discretization
    Ad = np.eye(nx) + Ac*Ts
    Bd = Bc*Ts
    Cd = Cc
    Dd = Dc

    # Standard deviation of the measurement noise on position and angle
    std_npos = 0*0.005
    std_nphi = 0*0.005

    # Reference input and states
    xref = np.array([0.3, 0.0, 0.0, 0.0]) # reference state
    uref = np.array([0.0])    # reference input
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-1.0, -100, -100, -100])
    xmax = np.array([1.0,   100.0, 100, 100])

    umin = np.array([-20])
    umax = np.array([20])

    Dumin = np.array([-5])
    Dumax = np.array([5])

    # Objective function weights
    Qx = sparse.diags([0.3, 0, 1.0, 0])   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = sparse.diags([0.3, 0, 1.0, 0])  # Quadratic cost for xN
    Qu = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 0.01 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initialize simulation system
    phi0 = 15*2*np.pi/360
    x0 = np.array([0, 0, phi0, 0]) # initial state
    system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
    system_dyn.set_initial_value(x0, 0)
    system_dyn.set_f_params(0.0)

    # Basic Kalman filter design
    Q_kal = 10 * np.eye(nx)
    R_kal = np.eye(ny)
    L, P, W = kalman_design_simple(Ad, Bd, Cd, Dd, Q_kal, R_kal, type='filter')
    x0_est = x0
    KF = LinearStateEstimator(x0_est, Ad, Bd, Cd, Dd, L)

    # Prediction horizon
    Np = 200

    # Initialize controller
    K = MPCController(Ad,Bd,Np=Np, x0=x0,xref=xref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                      xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax,
                      eps_feas = 1e3)
    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape  # number of states and number or inputs
    len_sim = 10  # simulation length (s)
    nsim = int(len_sim / Ts)  # simulation length(timesteps)
    x_vec = np.zeros((nsim, nx))
    #x_vec_EA = np.zeros((nsim, nx))

    y_vec = np.zeros((nsim, ny))
    y_meas_vec = np.zeros((nsim, ny))
    y_est_vec = np.zeros((nsim, ny))
    x_est_vec = np.zeros((nsim, nx))
    u_vec = np.zeros((nsim, nu))
    t_vec = np.arange(0, nsim) * Ts
    t_MPC_CPU = np.arange(0, nsim) * Ts
    x_MPC_pred = np.zeros((nsim, Np+1, nx)) # on-line predictions from the Kalman Filter

    time_start = time.time()

    uMPC = uminus1
    y_step = None
    ymeas_step = None
    #x_step = x0
    for i in range(nsim):
        # Output for step i
        # System
        y_step = Cd.dot(system_dyn.y)  # y[i] from the system
        ymeas_step = y_step
        ymeas_step[0] += std_npos * np.random.randn()
        ymeas_step[1] += std_nphi * np.random.randn()
        # Estimator

#        time_MPC_start = time.time()
        uMPC, infoMPC = K.output(return_x_seq=True) # u[i] = k(\hat x[i]) possibly computed at time instant -1
#        t_MPC_CPU[i] = time.time() - time_MPC_start

        x_MPC_pred[i, :, :] = infoMPC['x_seq']   # x_MPC_pred[i,i+1,...| possibly computed at time instant -1]

        # Save output for step i
        y_vec[i, :] = y_step # y[i]
        y_meas_vec[i,:] = ymeas_step # y_meas[i]
        x_vec[i, :] = system_dyn.y  # x[i]
        y_est_vec[i,:] = KF.y  # \hat y[i|i-1]
        x_est_vec[i, :] = KF.x # \hat x[i|i-1]
        u_vec[i, :] = uMPC    # u[i]
#        x_vec_EA[i,:] = x_step

        # Update i+1
        # System
        system_dyn.set_f_params(uMPC) # set current input value to uMPC
        system_dyn.integrate(system_dyn.t + Ts)
        #x_step = system_dyn.y
#        der = f_ODE(0,x_step,uMPC)
        #x_step = x_step + der * Ts  # true system evolves to x[i+1]
        #system_dyn.set_initial_value(x_step, 0)

        # Kalman filter: update and predict
        KF.update(ymeas_step) # update  \hat x[i|i-1] to \hat x[i|i] updated using ymeas[i]
        KF.predict(uMPC)      # predict \hat x[i+1|i] using u[i]

        # MPC update for step i+1
        time_MPC_start = time.time()
        K.update(KF.x, uMPC)  # update with measurement (and possibly pre-compute u[i+1])
        t_MPC_CPU[i] = time.time() - time_MPC_start

    time_sim = time.time() - time_start

    fig,axes = plt.subplots(3,1, figsize=(10,10))
    axes[0].plot(t_vec, x_vec[:,0], "k", label='p')
    axes[0].plot(t_vec, xref[0]*np.ones(np.shape(t_vec)), "r--", label="p_ref")
    #axes[0].plot(t_vec, x_vec_EA[:,0]*np.ones(np.shape(t_vec)), "r--", label="p_EA")
    axes[0].set_title("Position (m)")

    axes[1].plot(t_vec, x_vec[:,2]*360/2/np.pi, label="phi")
    axes[1].plot(t_vec, xref[2]*360/2/np.pi*np.ones(np.shape(t_vec)), "r--", label="phi_ref")
    #axes[1].plot(t_vec, x_vec_EA[:,2]*np.ones(np.shape(t_vec)), "r--", label="p_EA")
    axes[1].set_title("Angle (deg)")

    axes[2].plot(t_vec, u_vec[:,0], label="u")
    axes[2].plot(t_vec, uref*np.ones(np.shape(t_vec)), "r--", label="u_ref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()


    fig,ax = plt.subplots(1,1, figsize=(5,5))
    ax.hist(t_MPC_CPU*1000, bins=100)
    ax.grid(True)
    ax.set_xlabel('MPC computation CPU time (ms)')
