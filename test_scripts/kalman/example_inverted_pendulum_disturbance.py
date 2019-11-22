import numpy as np
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.interpolate import interp1d
from kalman import kalman_filter_simple, kalman_filter, LinearStateEstimator
from pyMPC.mpc import MPCController
import control
import control.matlab


RAD_TO_DEG = 180.0/np.pi

if __name__ == '__main__':

    # Constants #
    M = 0.5
    m = 0.2
    b = 0.1
    ftheta = 0.1
    l = 0.3
    g = 9.81

    Ts_MPC = 10e-3
    Ts_sim = 1e-3
    ratio_Ts = int(Ts_MPC // Ts_sim)

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
        der = np.zeros(4)
        der[0] = v
        der[1] = (m * l * np.sin(theta) * omega ** 2 - m * g * np.sin(theta) * np.cos(theta) + m * ftheta * np.cos(
            theta) * omega + F - b * v) / (M + m * (1 - np.cos(theta) ** 2))
        der[2] = omega
        der[3] = ((M + m) * (g * np.sin(theta) - ftheta * omega) - m * l * omega ** 2 * np.sin(theta) * np.cos(
            theta) - (
                          F - b * v) * np.cos(theta)) / (l * (M + m * (1 - np.cos(theta) ** 2)))
        return der

    # Brutal forward euler discretization
    Ad = np.eye(nx) + Ac * Ts_MPC
    Bd = Bc * Ts_MPC
    Cd = Cc
    Dd = Dc

    std_npos = 1*0.005
    std_nphi = 1*0.005

    # Force disturbance
    wu = 10  # bandwidth of the force disturbance
    std_du = 0.1
    Ts = 1e-3
    Hu = control.TransferFunction([1], [1 / wu, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts)
    t_imp = np.arange(5000) * Ts
    t, y = control.impulse_response(Hud, t_imp)
    y = y[0]
    std_tmp = np.sqrt(np.sum(y ** 2))  # np.sqrt(trapz(y**2,t))
    Hu = Hu / (std_tmp) * std_du
    N_sim = 100000
    e = np.random.randn(N_sim)
    te = np.arange(N_sim) * Ts
    _, d_fast, _ = control.forced_response(Hu, te, e)
    d_fast = d_fast[1000:]

    # Reference input and states
    t_ref_vec = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    p_ref_vec = np.array([0.0, 0.3, 0.3, 0.0, 0.0])
    rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='zero')

    def xref_fun(t):
        return np.array([rp_fun(t), 0.0, 0.0, 0.0])


    xref = xref_fun(0) # reference state
    uref = np.array([0.0])    # reference input
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-1.0, -100, -100, -100])
    xmax = np.array([0.5,   100.0, 100, 100])

    umin = np.array([-20])
    umax = np.array([20])

    Dumin = np.array([-100*Ts_MPC]) # 100 N/s
    Dumax = np.array([100*Ts_MPC])

    # Objective function weights
    Qx = sparse.diags([1.0, 0, 10.0, 0])   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = sparse.diags([1.0, 0, 10.0, 0])  # Quadratic cost for xN
    Qu = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 0.1 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    phi0 = 15*2*np.pi/360
    x0 = np.array([0, 0, phi0, 0]) # initial state
    t0 = 0
    system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
    system_dyn.set_initial_value(x0, t0)
    system_dyn.set_f_params(0.0)

    # Prediction horizon
    Np = 100
    Nc = 100

    K = MPCController(Ad,Bd,Np=Np, x0=x0,xref=xref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                      xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax,
                      eps_feas = 1e3)
    K.setup()

    # Basic Kalman filter design
    Q_kal =  np.diag([0.0001, 100, 0.0001, 100])
    #Q_kal =  np.diag([100, 100, 100, 100])
    R_kal = 1*np.eye(ny)
    L, P, W = kalman_filter_simple(Ad, Bd, Cd, Dd, Q_kal, R_kal)
    #Bd_kal = np.hstack([Bd, Bd])
    #Dd_kal = np.array([[0, 0]])
    #Q_kal = np.array([[1e4]]) # nw x nw matrix, w general (here, nw = nu)
    #R_kal = np.eye(ny) # ny x ny)
    #L,P,W = kalman_filter(Ad, Bd_kal, Cd, Dd_kal, Q_kal, R_kal)

    x0_est = x0
    KF = LinearStateEstimator(x0_est, Ad, Bd, Cd, Dd,L)


    # Simulate in closed loop
    [nx, nu] = Bd.shape # number of states and number or inputs
    len_sim = 40 # simulation length (s)
    nsim = int(np.ceil(len_sim / Ts_MPC))  # simulation length(timesteps) # watch out! +1 added, is it correct?
    t_vec = np.zeros((nsim, 1))
    t_calc_vec = np.zeros((nsim,1)) # computational time to get MPC solution (+ estimator)
    status_vec = np.zeros((nsim,1))
    x_vec = np.zeros((nsim, nx))
    x_ref_vec = np.zeros((nsim, nx))
    y_vec = np.zeros((nsim, ny))
    y_meas_vec = np.zeros((nsim, ny))
    y_est_vec = np.zeros((nsim, ny))
    x_est_vec = np.zeros((nsim, nx))
    u_vec = np.zeros((nsim, nu))
    x_MPC_pred = np.zeros((nsim, Np+1, nx)) # on-line predictions from the Kalman Filter

    nsim_fast = int(len_sim // Ts_sim)
    t_vec_fast = np.zeros((nsim_fast, 1))
    x_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluation
    x_ref_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluatio
    u_vec_fast = np.zeros((nsim_fast, nu)) # finer integration grid for performance evaluatio
    Fd_vec_fast = np.zeros((nsim_fast, nu))  #

    t_step = t0
    u_MPC = None
    for idx_fast in range(nsim_fast):

        ## Determine step type: fast simulation only or MPC step
        idx_MPC = idx_fast // ratio_Ts
        run_MPC = (idx_fast % ratio_Ts) == 0

        # Output for step i
        # Ts_MPC outputs
        if run_MPC: # it is also a step of the simulation at rate Ts_MPC
            t_vec[idx_MPC, :] = t_step
            x_vec[idx_MPC, :] = system_dyn.y
            xref_MPC = xref_fun(t_step)  # reference state
            x_ref_vec[idx_MPC,:] = xref_MPC

            u_MPC, info_MPC = K.output(return_x_seq=True, return_status=True)  # u[i] = k(\hat x[i]) possibly computed at time instant -1
            x_MPC_pred[idx_MPC, :, :] = info_MPC['x_seq']  # x_MPC_pred[i,i+1,...| possibly computed at time instant -1]
            u_vec[idx_MPC, :] = u_MPC

            y_step = Cd.dot(system_dyn.y)  # y[i] measured from the system
            ymeas_step = y_step
            ymeas_step[0] += std_npos * np.random.randn()
            ymeas_step[1] += std_nphi * np.random.randn()
            y_meas_vec[idx_MPC,:] = ymeas_step
            y_vec[idx_MPC,:] = y_step
            status_vec[idx_MPC,:] = (info_MPC['status'] != 'solved')

        # Ts_fast outputs
        t_vec_fast[idx_fast,:] = t_step
        x_vec_fast[idx_fast, :] = system_dyn.y
        x_ref_vec_fast[idx_fast, :] = xref_MPC
        u_fast = u_MPC + d_fast[idx_fast]
        u_vec_fast[idx_fast,:] = u_fast
        Fd_vec_fast[idx_fast,:] = d_fast[idx_fast]

        ## Update to step i+1

        # Controller simulation step at rate Ts_MPC
        if run_MPC:
            time_calc_start = time.time()
            # Kalman filter: update and predict
            KF.update(ymeas_step) # \hat x[i|i]
            KF.predict(u_MPC)    # \hat x[i+1|i]
            # MPC update
            #K.update(system_dyn.y, u_MPC, xref=xref_MPC) # update with measurement
            K.update(KF.x, u_MPC, xref=xref_MPC)  # update with measurement
            t_calc_vec[idx_MPC,:] = time.time() - time_calc_start

        # System simulation step at rate Ts_fast
        system_dyn.set_f_params(u_fast)
        system_dyn.integrate(t_step + Ts_sim)

        # Time update
        t_step += Ts_sim


    y_OL_pred = np.zeros((nsim-Np-1, Np+1, ny)) # on-line predictions from the Kalman Filter
    y_MPC_pred = x_MPC_pred[:, :, [0, 2]] # how to vectorize C * x_MPC_pred??
    y_MPC_err = np.zeros(np.shape(y_OL_pred))
    y_OL_err = np.zeros(np.shape(y_OL_pred))
    for i in range(nsim-Np-1):
        u_init = u_vec[i:i+Np+1, :]
        x_init = x_vec[i,:]
        y_OL_pred[i,:,:] = KF.sim(u_init,x_init)
        y_OL_err[i, :, :] = y_OL_pred[i, :, :] - y_meas_vec[i:i + Np + 1]
        y_MPC_err[i, :, :] = y_MPC_pred[i, :, :] - y_meas_vec[i:i + Np + 1]


    fig,axes = plt.subplots(3,1, figsize=(10,10), sharex=True)
    axes[0].plot(t_vec, y_meas_vec[:, 0], "b", label='p_meas')
    axes[0].plot(t_vec_fast, x_vec_fast[:, 0], "k", label='p')
    axes[0].plot(t_vec_fast, x_ref_vec_fast[:,0], "r--", label="p_ref")
    idx_pred = 0
    axes[0].plot(t_vec[idx_pred:idx_pred+Np+1], y_OL_pred[idx_pred, :, 0], 'r', label='Off-line k-step prediction')
    axes[0].plot(t_vec[idx_pred:idx_pred+Np+1], y_MPC_pred[idx_pred, :, 0], 'c', label='MPC k-step prediction' )
    axes[0].set_title("Position (m)")

    axes[1].plot(t_vec, y_meas_vec[:, 1]*RAD_TO_DEG, "b", label='phi_meas')
    axes[1].plot(t_vec_fast, x_vec_fast[:, 2]*RAD_TO_DEG, 'k', label="phi")
    axes[1].plot(t_vec_fast, x_ref_vec_fast[:,2]*RAD_TO_DEG, "r--", label="phi_ref")
    idx_pred = 0
    axes[1].plot(t_vec[idx_pred:idx_pred+Np+1], y_OL_pred[idx_pred, :, 1]*RAD_TO_DEG, 'r', label='Off-line k-step prediction')
    axes[1].plot(t_vec[idx_pred:idx_pred+Np+1], y_MPC_pred[idx_pred, :, 1]*RAD_TO_DEG, 'c', label='MPC k-step prediction' )
    axes[1].set_title("Angle (deg)")

    axes[2].step(t_vec_fast, u_vec_fast[:, 0], where='post', label="F")
    axes[2].step(t_vec, u_vec[:, 0], where='post',  label="F_MPC")
    axes[2].step(t_vec_fast, Fd_vec_fast[:, 0], where='post', label="F_d")
    axes[2].plot(t_vec, uref * np.ones(np.shape(t_vec)), "r--", label="u_ref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    # In[CPU time]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(t_calc_vec*1000, bins=100)
    ax.grid(True)
    ax.set_title('CPU time (ms)')
