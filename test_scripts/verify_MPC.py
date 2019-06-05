import numpy as np
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from pyMPC.mpc import MPCController


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    # Constants #
    Ts = 0.2 # sampling time (s)
    M = 2    # mass (Kg)
    b = 0.3  # friction coefficient (N*s/m)

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

    # Reference input and states
    pref = 7.0
    vref = 0.0
    xref = np.array([pref, vref]) # reference state
    uref = np.array([5])    # reference input
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-10, -10.0])
    xmax = np.array([100.0,   100.0])

    umin = np.array([-1.2])
    umax = np.array([1.2])

    Dumin = np.array([-2e-1])
    Dumax = np.array([2e-1])

    # Objective function
    Qx = sparse.diags([0.5, 0.1])   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = sparse.diags([0.5, 0.1])  # Quadratic cost for xN
    Qu = 0.1 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 0.0 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    x0 = np.array([0.1, 0.2]) # initial state

    # Prediction horizon
    Np = 25
    Nc = 25

    K = MPCController(Ad,Bd,Np=Np,Nc=Nc,x0=x0,xref=xref,uref=uref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                      xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax)
    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape # number of states and number or inputs
    len_sim = 30 # simulation length (s)
    nsim = int(len_sim/Ts) # simulation length(timesteps)
    xsim = np.zeros((nsim,nx))
    usim = np.zeros((nsim,nu))
    tsim = np.arange(0,nsim)*Ts
    J_opt = np.zeros((nsim,1))

    time_start = time.time()
    xstep = x0
    for i in range(nsim):
        uMPC, info = K.output(return_u_seq=True, return_x_seq=True, return_eps_seq=True, return_status=True, return_obj_val=True)
        xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # system step
        J_opt[i,:] = info['obj_val']
        K.update(xstep) # update with measurement
        K.solve()
        xsim[i,:] = xstep
        usim[i,:] = uMPC

    time_sim = time.time() - time_start

    #K.__controller_function__(np.array([0,0]), np.array([0]))

    fig,axes = plt.subplots(4,1, figsize=(10,10))
    axes[0].plot(tsim, xsim[:,0], "k", label='p')
    axes[0].plot(tsim, xref[0]*np.ones(np.shape(tsim)), "r--", label="pref")
    axes[0].set_title("Position (m)")

    axes[1].plot(tsim, xsim[:,1], label="v")
    axes[1].plot(tsim, xref[1]*np.ones(np.shape(tsim)), "r--", label="vref")
    axes[1].set_title("Velocity (m/s)")

    axes[2].plot(tsim, usim[:,0], label="u")
    axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="uref")
    axes[2].set_title("Force (N)")

    axes[3].plot(tsim, J_opt, "k", label="J_opt")

    for ax in axes:
        ax.grid(True)
        ax.legend()


    x_seq = info['x_seq']
    eps_seq = info['eps_seq']
    u_seq = info['u_seq']
    eps_recalc_seq = np.zeros(eps_seq.shape)
    u_old = K.uminus1_rh
    x_new_dyn = x_seq[0,:]
    J_recalc_x = 0
    J_recalc_u = 0
    J_recalc_Du = 0
    J_recalc_eps = 0


    for i in range(Np):
        x_i = x_seq[i,:] # x[i]
        eps_i = eps_seq[i]
        if i < Nc:
            u_i = u_seq[i,:] # u[i]
        else:
            u_i = u_seq[Nc-1,:]
        eps_i_recalc = x_new_dyn - x_i # eps[i]
        J_recalc_x += 1/2*(x_i -xref).dot(K.Qx.dot((x_i -xref)))
        J_recalc_u += 1/2*(u_i -uref).dot(K.Qu.dot((u_i -uref)))
        J_recalc_Du += 1/2*(u_i -u_old).dot(K.QDu.dot((u_i -u_old)))
        J_recalc_eps += 1/2*(eps_i_recalc).dot(K.Qeps.dot((eps_i_recalc)))
        x_new_dyn = K.Ad.dot(x_i) + K.Bd.dot(u_i)
        u_old = u_i

    x_i = x_seq[Np,:]
    eps_i_recalc = x_new_dyn - x_i
    J_recalc_x += 1/2*(x_i -xref).dot(K.QxN.dot((x_i -xref)))
    J_recalc_eps += 1/2*(eps_i_recalc).dot(K.Qeps.dot((eps_i_recalc)))

    J_recalc = J_recalc_x + J_recalc_u + J_recalc_Du + J_recalc_eps
