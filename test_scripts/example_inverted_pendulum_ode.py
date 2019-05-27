import numpy as np
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from scipy.integrate import ode
from pyMPC.mpc import MPCController

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

    [nx, nu] = Bc.shape # number of states and number or inputs

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

    # Brutal forward euler discretization
    Ad = np.eye(nx) + Ac*Ts
    Bd = Bc*Ts

    # Reference input and states
    xref = np.array([0.3, 0.0, 0.0, 0.0]) # reference state
    uref = np.array([0.0])    # reference input
    uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

    # Constraints
    xmin = np.array([-1.0, -100, -100, -100])
    xmax = np.array([0.3,   100.0, 100, 100])

    umin = np.array([-20])
    umax = np.array([20])

    Dumin = np.array([-5])
    Dumax = np.array([5])

    # Objective function weights
    Qx = sparse.diags([0.3, 0, 1.0, 0])   # Quadratic cost for states x0, x1, ..., x_N-1
    QxN = sparse.diags([0.3, 0, 1.0, 0])  # Quadratic cost for xN
    Qu = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
    QDu = 0.01 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

    # Initial state
    phi0 = 15*2*np.pi/360
    x0 = np.array([0, 0, phi0, 0]) # initial state
    system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
    system_dyn.set_initial_value(x0, 0)
    system_dyn.set_f_params(0.0)

    # Prediction horizon
    Np = 20

    K = MPCController(Ad,Bd,Np=Np, x0=x0,xref=xref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                      xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax,
                      eps_feas = 1e3)
    K.setup()

    # Simulate in closed loop
    [nx, nu] = Bd.shape # number of states and number or inputs
    len_sim = 20 # simulation length (s)
    nsim = int(len_sim/Ts) # simulation length(timesteps)
    xsim = np.zeros((nsim,nx))
    usim = np.zeros((nsim,nu))
    tsim = np.arange(0,nsim)*Ts

    time_start = time.time()

    xstep = x0
    uMPC =  uminus1
    for i in range(nsim):
        xsim[i,:] = xstep

        # MPC update and step. Could be in just one function call
        K.update(xstep, uMPC) # update with measurement
        uMPC = K.output() # MPC step (u_k value)
        usim[i,:] = uMPC

        # System simulation step
        system_dyn.set_f_params(uMPC) # set current input value
        system_dyn.integrate((i+1)*Ts)
        xstep = system_dyn.y

    time_sim = time.time() - time_start

    fig,axes = plt.subplots(3,1, figsize=(10,10))
    axes[0].plot(tsim, xsim[:,0], "k", label='p')
    axes[0].plot(tsim, xref[0]*np.ones(np.shape(tsim)), "r--", label="p_ref")
    axes[0].set_title("Position (m)")

    axes[1].plot(tsim, xsim[:,2]*360/2/np.pi, label="phi")
    axes[1].plot(tsim, xref[2]*360/2/np.pi*np.ones(np.shape(tsim)), "r--", label="phi_ref")
    axes[1].set_title("Angle (deg)")

    axes[2].plot(tsim, usim[:,0], label="u")
    axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="u_ref")
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()


