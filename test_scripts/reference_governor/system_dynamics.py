import numpy as np
import scipy
import control


# In[System dynamics]

Ts = 1.0
r_den_1 = 0.9  # magnitude of poles
wo_den_1 = 0.2  # phase of poles (approx 2.26 kHz)

# Build a second-order discrete-time dynamics with dcgain=1 (inner loop model)
G_1 = control.TransferFunction([1], [1, -2 * r_den_1 * np.cos(wo_den_1), r_den_1 ** 2], Ts)
G_1 = G_1 / control.dcgain(G_1)
G_1_ss = control.ss(G_1)

# SISO state-space matrices subsystem 11
A_1 = np.array(G_1_ss.A)
B_1 = np.array(G_1_ss.B)
C_1 = np.array(G_1_ss.C)
D_1 = np.array(G_1_ss.D)

r_den_2 = 0.9  # magnitude of poles
wo_den_2 = 0.4  # phase of poles (approx 2.26 kHz)

# Build a second-order discrete-time dynamics with dcgain=1 (inner loop model)
G_2 = control.TransferFunction([1], [1, -2 * r_den_2 * np.cos(wo_den_2), r_den_2 ** 2], Ts)
G_2 = G_2 / control.dcgain(G_2)
G_2_ss = control.ss(G_2)

# SISO state-space matrices subsystem 22
A_2 = np.array(G_2_ss.A)
B_2 = np.array(G_2_ss.B)
C_2 = np.array(G_2_ss.C)
D_2 = np.array(G_2_ss.D)

# MIMO state-space matrices
Ad = scipy.linalg.block_diag(A_1, A_2)
Bd = scipy.linalg.block_diag(B_1, B_2)
Cd = scipy.linalg.block_diag(C_1, C_2)
Dd = scipy.linalg.block_diag(D_1, D_2)