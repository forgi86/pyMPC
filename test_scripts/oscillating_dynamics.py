import matplotlib.pyplot as plt
import control
import numpy as np

ts = 1.0
r_den = 0.9  # magnitude of poles
wo_den = 0.2  # phase of poles (approx 2.26 kHz)

H_noise = control.TransferFunction([1], [1, -2 * r_den * np.cos(wo_den), r_den ** 2], ts)
H_noise = H_noise/control.dcgain(H_noise)
H_ss = control.ss(H_noise)
t, val = control.step_response(H_ss, np.arange(100))


plt.plot(val[0, :])