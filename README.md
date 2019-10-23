# pyMPC

Implementation of a linear, constrained MPC controller in Python:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;&\min_{u_0,&space;\dots,&space;u_{N_{p}-1},x_0,\dots,x_{N_p}}&space;\sum_{k=0}^{N_p-1}&space;\bigg[&space;\big(x_k&space;-&space;x_{ref}\big)^\top&space;Q_x\big(x_k&space;-&space;x_{ref}\big)&space;&plus;&space;\big(u_k&space;-&space;u_{ref}\big)^\top&space;Q_u&space;\big(u_k&space;-&space;u_{ref}\big)&space;&plus;&space;\Delta&space;u_k^\top&space;Q_{\Delta}&space;\Delta&space;u_k&space;\bigg&space;]&space;&plus;&space;\big(x_{N_p}&space;-&space;x_{ref}\big)^\top&space;Q_x\big(x_{N_p}&space;-&space;x_{ref}\big)&space;\\&space;&\text{subject&space;to}&space;\nonumber\\&space;&x_{k&plus;1}&space;=&space;Ax_k&space;&plus;&space;B&space;u_k&space;\label{eq:linear_dynamics}&space;\\&space;&u_{min}&space;\leq&space;u_k&space;\leq&space;u_{max}\\&space;&x_{min}&space;\leq&space;x_k&space;\leq&space;x_{max}\\&space;&\Delta&space;u_{min}&space;\leq&space;\Delta&space;u_k&space;\leq&space;\Delta&space;u_{max}\\&space;&u_{-1}&space;=&space;\bar&space;u&space;\\&space;&x_0&space;=&space;\bar&space;x&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&\min_{u_0,&space;\dots,&space;u_{N_{p}-1},x_0,\dots,x_{N_p}}&space;\sum_{k=0}^{N_p-1}&space;\bigg[&space;\big(x_k&space;-&space;x_{ref}\big)^\top&space;Q_x\big(x_k&space;-&space;x_{ref}\big)&space;&plus;&space;\big(u_k&space;-&space;u_{ref}\big)^\top&space;Q_u&space;\big(u_k&space;-&space;u_{ref}\big)&space;&plus;&space;\Delta&space;u_k^\top&space;Q_{\Delta}&space;\Delta&space;u_k&space;\bigg&space;]&space;&plus;&space;\big(x_{N_p}&space;-&space;x_{ref}\big)^\top&space;Q_x\big(x_{N_p}&space;-&space;x_{ref}\big)&space;\\&space;&\text{subject&space;to}&space;\nonumber\\&space;&x_{k&plus;1}&space;=&space;Ax_k&space;&plus;&space;B&space;u_k&space;\label{eq:linear_dynamics}&space;\\&space;&u_{min}&space;\leq&space;u_k&space;\leq&space;u_{max}\\&space;&x_{min}&space;\leq&space;x_k&space;\leq&space;x_{max}\\&space;&\Delta&space;u_{min}&space;\leq&space;\Delta&space;u_k&space;\leq&space;\Delta&space;u_{max}\\&space;&u_{-1}&space;=&space;\bar&space;u&space;\\&space;&x_0&space;=&space;\bar&space;x&space;\end{align*}" title="\begin{align*} &\min_{u_0, \dots, u_{N_{p}-1},x_0,\dots,x_{N_p}} \sum_{k=0}^{N_p-1} \bigg[ \big(x_k - x_{ref}\big)^\top Q_x\big(x_k - x_{ref}\big) + \big(u_k - u_{ref}\big)^\top Q_u \big(u_k - u_{ref}\big) + \Delta u_k^\top Q_{\Delta} \Delta u_k \bigg ] + \big(x_{N_p} - x_{ref}\big)^\top Q_x\big(x_{N_p} - x_{ref}\big) \\ &\text{subject to} \nonumber\\ &x_{k+1} = Ax_k + B u_k \label{eq:linear_dynamics} \\ &u_{min} \leq u_k \leq u_{max}\\ &x_{min} \leq x_k \leq x_{max}\\ &\Delta u_{min} \leq \Delta u_k \leq \Delta u_{max}\\ &u_{-1} = \bar u \\ &x_0 = \bar x \end{align*}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;u_k&space;=&space;u_k&space;-&space;u_{k-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;u_k&space;=&space;u_k&space;-&space;u_{k-1}" title="\Delta u_k = u_k - u_{k-1}" /></a>.
## Requirements

pyMPC requires the following packages:
* numpy
* scipy
* [OSQP](https://osqp.org/)
* matplotlib

## Installation

1. Get a local copy the pyMPC project. For instance, run 
```
git clone https://github.com/forgi86/pyMPC.git
```
in a terminal to clone the project using git. Alternatively, download the zipped pyMPC project from [this link](https://github.com/forgi86/pyMPC/zipball/master) and extract it in a local folder

2. Install pyMPC by running
```
pip install -e .
```
in the pyMPC project root folder (where the file setup.py is located).

## Usage 

This code snippets illustrates the use of the MPCController class:

```
from pyMPC.mpc import MPCController

K = MPCController(Ad,Bd,Np=20, x0=x0,xref=xref,uminus1=uminus1,
                  Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                  xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax)
K.setup()

...

xstep = x0
for i in range(nsim): 
  uMPC = K.output()
  xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # system simulation steps
  K.update(xstep) # update with measurement
```
Full working examples are given in the [examples](examples) folder:
 * [Point mass with input force and friction](examples/example_point_mass.ipynb)
 * [Inverted pendulum on a cart](examples/example_inverted_pendulum.ipynb)
 * [Inverted pendulum on a cart with kalman filter](examples/example_inverted_pendulum_kalman.ipynb)
