# pyMPC

Implementation of a linear, constrained MPC controller in python.

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
