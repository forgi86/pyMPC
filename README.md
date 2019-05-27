# pyMPC

Implementation of an MPC controller in python based on the <a href="https://osqp.org/">OSQP</a> solver.

## Installation

1. Install the [QSQP](https://osqp.org/) solver.
```
pip install osqp
```
2. Copy or clone the pyMPC project in a local folder. For instance, run 
```
git clone https://github.com/forgi86/pyMPC.git
```
3. Install pyMPC by running
```
pip install -e .
```
in the pyMPC root folder (where the file setup.py is located).

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
