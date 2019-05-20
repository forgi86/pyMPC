# pyMPC

Implementation of an MPC controller in python based on the <a href="https://osqp.org/">OSQP</a> solver.

## Installation

1. install the [QSQP](https://osqp.org/) solver.
```
pip install osqp
```
2. Copy or clone the pyMPC project in a local folder. For instance, run 
```
git clone https://github.com/forgi86/pyMPC.git
```
3. Run the command
```
pip install -e .
```
in the pyMPC project root folder (where the file setup.py is located)

## Usage 

This is a snippet code illustrating how to use the MPCController class contained in the pyMPC project:

```
from pyMPC.mpc import MPCController

K = MPCController(Ad,Bd,Np=20, x0=x0,xref=xref,uminus1=uminus1,
                  Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                  xmin=xmin,xmax=xmax,umin=umin,umax=umax,Dumin=Dumin,Dumax=Dumax)
K.setup()

...

xstep = x0
for i in range(nsim): 
  uMPC = K.step()
  xstep = Ad.dot(xstep) + Bd.dot(uMPC)  # system simulation steps
  K.update(xstep) # update with measurement
```
A full working example is available [here](examples/example_mpc.ipynb). 
