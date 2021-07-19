# pyMPC

Linear Constrained Model Predictive Control (MPC) in Python:

<img src="http://www.marcoforgione.it/pyMPC/img/pyMPC_formula_1.png"></img>

where 
<img src="http://www.marcoforgione.it/pyMPC/img/pyMPC_formula_2.png" height="14"></img>
## Requirements

pyMPC requires the following packages:
* numpy
* scipy
* [OSQP](https://osqp.org/)
* matplotlib

## Installation

### Stable version from PyPI

Run the command 

```
pip install python-mpc
```
This will install the [stable version](https://pypi.org/project/python-mpc/0.1.1/) of pyMPC from the PyPI package repository.

### Latest version from GitHub
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

## Supported platforms

We successfully tested pyMPC on the following platforms:
* Windows 10 on a PC with x86-64 CPU
* Ubuntu 18.04 LTS on a PC with x86-64 CPU
* Raspbian Buster on a Raspberry PI 3 rev B

Detailed instructions for the Raspberry PI platform are available [here](README_PI.md).

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

## Contributing

I am slowly adding new functionalities to pyMPC according to my research needs. If you also wanna contribute, feel free to write me an email: marco.forgione@idsia.ch

## Citing

If you find this project useful, we encourage you to

* Star this repository :star: 
* Cite the [paper](https://arxiv.org/pdf/1911.13021) 
```
@inproceedings{forgione2020efficient,
  title={Efficient Calibration of Embedded {MPC}},
  author={Forgione, Marco and Piga, Dario and Bemporad, Alberto},
  booktitle={Proc. of the 21st IFAC World Congress 2020, Berlin, Germany, July 12-17 2020},
  year={2020}
}
```
