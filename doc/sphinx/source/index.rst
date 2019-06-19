.. pyMPC documentation master file, created by
   sphinx-quickstart on Tue May 28 18:47:14 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyMPC
=====
---------------------------------------------
A python library for Model Predictive Control
---------------------------------------------

pyMPC is an open-source python library for Model Predictive Control (MPC).
The project is hosted on this `GitHub repository <https://github.com/forgi86/pyMPC>`_.

Requirements
------------

In order to run pyMPC, you need a python 3.x environment and the following packages:

* `numpy <https://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `OSQP <https://osqp.org/>`_
* `matplotlib <https://matplotlib.org/>`_

The installation procedure below should take case of installing pyMPC with all its dependencies.

Installation
------------
1. Copy or clone the pyMPC project into a local folder. For instance, run

.. code-block:: bash

   git clone https://github.com/forgi86/pyMPC.git

from the command line

2. Navigate to your local pyMPC folder

.. code-block:: bash

   cd PYMPC_LOCAL_FOLDER

where PYMPC_LOCAL_FOLDER is the folder where you have just downloaded the code in step 2

3. Install pyMPC in your python environment: run

.. code-block:: bash

   pip install -e .

from the command line, in the working folder PYMPC_LOCAL_FOLDER

Usage
-----

This code snippets illustrates the use of the MPCController class:

.. code-block:: python

   from pyMPC.mpc import MPCController

   ...

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


Examples
--------

Full working examples are given in the `examples <https://github.com/forgi86/pyMPC/tree/master/examples/>`_ folder on git:
 * `Point mass, full state feedback <https://github.com/forgi86/pyMPC/tree/master/examples/example_point_mass.ipynb>`_
 * `Cart-pole system, full state feedback <https://github.com/forgi86/pyMPC/tree/master/examples/example_inverted_pendulum.ipynb>`_
 * `Cart-pole system, with Kalman Filter <https://github.com/forgi86/pyMPC/tree/master/examples/example_inverted_pendulum_kalman.ipynb>`_


Content
-------
.. toctree::
   :maxdepth: 2

   code
   math



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
