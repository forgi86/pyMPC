.. pyMPC documentation master file, created by
   sphinx-quickstart on Tue May 28 18:47:14 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyMPC
=====

A python library for Model Predictive Control
---------------------------------------------

pyMPC is an open-source python library for Model Predictive Control (MPC).
The project is hosted on this `GitHub repository <https://github.com/forgi86/pyMPC>`_.

Requirements
------------

As a bare minimum, you will need:

* `numpy <https://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `OSQP <https://osqp.org/>`_
* `matplotlib <https://matplotlib.org/>`_

All dependencies should be installed automatically following the pyMPC installation instructions below

Installation
------------
1. Copy or clone the pyMPC project in a local folder. For instance, run

.. code-block:: bash

   git clone https://github.com/forgi86/pyMPC.git


2. Install pyMPC by running

.. code-block:: bash

   pip install -e .

in the pyMPC root folder (where the file setup.py is located).

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
