The Math
=================================

The MPC problem is:

.. math::
    \newcommand{\MPC}{\mathrm{MPC}}
    \newcommand{\varx}{\mathbf{x}}
    \newcommand{\varu}{\mathbf{u}}
    \newcommand{\slack}{\epsilon}
    \newcommand{\QxN}{Q_{x_N}}
    \newcommand{\Qx}{Q_{x}}
    \newcommand{\Qu}{Q_{u}}
    \newcommand{\Qdu}{Q_{\Delta u}}
    \newcommand{\Np}{{N_p}}
    \newcommand{\Nc}{{N_c}}
    \newcommand{\blkdiag}{\text{blkdiag}}

.. math::

    \begin{multline}
      \arg \min_{\mathbf{x},\mathbf{u}}
        \big(x_\Np - x_{ref}\big)^\top Q_x \big(x_\Np - x_{ref}\big) +
        \bigg [
         \sum_{k=0}^{\Np-1} \big(x_k - x_{ref}\big)^\top Q_x \big(x_k - x_{ref}\big) +
        \big(u_k - u_{ref}\big)^\top Q_u \big(u_k - u_{ref}\big) +
        \Delta u_k^\top Q_u \Delta u_k
        \bigg ]
    \end{multline}.

Under the hood, pyMPC transofrms this optimization in a form that can be solved using a standard QP
solver, i.e.,

.. math::

    \begin{align}
     &\min \frac{1}{2} x^\top P x +  q^\top x \\
     &\text{subject to} \nonumber \\
     &l \leq Ax \leq u
    \end{align}



