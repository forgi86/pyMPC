from setuptools import setup, find_packages

setup(
    name='python-mpc',
    version='0.1.1',
    url='https://github.com/forgi86/pyMPC.git',
    author='Marco Forgione',
    author_email='marco.forgione1986@gmail.com',
    description='MPC package for python',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'osqp'],  # to be checked
    extras_require={
        'cvx experiments': ["cvxpy"]
    }
)
