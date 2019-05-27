from setuptools import setup, find_packages

setup(
    name='pyMPC',
    version='0.1',
    url='https://github.com/forgi86/pyMPC.git',
    author='Marco Forgione',
    author_email='marco.forgione1986@gmail.com',
    description='MPC package for python',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'scipy', 'matplotlib >= 1.5.1', 'osqp'], # to be checked
)
