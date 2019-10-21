## pyMPC on Raspberry PI

pyMPC can run on Raspberry PI (we tested model 3B). This procedure may also work on other versions and maybe other ARM machines, but we haven't tested yet.

## Installation procedure

We assume you already have Raspbian installed on the Raspberry PI. 

1. Install the following packages via apt
```
apt-get install git
apt-get install cmake
```

2. Install the berryconda conda distribution 
```
wget https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh 
chmod +x Berryconda3-2.0.0-Linux-armv7l.sh
./Berryconda3-2.0.0-Linux-armv7l.sh
```
In the installation process, you will be asked for an installation folder. The default /home/pi/berryconda3 is fine

3. Install required packages in your new (berry)conda distribtion
```
cd /home/pi/berryconda3
./conda install numpy scipy matplotlib
./conda install -c numba numba
./pip install control
apt-get install cmake
./pip install osqp
```

4. Get a local copy the pyMPC project
```
git clone https://github.com/forgi86/pyMPC.git
```

5. Install pyMPC. In the folder where you cloned the project, run

```
pip install -e .
```
