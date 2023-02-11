# General
sudo apt update

sudo apt install gcc
sudo aptitude install g++ # Answers of the questions: n; 1; y; y. (Procedure explained in: https://github.com/madMAx43v3r/chia-plotter/issues/199)
sudo apt install gfortran
sudo apt install make

# CartPole
sudo apt install ffmpeg freeglut3-dev xvfb
pip install stable-baselines3[extra]
pip install opencv-python==4.5.5.64
pip install scipy
pip install sklearn
pip install tqdm
pip install scikit-learn

# SymbolicRegressor
pip install gplearn
pip install graphviz
pip install plotly

# Turbines
pip install openpyxl
pip install termcolor

# MuJoCo
sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install swig

pip install -r requirements.txt
pip install -r cleanrl/requirements/requirements-mujoco.txt
pip install -r cleanrl/requirements/requirements-mujoco_py.txt

# WindFLO
pip install f90nml
pip install git+https://github.com/CMA-ES/pycma.git@master
cd windflo/release/
make OS=LINUX MAIN=main
cd ../../


