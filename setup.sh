# mujoco
python3 -m venv venv
source venv/bin/activate
pip install -U pip

pip install -r requirements.txt
pip install -r cleanrl-master/requirements/requirements-mujoco.txt
pip install -r cleanrl-master/requirements/requirements-mujoco_py.txt


# windflo
pip install f90nml
pip install git+https://github.com/CMA-ES/pycma.git@master
cd windflo/release/
make OS=LINUX MAIN=main
cd ../../
