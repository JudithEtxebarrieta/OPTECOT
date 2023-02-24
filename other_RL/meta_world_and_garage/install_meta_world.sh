#!/bin/bash

# install mujoco
cd ~/Downloads
wget -nc -c  wget -c 	https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz  -O mujoco210.tar.gz
tar -x -k -f mujoco210.tar.gz
mkdir -p ~/.mujoco
rsync -vraR mujoco210 ~/.mujoco/






# install apt dependencies
sudo apt update
sudo apt install patchelf
sudo apt install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
sudo apt install -y libglew1.5-dev
sudo apt install libxslt1-dev zlib1g-dev libglew-dev python3-pip
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so


# install python dependencies
cd ~/Downloads
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip install gym==0.24

reset


# install mujoco-py
pip3 install 'mujoco-py<2.2,>=2.1'

# Test mujoco-py
echo
echo "--------------------------------"

echo "OPTIONAL: to test mujoco-py, quit this script and run: 

python3
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)

sim.step()
print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06 ...]"
echo "--------------------------------"



# install metaworld
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld



# install garage
pip install TensorFlow==2.9
pip install gym==0.21.0
pip install garage



echo "---------------"
echo  "Add variables to path. Execute only once."
echo 
echo 'export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
echo 'export PATH="$LD_LIBRARY_PATH:$PATH"' >> ~/.bashrc
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc

echo
echo "RESTART PC"