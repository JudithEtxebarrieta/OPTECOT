from argparse import ArgumentError
from statistics import median
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import subprocess
import time
import re
from os.path import exists
import sys
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse


import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'Other_RL'))

from other_RL.meta_world_and_garage.test_example_garage_cart_pole_CMA_ES import launch_from_python
# run 'chmod -R u+rw ~/Dropbox' to refresh dropbox


gens = 200
seeds = list(range(2,32))
parallel_threads = 1



method = "constant"


# DTU = DisableTerminateUnhealthy
gymEnvName_list =         ['CartPole-v1',  'Pendulum-v1',  'HalfCheetah-v3',  'InvertedDoublePendulum-v2',  'Swimmer-v3', 'Hopper-v3' , 'Ant-v3'    , 'Walker2d-v3', 'Hopper-v3_DTU' , 'Ant-v3_DTU'    , 'Walker2d-v3_DTU']
is_reward_monotone_list = [True         ,  True         ,  False           ,  False                      ,   False      , False       ,  False      , False        , False           ,  False          , False            ]
action_space_list =       ["discrete"   ,   "continuous",  "continuous"    ,  "continuous"               ,  "continuous", "continuous", "continuous", "continuous" , "continuous"    , "continuous"    , "continuous"     ]
max_episode_length_list = [          400,            200,  1000            ,                         1000,  1000        ,         1000,    1000     ,  1000        ,         1000    ,    1000         ,  1000            ]




for index, gymEnvName, action_space, max_episode_length, is_reward_monotone in zip(range(len(gymEnvName_list)), gymEnvName_list, action_space_list, max_episode_length_list,  is_reward_monotone_list):

    gracetime = round(max_episode_length * 0.2)


    for seed in seeds:
        res_filepath = f"results/data/garage_gym/gymEnvName_{gymEnvName}_{method}_{seed}.txt"
        launch_from_python(seed, method, gymEnvName, action_space, gracetime, gens, max_episode_length, res_filepath) 


    



    









