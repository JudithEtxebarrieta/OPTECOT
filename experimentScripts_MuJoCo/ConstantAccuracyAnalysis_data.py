
'''
This script applies the CMA-ES algorithm on the MuJoCo's Swimmer environment, during a maximum number 
of steps, considering 10 different values of the time-step parameter and 100 seeds for each parameter
value. For each time-step value, a database with the relevant information of the execution process is built.

Based on:
https://github.com/rlworkgroup/garage/blob/master/src/garage/examples/np/cma_es_cartpole.py
https://huggingface.co/sb3/ppo-Swimmer-v3/tree/main

'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'cleanrl'))

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

from garage import wrap_experiment
import garage.sampler.default_worker
from garage.sampler.default_worker import *
from garage._functions import *
import numpy as np


import garage.trainer
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.algos import CMAES
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy
from garage.trainer import TFTrainer

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

import pandas as pd
from dowel.logger import Logger
import warnings
import multiprocessing as mp
import psutil as ps

#==================================================================================================
# NEW FUNCTIONS
#==================================================================================================

def evaluate_policy(agent,test_env,n_eval_episodes):
    '''Validate a policy.'''

    # Set environment seed (so that the episodes to be validated are the same for each call to the
    # function) and define the initial state (obs) of the first episode.
    test_env._env.seed(seed=test_env_seed)
    obs,_=test_env.reset()

    # Save reward per evaluated episode.
    all_ep_reward=[]
    for i in range(n_eval_episodes):
        # Evaluate episode with the policy and save the associated reward.
        episode_reward=0
        done=False
        while not done:
            action,_=agent.get_action(obs)
            env_step_info=test_env.step(action)
            episode_reward+=env_step_info.reward
            done=env_step_info.last
            obs=env_step_info.observation
        all_ep_reward.append(episode_reward)
        obs,_=test_env.reset()

    return np.mean(all_ep_reward)

@wrap_experiment(snapshot_mode="none", log_dir="/tmp/",archive_launch_repo=False)
def learn(ctxt=None, gymEnvName=None, action_space=None, max_episode_length=None,
                       policy_name=None,seed=None,accuracy=1.0):
    
    '''Learning the optimal policy using CMA-ES.'''

    # Initialization of counters.
    global n_steps,n_episodes,n_generations
    n_steps=0 # Counter indicating the number of steps consumed so far.
    n_episodes = 0 # Counter that indicates which episode we are in.
    n_generations=0 # Counter that indicates which generation we are in.

    # Definition of parameters for the CMA-ES algorithm.
    global popsize,batch_size_ep,total_generations
    popsize= 20 # Size of populations in CMA-ES. (Reference: https://github.com/rlworkgroup/garage/blob/master/src/garage/examples/np/cma_es_cartpole.py)
    batch_size_ep=1 # Number of episodes to be considered in evaluating each generation policy.

    # Set seed.
    set_seed(seed)

    with TFTrainer(ctxt) as trainer:

        # Define training environment with the selected accuracy.
        global current_max_episode_length
        current_max_episode_length=max_episode_length*accuracy
        train_env = GymEnv(gymEnvName, max_episode_length=current_max_episode_length)
        train_env._env.unwrapped.model.opt.timestep=default_frametime/accuracy
        train_env._env.seed(seed=train_env_seed)

        # Define validation environment with maximum accuracy.
        global test_env
        test_env = GymEnv(gymEnvName, max_episode_length=max_episode_length)
        test_env._env.unwrapped.model.opt.timestep=default_frametime
        
        # Define type of policy.
        is_action_space_discrete=bool(["continuous", "discrete"].index(action_space))
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name=policy_name, env_spec=train_env.spec)
        else:
            policy = ContinuousMLPPolicy(name=policy_name, env_spec=train_env.spec)
        sampler = LocalSampler(agents=policy, envs=train_env, max_episode_length=train_env.spec.max_episode_length, is_tf_worker=True)

        # Initialize CMA-ES algorithm.
        algo= CMAES(env_spec=train_env.spec, policy=policy, sampler=sampler, n_samples=popsize)
        trainer.setup(algo, train_env)

        # Execution process (policy training).
        total_generations= int(max_steps/(popsize*batch_size_ep*current_max_episode_length)) # Number of generations to be evaluated, depending on the maximum number of steps defined for training (STOPPING CRITERION).
        trainer.train(n_epochs=total_generations, batch_size=batch_size_ep*current_max_episode_length)


#==================================================================================================
# FUNCTIONS DESIGNED TO REPLACE SOME EXISTING ONES
#==================================================================================================
def start_episode(self):
    """
    To be able to evaluate each policy of each population during training with the same set of episodes,
    the original function is modified to make the process deterministic and comparisons between individuals 
    per population fair.
    
    Begin a new episode.
    """

    self._eps_length = 0

    # MODIFICATION: so that the same episodes are always used to validate the policies and these can be comparable.
    global n_episodes,batch_size_ep,n_generations
    if n_episodes%batch_size_ep==0:
        self.env._env.seed(seed=n_generations)#seed=train_env_seed (if the same episodes are to be evaluated in all generations).
    self._prev_obs, episode_info = self.env.reset()

    for k, v in episode_info.items():
        self._episode_infos[k].append(v)

    self.agent.reset()

def rollout(self):
    '''The original function is modified to be able to store data of interest during training.'''

    global n_steps,n_episodes,n_generations,df_acc,current_max_episode_length,n_eval_episodes,popsize,total_generations
    global seed,accuracy
    global list_gen_policies,list_gen_policies_rewards,policy_reward_per_ep

    # Lists to store the policies associated with a population and their corresponding rewards.
    n_policies=n_episodes/batch_size_ep # Evaluated policies so far.
    if n_policies%popsize==0:

        list_gen_policies=[]
        list_gen_policies_rewards=[]

        policy_reward_per_ep=[]

    # In case of using an environment with stop criteria dependent on the terminate_when_unhealthy parameter, override this type of stop.
    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    # Initialize episode.
    self.start_episode() 
    self._max_episode_length = current_max_episode_length # Set maximum episode size.
    
    # Update counters after evaluating the episode.
    episode_steps = 0 # Initialize counter to add the number of steps taken in this episode.
    policy_reward=0
    while not self.step_episode(): # Until more steps can not be given in the episode.
        policy_reward+=self._env_steps[episode_steps].reward # Take a new step.
        episode_steps+= 1 # Add step.

    n_steps += episode_steps # Total steps taken so far.
    n_episodes += 1 # Evaluated episodes so far.
    policy_reward_per_ep.append(policy_reward)

    # When a complete policy has been evaluated (number of episodes evaluated=batch_size_ep), 
    # the policy and its corresponding reward (total reward in batch_size_ep) are stored.
    if n_episodes%batch_size_ep==0:
        list_gen_policies.append(self.agent)
        list_gen_policies_rewards.append(np.mean(policy_reward_per_ep))
        policy_reward_per_ep=[]
    
    # When enough policies have been evaluated to be able to complete a new population, 
    # the information obtained from the new population is saved.
    n_policies=n_episodes/batch_size_ep # Policies evaluated so far.
    if n_policies!=0 and n_policies%popsize == 0:
        best_policy=list_gen_policies[list_gen_policies_rewards.index(max(list_gen_policies_rewards))]
        reward=evaluate_policy(best_policy,test_env,n_eval_episodes)
        df_acc.append([accuracy,seed,n_generations,n_episodes,n_steps,reward])
        n_generations+=1

    return self.collect_episode()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Modification of existing function (to save data during training).
garage.sampler.default_worker.DefaultWorker.rollout = rollout
# Modification of existing function (not to waste time).
garage.trainer.Trainer.save = lambda self, epoch: "skipp save."
# Modification of existing function (not to print episode information during training).
Logger.log= lambda self, data: 'skipp info message.'
warnings.filterwarnings("ignore")
# Modification of existing function (to evaluate all the policies of each population with the same episodes so that they can be comparable).
from garage.sampler.default_worker import DefaultWorker
DefaultWorker.start_episode=start_episode

# Environment features.
gymEnvName='Swimmer-v3'
action_space="continuous"
max_episode_length=1000
default_frametime=0.01 # Parameter from which the accuracy is modified.
policy_name='SwimmerPolicy'
DTU=False # Stopping criterion dependent on terminate_when_unhealthy.

# Grids and parameters for training.
list_train_seeds = list(range(2,102,1)) # list of training seeds.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
train_env_seed=0 # Seed for the training environment.
max_steps=1000000 # Training limit measured in steps (assuming that the cost of all steps is the same). (Reference: https://huggingface.co/sb3/ppo-Swimmer-v3/tree/main)

# Validation parameters.
test_env_seed=1 # Seed for the validation environment.
n_eval_episodes=10 # Number of episodes to be evaluated in the validation.

# Function for parallel execution.
def parallel_processing(arg):
    global df_acc
    df_acc=[]

    global seed,accuracy
    accuracy=arg
    for seed in tqdm(list_train_seeds):
        learn(gymEnvName=gymEnvName, action_space=action_space, max_episode_length=max_episode_length,policy_name=policy_name,seed=seed,accuracy=accuracy) 

    # Save database.
    df_acc=pd.DataFrame(df_acc,columns=['accuracy','train_seed','n_gen','n_ep','n_steps','reward'])
    df_acc.to_csv('results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc'+str(accuracy)+'.csv')

# Parallel processing.
phisical_cpu=ps.cpu_count(logical=True)
pool=mp.Pool(phisical_cpu)
pool.map(parallel_processing,list_acc)
pool.close()

# Save accuracy list.
np.save('results/data/MuJoCo/ConstantAccuracyAnalysis/list_acc',list_acc)

# Save runtime limit.
np.save('results/data/MuJoCo/ConstantAccuracyAnalysis/max_steps',max_steps)








