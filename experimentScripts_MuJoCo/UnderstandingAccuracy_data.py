'''
This script evaluates 100 random policies (associated with a maximum accuracy) on 10 episodes of 
a defined environment with 10 different accuracy values for the time-step parameter. The relevant
data (average rewards and number of steps per evaluation) is stored for later access.
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import os
import sys
from shutil import rmtree
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import numpy as np
from garage.envs import GymEnv
import pandas as pd
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy
from garage.tf.policies.policy import Policy
from garage import wrap_experiment
from garage.trainer import TFTrainer
from tqdm import tqdm as tqdm
from garage.experiment.deterministic import set_seed
from garage.np.algos import CMAES
from garage.sampler import LocalSampler
import garage.sampler.default_worker
from garage.sampler.default_worker import *
import dowel.logger
from dowel.logger import *

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

import multiprocessing as mp
import psutil as ps


#==================================================================================================
# FUNCTIONS
#==================================================================================================

def rollout(self):
    '''The original function is modified to be able to store the data of interest during the execution process.'''

    self.start_episode()
    while not self.step_episode():
        pass

    # MODIFICATION: every time this function is called we will be working with a new policy, 
    # therefore this is the moment when we must evaluate the policy and save the information of interest.
    global n_policy,global_n_sample
    print('Policy: '+str(n_policy)+str('/')+str(global_n_sample))
    for accuracy in tqdm(list_acc):
        reward,steps=evaluate_policy(self.agent,accuracy)
        df.append([accuracy,n_policy,reward,steps])
    n_policy+=1

    return self.collect_episode()

@wrap_experiment
def evaluate_policy_sample(ctxt=None,n_sample=100):
    '''To evaluate a random sample of policies, considering individuals from a single population of the CMA-ES algorithm.'''

    global global_n_sample
    global_n_sample=n_sample

    with TFTrainer(ctxt) as trainer:

        set_seed(0)

        # Initialize the environment in which the policies are defined (with maximum accuracy).
        env = GymEnv(gymEnvName, max_episode_length=max_episode_length)
        env._env.seed(seed=0)

        # In case of using an environment with stop criteria dependent on the terminate_when_unhealthy
        # parameter, override this type of stop.        
        if DTU:
            env._env.env._terminate_when_unhealthy = False 

        # Evaluate a random sample of 100 policies with the selected accuracies.
        global n_policy
        n_policy=1
        is_action_space_discrete=bool(["continuous", "discrete"].index(action_space))
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name=policy_name, env_spec=env.spec)
        else:
            policy = ContinuousMLPPolicy(name=policy_name, env_spec=env.spec)

        sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True)
        algo= CMAES(env_spec=env.spec, policy=policy, sampler=sampler, n_samples=n_sample)
        trainer.setup(algo, env)
        trainer.train(n_epochs=1, batch_size=env.spec.max_episode_length) # Esta funcion llamara internamente a "rollout".


def evaluate_policy(policy,accuracy):
    '''Evaluate a single policy.'''

    # Initialize validation environment by setting the appropriate accuracy value.
    eval_env = GymEnv(gymEnvName, max_episode_length=max_episode_length*accuracy)
    eval_env._env.unwrapped.model.opt.timestep=default_frametime/accuracy

    # Set seed so that the episodes to be evaluated are the same for each function call 
    # and define the initial state (obs) of the first episode.
    eval_env._env.seed(seed=0)
    obs,_=eval_env.reset()

    # Save reward for evaluated episodes.
    all_ep_reward=[]
    all_ep_steps=[]
    steps=0
    for _ in range(10):
        # Evaluate episode with the policy and save the associated reward.
        episode_reward=0
        episode_steps=0
        done=False
        while not done:
            action,_=policy.get_action(obs)
            env_step_info=eval_env.step(action)
            episode_reward+=env_step_info.reward
            done=env_step_info.last
            obs=env_step_info.observation
            episode_steps+=1
        all_ep_reward.append(episode_reward)
        all_ep_steps.append(episode_steps)
        obs,_=eval_env.reset()

    reward=np.mean(all_ep_reward)
    steps=np.mean(all_ep_steps)
    return reward,steps
 
#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Modification of existing function (to save data during training).
garage.sampler.default_worker.DefaultWorker.rollout = rollout
# Modification of existing function (not to waste time).
garage.trainer.Trainer.save = lambda self, epoch: "skipp save."
# Modification of existing function (not to print episode information during execution process).
Logger.log= lambda self, data: 'skipp info message.'
warnings.filterwarnings("ignore")

# Environment characteristics (to run the script with another MuJoCo environment these are the only variables to modify).
gymEnvName='Swimmer-v3'
action_space="continuous"
max_episode_length=1000
default_frametime=0.01 # Parameter from which the accuracy will be modified.
policy_name='SwimmerPolicy'
DTU=False # Stop criteria dependent on terminate_when_unhealthy.

#--------------------------------------------------------------------------------------------------
# For motivation analysis.
#--------------------------------------------------------------------------------------------------

# List of accuracies with which the previous sample will be evaluated.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# Evaluate a random sample of 100 policies using different values of accuracy and build a database with 
# the scores (rewards) and execution times (steps) per evaluation.
df=[]
evaluate_policy_sample(n_sample=100)

# Save database.
df_motivation=pd.DataFrame(df,columns=['accuracy','n_policy','reward','steps'])
df_motivation.to_csv('results/data/MuJoCo/UnderstandingAccuracy/df_UnderstandingAccuracy.csv')


#--------------------------------------------------------------------------------------------------
# For the definition of the values (time) on which the bisection will be applied.
#--------------------------------------------------------------------------------------------------
# Save database.
df_bisection=pd.DataFrame(df,columns=['accuracy','n_policy','reward','cost_per_eval'])
df_bisection=df_bisection[['accuracy','cost_per_eval']]
df_bisection=df_bisection.groupby('accuracy').mean()
df_bisection.to_csv('results/data/MuJoCo/UnderstandingAccuracy/df_Bisection.csv')

# Delete auxiliary files.
sys.path.append('data')
rmtree('data')

