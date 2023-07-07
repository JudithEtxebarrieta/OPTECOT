'''
In this script, the proposed heuristics are applied to the MuJoCo's Swimmer environment. The CMA-ES 
algorithm is run on this environment using 100 different seeds for each heuristic. A database is built
with the relevant information obtained during the execution process. 

The general descriptions of the heuristics are:
HEURISTIC I: The accuracy is updated using the constant frequency calculated in experimentScripts_general/SampleSize_Frequency_bisection_method.py.
HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the last population is significantly different from the previous ones.
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
import garage.trainer
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.algos import CMAES
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy
from garage.trainer import TFTrainer,Trainer

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

import numpy as np
import pandas as pd
from dowel.logger import Logger
import warnings
import multiprocessing as mp
import psutil as ps
import scipy as sc
import random
import cma
import copy

#==================================================================================================
# NEW FUNCTIONS
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Functions for the search process of the optimal policy.
#--------------------------------------------------------------------------------------------------

@wrap_experiment(snapshot_mode="none", log_dir="/tmp/",archive_launch_repo=False)
def learn(ctxt=None, gymEnvName=None, action_space=None, max_episode_length=None,
                       policy_name=None,seed=None,heuristic=None,heuristic_param=None):
    
    '''Learning optimal policy with CMA-ES.'''

    global global_heuristic, global_heuristic_param,optimal_acc,train_env
    global_heuristic=heuristic
    global_heuristic_param=heuristic_param
    optimal_acc=1
    
    # Initialization of counters.
    global n_steps_proc,n_steps_acc,n_episodes,n_generations
    n_steps_proc=0 # Counter indicating the number of steps consumed so far for the procedure.
    n_steps_acc=0 # Counter indicating the number of steps consumed so far for accuracy adjustment.
    n_episodes = 0 # Counter that indicates which episode we are in.
    n_generations=0 # Counter that indicates which generation we are in.

    # Definition of parameters for the CMA-ES algorithm.
    global popsize,batch_size_ep,total_generations
    popsize= 20 # Population size in CMA-ES.
    batch_size_ep=1 # Number of episodes to be considered in evaluating each policy for a population.

    # Set seed.
    set_seed(seed)

    with TFTrainer(ctxt) as trainer:

        # Define training environment with the selected accuracy.
        current_max_episode_length=int(max_episode_length*optimal_acc)
        train_env = GymEnv(gymEnvName, max_episode_length=current_max_episode_length)
        train_env._env.unwrapped.model.opt.timestep=default_frametime/optimal_acc
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
        total_generations= int(max_steps/(popsize*batch_size_ep*current_max_episode_length)) # Number of populations to be evaluated, depending on the maximum number of steps defined for training (STOPPING CRITERION).
        trainer.train(n_epochs=total_generations, batch_size=batch_size_ep*current_max_episode_length)

#--------------------------------------------------------------------------------------------------
# Auxiliary functions to define the appropriate accuracy during the execution process.
#--------------------------------------------------------------------------------------------------
def spearman_corr(x,y):
    '''Calculation of Spearman's correlation between two sequences.'''
    return sc.stats.spearmanr(x,y)[0]

def from_rewards_to_ranking(list_rewards):
    '''Convert reward list to ranking list.'''
    list_pos_ranking=np.argsort(np.array(list_rewards))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

def evaluate_policy_test(agent,env,n_eval_episodes):
    '''Validate a policy.'''
    global n_generations

    # Set seed of the environment (so that the episodes to validate are the same 
    # for each call to the function) and define the initial state (obs) of the first episode. 
    env._env.seed(seed=test_env_seed)
    obs,_=env.reset()

    # Save reward for evaluated episodes.
    all_ep_reward=[]
    for _ in range(n_eval_episodes):
        # Evaluate episode with the policy and save the associated reward.
        episode_reward=0
        done=False
        while not done:
            action,_=agent.get_action(obs)
            env_step_info=env.step(action)
            episode_reward+=env_step_info.reward
            done=env_step_info.last
            obs=env_step_info.observation
        all_ep_reward.append(episode_reward)
        obs,_=env.reset()

    return np.mean(all_ep_reward)

def generation_reward_list(df_sample_policy_info,accuracy,count_time_acc=True):
    '''Generate a list of rewards associated with each policy that forms the sample population for the bisection method.'''

    global batch_size_ep,n_steps_acc
    
    # Obtain rewards associated with the policies that make up the sample population.
    list_rewards=list(df_sample_policy_info[df_sample_policy_info['accuracy']==accuracy]['reward'])

    # Update counters related to training time (steps).
    if count_time_acc:
        n_steps_acc+=len(list_rewards)*batch_size_ep*int(max_episode_length*accuracy)

    return list_rewards
#--------------------------------------------------------------------------------------------------
# Functions associated with the heuristics to be applied to adjust the accuracy.
#--------------------------------------------------------------------------------------------------
def possible_accuracy_values_bisection():
    '''Calculating the possible midpoints that can be evaluated in the bisection method.'''

    df_bisection=pd.read_csv('results/data/MuJoCo/UnderstandingAccuracy/df_Bisection.csv')
    interpolation_acc=list(df_bisection['accuracy'])
    interpolation_time=list(df_bisection['cost_per_eval'])
    lower=min(list(interpolation_time))
    upper=max(list(interpolation_time))

    list_bisection_acc=[]
    list_bisection_time=np.arange(lower,upper+(upper-lower)/(2**4),(upper-lower)/(2**4))[1:]# Four iterations of the bisection method are considered.
    for time in list_bisection_time:
        list_bisection_acc.append(np.interp(time,interpolation_time,interpolation_acc))
    
    return list_bisection_acc


def bisection_method(lower_time,upper_time,df_sample_policy_info,interpolation_pts,threshold=0.95):
    '''Adapted implementation of bisection method.'''

    # Initialize lower and upper limit.
    time0=lower_time
    time1=upper_time 

    # First midpoint.
    prev_m=lower_time
    m=(time0+time1)/2

    # Function to calculate the correlation between the rankings of random sample_size policies using the current and maximum accuracy.
    def similarity_between_current_best_acc(acc,df_sample_policy_info,first_iteration):

        # Save the rewards associated with each selected solution.
        best_rewards=generation_reward_list(df_sample_policy_info,1,count_time_acc=first_iteration)# with the maximum accuracy.
        new_rewards=generation_reward_list(df_sample_policy_info,acc)# With the new accuracy. 

        # Obtain lists of associated rankings.
        new_ranking=from_rewards_to_ranking(new_rewards)# New accuracy. 
        best_ranking=from_rewards_to_ranking(best_rewards)# Maximum accuracy. 

        # Compare both rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)

        return metric_value

    # Reset interval limits until the interval has a sufficiently small range.
    first_iteration=True
    stop_threshold=(time1-time0)*0.1# Equivalent to 4 iterations.
    while time1-time0>stop_threshold:
        metric_value=similarity_between_current_best_acc(np.interp(m,interpolation_pts[0],interpolation_pts[1]),df_sample_policy_info,first_iteration)
        if metric_value>=threshold:
            time1=m
        else:
            time0=m

        prev_m=m
        m=(time0+time1)/2
        
        first_iteration=False

    return np.interp(prev_m,interpolation_pts[0],interpolation_pts[1])


def execute_heuristic(gen,acc,df_sample_policy_info,list_accuracies,list_variances,heuristic,param):
    '''Running heuristics during the training process.'''

    global last_time_heuristic_accepted,unused_bisection_executions, stop_heuristic
    global n_steps_proc
    global n_steps_acc
    global max_steps
    global batch_size_ep
    global heuristic_accepted

    heuristic_accepted=False
    
    # For interpolation in the bisection method.
    df_interpolation=pd.read_csv('results/data/MuJoCo/UnderstandingAccuracy/df_Bisection.csv')
    df_bisection=pd.read_csv('results/data/general/ExtraCost_SavePopEvalCost/ExtraCost_SavePopEvalCost_MuJoCo.csv',float_precision='round_trip')
    interpolation_acc=list(df_interpolation['accuracy'])
    interpolation_time=list(df_interpolation['cost_per_eval'])
    lower_time=min(interpolation_time)
    upper_time=max(interpolation_time)
    interruption_threshold=float(max(df_bisection['opt_acc']))

    # HEURISTIC I: The accuracy is updated using a constant frequency.
    if heuristic=='I': 
        if gen==0:
            acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc],threshold=param)
            heuristic_accepted=True
        else:
            if (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq:
                acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc],threshold=param)
                heuristic_accepted=True

    # HEURISTIC II: The accuracy is updated when it is detected that the variance of the scores of the
    # last population is significantly different from the previous ones. In addition, when it is observed 
    # that in the last populations the optimum accuracy considered is equal to de maximum possible, the accuracy will 
    # no longer be adjusted and the maximum accuracy will be considered for the following populations.    
    if heuristic=='II': 
        if gen==0: 
            acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc])
            unused_bisection_executions=0
            heuristic_accepted=True
            
        else:
            if len(list_accuracies)>=param[1]:
                if stop_heuristic==False:
                    prev_acc=list_accuracies[-param[1]:]
                    prev_acc_high=np.array(prev_acc)==interruption_threshold
                    if sum(prev_acc_high)==param[1]:
                        stop_heuristic=True
                        acc=1

            if len(list_variances)>=param[0]+1 and stop_heuristic==False:
                # Calculate the confidence interval.
                variance_q05=np.mean(list_variances[(-1-param[0]):-1])-2*np.std(list_variances[(-1-param[0]):-1])
                variance_q95=np.mean(list_variances[(-1-param[0]):-1])+2*np.std(list_variances[(-1-param[0]):-1])

                last_variance=list_variances[-1]

                # Calculate the minimum accuracy with which the maximum quality is obtained.
                if last_variance<variance_q05 or last_variance>variance_q95:

                    if (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq:   
                        unused_bisection_executions+=int((n_steps_proc+n_steps_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                        acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc])
                        heuristic_accepted=True
                    else:
                        if unused_bisection_executions>0:
                            acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc])
                            unused_bisection_executions-=1
                            heuristic_accepted=True


    return acc

#==================================================================================================
# FUNCTIONS DESIGNED TO REPLACE SOME ALREADY EXISTING ONES
#==================================================================================================

def save_policy_train_bisection_info(self):
    '''
    This function is designed to replace the existing "rollout" function. It is modified to be 
    able to evaluate the policy sample of a population in the bisection method.
    '''

    # New code.
    global df_policy_train_bisection_info,list_idx_bisection,idx_sample,list_bisection_acc,n_sample
    if idx_sample in list_idx_bisection:
        n_sample+=1
        for accuracy in list_bisection_acc:
            self.start_episode()
            self._max_episode_length = int(max_episode_length*accuracy) # Set maximum episode size.
            policy_reward=0
            episode_steps=0
            while not self.step_episode(): # Until more steps can not be given in the episode.
                policy_reward+=self._env_steps[episode_steps].reward # Dar un nuevo step.
                episode_steps+= 1 # Add step.
            df_policy_train_bisection_info.append([n_sample,accuracy,policy_reward])

    # Default code.
    self.start_episode()
    while not self.step_episode():
        pass
    return self.collect_episode()

def rollout(self):
    '''The original function is modified to be able to store the data of interest during the execution process.'''

    global n_steps_proc,n_steps_acc,n_episodes,n_generations,df,test_n_eval_episodes,popsize,total_generations
    global seed,global_heuristic_param,optimal_acc
    global list_gen_policies,list_gen_policies_rewards,policy_reward_per_ep
    global sample_size,heuristic_accepted,last_time_heuristic_accepted

    # Lists to store the rewards of the policies associated with a population.
    n_policies=n_episodes/batch_size_ep # Policies evaluated so far.
    if n_policies%popsize==0:
        list_gen_policies=[]
        list_gen_policies_rewards=[]

        policy_reward_per_ep=[]

    # In case of using an environment with stopping criterion dependent on the terminate_when_unhealthy parameter, 
    # override this type of stop.
    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    # Initialize episode.
    self.start_episode() 
    self._max_episode_length = int(max_episode_length*optimal_acc) # Set maximum episode size.

    # Update counters after evaluating the episode.
    episode_steps = 0 # Initialize counter to add the number of steps taken in this episode.
    policy_reward=0
    while not self.step_episode(): # Until more steps can not be given in the episode.
        policy_reward+=self._env_steps[episode_steps].reward # Take a new step.
        episode_steps+= 1 # Add step.

    n_steps_proc += episode_steps # Total steps taken so far.
    n_episodes += 1 # Episodes evaluated so far.
    policy_reward_per_ep.append(policy_reward)

    # When a complete policy has been evaluated (number of episodes evaluated=batch_size_ep), 
    # its corresponding reward is stored (total reward in batch_size_ep).
    if n_episodes%batch_size_ep==0:
        list_gen_policies.append(self.agent)
        list_gen_policies_rewards.append(np.mean(policy_reward_per_ep))
        policy_reward_per_ep=[]
    
    # When enough policies have been evaluated to be able to complete a new population, the
    # information obtained from the new population is saved.    
    n_policies=n_episodes/batch_size_ep # Policies evaluated so far.
    if n_policies!=0 and n_policies%popsize == 0:

        # Subtract number of steps added in duplicate (in the last iteration of the bisection 
        # method we have already evaluated sample_size policies that form the population, and 
        # the given steps have now been re-counted in n_steps_proc).        
        if heuristic_accepted:
            n_steps_acc-=sample_size*batch_size_ep*int(max_episode_length*optimal_acc)
            last_time_heuristic_accepted=n_steps_proc+n_steps_acc

        # Update database.
        n_generations+=1
        best_policy=list_gen_policies[list_gen_policies_rewards.index(max(list_gen_policies_rewards))]
        reward=evaluate_policy_test(best_policy,test_env,test_n_eval_episodes)
        df.append([global_heuristic_param,seed,n_generations,reward,optimal_acc,np.var(list_gen_policies_rewards),heuristic_accepted,n_steps_proc,n_steps_acc,n_steps_proc+n_steps_acc])
        
    return self.collect_episode()

def train(self, trainer):
    '''The original function is modified so that heuristics can be applied during the execution process.'''

    init_mean = self.policy.get_param_values()
    self._es = cma.CMAEvolutionStrategy(init_mean, self._sigma0,
                                        {'popsize': self._n_samples})
    self._all_params = self._sample_params()
    self._cur_params = self._all_params[0]
    self.policy.set_param_values(self._cur_params)
    self._all_returns = []

    # start actual training
    last_return = None

    global n_generations,df,global_heuristic,global_heuristic_param,optimal_acc,batch_size_ep
    global stop_heuristic
    global n_steps_proc,n_steps_acc,train_env,last_time_heuristic_accepted,unused_bisection_executions
    while n_steps_proc+n_steps_acc<max_steps:# MODIFICATION: Change stopping criterion.
        # MODIFICATION: To obtain individuals from the population.
        global df_policy_train_bisection_info
        def obtain_population():
            global seed,idx_sample,df_policy_train_bisection_info,list_idx_bisection,list_bisection_acc,n_sample

            garage.sampler.default_worker.DefaultWorker.rollout = save_policy_train_bisection_info

            df_policy_train_bisection_info=[]
            list_bisection_acc=possible_accuracy_values_bisection()
            n_sample=0

            random.seed(seed)
            list_idx_bisection=random.sample(range(self._n_samples),sample_size)


            trainer_pop=trainer
            trainer_pop.step_itr = trainer._stats.total_itr
            trainer_pop.step_episode = None
            for idx_sample in range(self._n_samples):

                trainer_pop.step_episode = trainer_pop.obtain_episodes(trainer_pop.step_itr)
                last_return = trainer_pop._algo._train_once(trainer_pop.step_itr,trainer_pop.step_episode)
                trainer_pop.step_itr += 1

        if n_generations==0:
            obtain_population()
        else:
            if global_heuristic=='I' and (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq: 
                obtain_population()
            elif global_heuristic=='II':
                df_seed=pd.DataFrame(df)
                df_seed=df_seed[df_seed[1]==seed]
                list_variances=list(df_seed[5])

                # Calculate the confidence interval.
                variance_q05=np.mean(list_variances[(-1-global_heuristic_param[0]):-1])-2*np.std(list_variances[(-1-global_heuristic_param[0]):-1])
                variance_q95=np.mean(list_variances[(-1-global_heuristic_param[0]):-1])+2*np.std(list_variances[(-1-global_heuristic_param[0]):-1])
                
                last_variance=list_variances[-1]

                if (len(list_variances)>=global_heuristic_param[0]+1) and (last_variance<variance_q05 or last_variance>variance_q95):
                    if (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq: 
                        obtain_population()
                    elif unused_bisection_executions>0:
                        obtain_population()

        # MODIFICATION: apply the heuristic.
        df_policy_train_bisection_info=pd.DataFrame(df_policy_train_bisection_info,columns=['n_sample','accuracy','reward'])

        if n_generations==0:
            optimal_acc=execute_heuristic(n_generations,optimal_acc,df_policy_train_bisection_info,[],[],global_heuristic,global_heuristic_param)
            stop_heuristic=False
        else:
            df_seed=pd.DataFrame(df)
            df_seed=df_seed[df_seed[1]==seed]
            optimal_acc=execute_heuristic(n_generations,optimal_acc,df_policy_train_bisection_info,list(df_seed[4]),list(df_seed[5]),global_heuristic,global_heuristic_param)
        
        train_env = GymEnv(gymEnvName, max_episode_length=int(max_episode_length*optimal_acc))
        train_env._env.unwrapped.model.opt.timestep=default_frametime/optimal_acc
        train_env._env.seed(seed=train_env_seed)
        trainer._env=train_env   
        trainer._train_args.batch_size=batch_size_ep*int(max_episode_length*optimal_acc)

        # MODIFICATION: to evaluate the population.
        garage.sampler.default_worker.DefaultWorker.rollout = rollout
        trainer.step_itr = trainer._stats.total_itr
        trainer.step_episode = None
        for _ in range(self._n_samples):
            trainer.step_episode = trainer.obtain_episodes(trainer.step_itr)
            last_return = self._train_once(trainer.step_itr,trainer.step_episode)
            trainer.step_itr += 1

    return last_return


def start_episode(self):
    '''
    The original function is modified to be able to evaluate each policy of each population during
    training with the same set of episodes, this modification makes the process deterministic and the 
    comparisons of individuals per population are fair.
    '''

    self._eps_length = 0

    # MODIFICATION: so that the same episodes are always used to validate the policies and these can be comparable.
    global n_episodes,batch_size_ep,n_generations
    if n_episodes%batch_size_ep==0:
        self.env._env.seed(seed=n_generations)#seed=train_env_seed (if the same episodes are to be evaluated in all populations).
    self._prev_obs, episode_info = self.env.reset()

    for k, v in episode_info.items():
        self._episode_infos[k].append(v)

    self.agent.reset()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

# Modification of existing function (not to waste time).
garage.trainer.Trainer.save = lambda self, epoch: "skipp save."
# Modification of existing function (not to print episode information during execution).
Logger.log= lambda self, data: 'skipp info message.'
warnings.filterwarnings("ignore")
# Modification of existing functions (to be able to apply heuristics during execution).
CMAES.train=train
#Modification of existing function (to evaluate all the policies of each population with the same episodes so that they can be comparable).
from garage.sampler.default_worker import DefaultWorker
DefaultWorker.start_episode=start_episode

# Environment features.
gymEnvName='Swimmer-v3'
action_space="continuous"
max_episode_length=1000
default_frametime=0.01 # Parameter from which the accuracy will be modified.
policy_name='SwimmerPolicy'
DTU=False # Stopping criterion dependent on terminate_when_unhealthy.

# Grids and parameters for training.
list_train_seeds = list(range(2,102,1)) # List of training seeds.
train_env_seed=0 # Seed for the training environment.
max_steps=1000000 # Training limit measured in steps (assuming that the cost of all steps is the same). (Reference: https://huggingface.co/sb3/ppo-Swimmer-v3/tree/main)

# Validation parameters.
test_env_seed=1 # Seed for the validation environment.
test_n_eval_episodes=10 # Number of episodes to be evaluated in the validation.

# Argument list for parallel processing.
list_arg=[['II',[5,3]],['II',[10,3]],['I',0.8],['I',0.95]]
df_sample_freq=pd.read_csv('results/data/general/SampleSize_Frequency_bisection_method.csv',index_col=0)
sample_size=int(df_sample_freq[df_sample_freq['env_name']=='MuJoCo']['sample_size'])
heuristic_freq=float(df_sample_freq[df_sample_freq['env_name']=='MuJoCo']['frequency_time'])

# Function for parallel execution.
def parallel_processing(arg):

    heuristic=arg[0]
    heuristic_param=arg[1]

    global df
    df=[]

    global seed

    for seed in tqdm(list_train_seeds):
        learn(gymEnvName=gymEnvName, action_space=action_space, max_episode_length=max_episode_length,policy_name=policy_name,seed=seed,heuristic=heuristic,heuristic_param=heuristic_param) 

    # Save database.
    df=pd.DataFrame(df,columns=['heuristic_param','seed','n_gen','reward','accuracy','variance','update','n_steps_proc','n_steps_acc','n_steps'])
    df.to_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(heuristic_param)+'.csv')

# Parallel processing.
phisical_cpu=ps.cpu_count(logical=False)
pool=mp.Pool(phisical_cpu)
pool.map(parallel_processing,list_arg)
pool.close()

# Join databases.
def join_df(heuristic,list_param):
    df=pd.read_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(param)+'.csv', index_col=0)
    for param in list_param[1:]:
        df_new=pd.read_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(param)+'.csv', index_col=0)
        df=pd.concat([df,df_new],ignore_index=True)

    df.to_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'.csv')

join_df('I',[0.8,0.95])
join_df('II',['[5, 3]','[10, 3]'])


