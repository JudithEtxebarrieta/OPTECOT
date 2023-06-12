'''
This script saves the relevant information extracted from the execution process of the PPO 
algorithm on the CartPole environment, using 10 different time-step values and considering 
a total of 30 seeds for each one. 

Based on:
https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
# Special libraries for the environment.
import stable_baselines3 # Library used to create an RL model, train it and evaluate it.
import gym # Stable-Baselines works in environments that follow the gym interface.
from stable_baselines3 import PPO # Import the RL model.
from stable_baselines3.ppo import MlpPolicy # Import the type of policy to be used to create the networks.
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from sklearn.neighbors import KernelDensity

# Generic libraries.
import numpy as np
from scipy.stats import norm
import time
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import pandas as pd
import multiprocessing as mp
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union


#==================================================================================================
# CLASSES
#==================================================================================================

class stopwatch:
    '''
    Defines the methods necessary to measure time during execution process.
    '''
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t

#==================================================================================================
# NEW FUNCTIONS
#==================================================================================================

def evaluate(model,eval_env,eval_seed,n_eval_episodes):
    '''
    The current policy is evaluated using the episodes of the validation environment.

    Parameters
    ==========
    model: Policy to be evaluated.
    eval_env : Validation environment.
    init_obs : Initial state of the first episode of the validation environment.
    seed (int): Seed of the validation environment.
    n_eval_episodes (int): Number of episodes (evaluations) in which the model will be evaluated.

    Returns
    =======
    Average of the rewards obtained in the n_eval_episodes episodes.
    '''
    # To save the reward per episode.
    all_episode_reward=[]

    # To ensure that the same episodes are used in each call to the function.
    eval_env.seed(eval_seed)
    obs=eval_env.reset()
    
    for i in range(n_eval_episodes):

        episode_rewards = 0
        done = False # Parameter that indicates after each action if the episode continues (False) or is finished (True).
        while not done:
            action, _states = model.predict(obs, deterministic=True) # The action to be taken with the model is predicted.         
            obs, reward, done, info = eval_env.step(action) # Action is applied in the environment.
            episode_rewards+=reward # The reward is saved.

        # Save total episode reward.
        all_episode_reward.append(episode_rewards)

        # Reset the episode.
        obs = eval_env.reset() 
    
    return np.mean(all_episode_reward)


#==================================================================================================
# MODIFIED EXISTING FUNCTIONS
#==================================================================================================

def callback_in_each_iteration(self, num_timesteps: int, total_timesteps: int) -> None:
    '''
    This function is the adapted version of the existing "_update_current_progress_remaining" function. 
    It is modified to be able to evaluate the solution (policy) during the training process and to collect 
    relevant information (number of steps, number of episodes, model quality measured in reward, 
    computational time spent, seed,...).
    '''

    # Pause time during validation.
    sw.pause() 

    # Extract relevant information.
    mean_reward = evaluate(model,eval_env,eval_seed,n_eval_episodes)
    info=pd.DataFrame(model.ep_info_buffer)
    info_steps=sum(info['r'])
    info_time=sum(info['t'])
    n_eval=len(info)
    max_step_per_eval=max(info['r'])

    # Save the extracted information.
    df_train_acc.append([num_timesteps, info_steps,model.seed,n_eval,max_step_per_eval,sw.get_time(),info_time,mean_reward])

    # Resume time.
    sw.resume()

    # This line is used by the function we are replacing. Do not change this line.
    self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps) 

def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        This function is the adapted version of the existing "_setup_learn" function. It is modified so 
        that the training limit is the defined number of training steps and not the default maximum 
        number of evaluations (maxlen).

        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=max_train_steps)#MODIFICATION(previously:maxlen=100)
            self.ep_success_buffer = deque(maxlen=max_train_steps)#MODIFICATION (previously:maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback


#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# To use the modified callback function.
import stable_baselines3.common.base_class
stable_baselines3.common.base_class.BaseAlgorithm._update_current_progress_remaining = callback_in_each_iteration
# To use the modified _setup_learn function function.
from stable_baselines3.common.base_class import *
BaseAlgorithm._setup_learn=_setup_learn
    
# Training environment and parameters.
train_env = gym.make('CartPole-v1')
max_train_steps=10000

# Validation environment and parameters.
eval_env = gym.make('CartPole-v1')
eval_seed=0
n_eval_episodes=100

# Default parameters.
default_tau = 0.02
default_max_episode_steps = 500

# Grids.
grid_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1] # Time-step accuracy.
grid_seed=range(1,31,1)

# Function for parallel execution.
def parallel_processing(accuracy):
    # Update parameters of the training environment.
    train_env.env.tau = default_tau / accuracy
    train_env.env.spec.max_episode_steps = int(default_max_episode_steps*accuracy)
    train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps
    
    # Save in a database the information of the training process for the selected accuracy.
    global df_train_acc,seed
    df_train_acc=[]

    for seed in tqdm(grid_seed):
        # Start counting time.
        global sw
        sw = stopwatch()
        sw.reset()

        # Algorithm execution.
        global model
        model = PPO(MlpPolicy,train_env,seed=seed, verbose=0,n_steps=train_env._max_episode_steps)
        model.set_random_seed(seed)
        model.learn(total_timesteps=max_train_steps)

    df_train_acc=pd.DataFrame(df_train_acc,columns=['steps','info_steps','seed','n_eval','max_step_per_eval','time','info_time','mean_reward'])
    df_train_acc.to_csv('results/data/CartPole/ConstantAccuracyAnalysis/df_train_acc'+str(accuracy)+'.csv')

# Parallel processing.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,grid_acc)
pool.close()

# Save the accuracy list.
np.save('results/data/CartPole/ConstantAccuracyAnalysis/grid_acc',grid_acc)

# Save execution time limit.
np.save('results/data/CartPole/ConstantAccuracyAnalysis/max_train_steps',max_train_steps)




