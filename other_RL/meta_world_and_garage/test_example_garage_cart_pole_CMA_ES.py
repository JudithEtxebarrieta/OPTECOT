#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.
Here it runs CartPole-v1 environment with 100 epoches.
Results:
    AverageReturn: 100
    RiseTime: epoch 38 (itr 760),
              but regression is observed in the course of training.
"""
from garage import wrap_experiment
import garage.sampler.default_worker
from garage.sampler.default_worker import *
from garage._functions import *
import time
import argparse
import sys
import numpy as np

TOTAL_COMPUTED_STEPS = 0 # number of episodes (1 episode is equal to a complete begining -> [environment-> action -> reward] -> end cycle)
RUNTIMES = []
START_REF_TIME = None
POPSIZE = 100  # CMA-ES Population size.

DTU = False

batch_size = 1
gymEnvName=None
res_filepath="demo.txt"

EPISODE_INDEX = 0
SEED = None

def rollout(self):
    print("Begin custom rollout")
    
    global RUNTIMES
    global START_REF_TIME
    global TOTAL_COMPUTED_STEPS
    global EPISODE_INDEX

    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    if START_REF_TIME is None:
        START_REF_TIME = time.time()


    sum_of_rewards = 0
    episode_start_ref_t = time.time()
    self.start_episode()
    i = -1
    self._max_episode_length = MAX_EPISODE_LENGTH
    was_early_stopped = False
    while not self.step_episode():
        i += 1
        step_reward = self._env_steps[i-1].reward
        sum_of_rewards = sum_of_rewards + step_reward

    self._max_episode_length = MAX_EPISODE_LENGTH       


    print("f =",sum_of_rewards)



    TOTAL_COMPUTED_STEPS += i

    RUNTIMES.append(time.time() - episode_start_ref_t)

    if EPISODE_INDEX%POPSIZE == POPSIZE-1:
        runtimes = "("+";".join(map(str, RUNTIMES))+")"
        with open(res_filepath, "a+") as f:
            print("seed_"+str(SEED)+"_gymEnvName_"+gymEnvName, time.time() - START_REF_TIME, TOTAL_COMPUTED_STEPS, EPISODE_INDEX, runtimes , file=f, sep=",", end="\n")
        RUNTIMES = []


    EPISODE_INDEX += 1
    return self.collect_episode()

# mock rollout function to introduce early stopping
garage.sampler.default_worker.DefaultWorker.rollout = rollout

@wrap_experiment(snapshot_mode="none", log_dir="/tmp/")
def launch_experiment(ctxt=None, DTU=None, is_action_space_discrete=None, seed=None, modifyRuntime_method=None, gymEnvName=None, action_space=None, gracetime=None, n_epochs=None, max_episode_length=None, res_filepath=None):
    
    import garage.trainer
    # mock save function to avoid wasting time
    garage.trainer.Trainer.save = lambda self, epoch: print("skipp save.")

    from garage.envs import GymEnv
    from garage.experiment.deterministic import set_seed
    from garage.np.algos import CMAES
    from garage.sampler import LocalSampler
    from garage.tf.policies import CategoricalMLPPolicy, ContinuousMLPPolicy
    from garage.trainer import TFTrainer


    set_seed(seed)
    SEED=seed
    gymEnvName = gymEnvName
    with TFTrainer(ctxt) as trainer:
        global MAX_EPISODE_LENGTH
        if DTU: # DTU  means "Disable terminate_when_unhealthy"
            env = GymEnv(gymEnvName.replace("_DTU", ""), max_episode_length=MAX_EPISODE_LENGTH)
            print("terminate_when_unhealthy will be disabled in each episode. (DTU = True)")
        else:
            env = GymEnv(gymEnvName, max_episode_length=MAX_EPISODE_LENGTH)
            
        
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
        else:
            policy = ContinuousMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
        sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True)
        algo = CMAES(env_spec=env.spec, policy=policy, sampler=sampler, n_samples=POPSIZE)

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size)

def launch_from_python(seed, modifyRuntime_method, gymEnvName, action_space, gracetime, gens, max_episode_length, res_filepath):

    DTU = False
    if "DTU" in gymEnvName:
        DTU = True

    is_action_space_discrete = bool(["continuous", "discrete"].index(action_space))
    GRACE = gracetime
    n_epochs = gens
    MAX_EPISODE_LENGTH = max_episode_length
    res_filepath = res_filepath

    launch_experiment(
        DTU=DTU, 
        is_action_space_discrete=is_action_space_discrete,
        seed=seed, 
        modifyRuntime_method=modifyRuntime_method, 
        gymEnvName=gymEnvName, 
        action_space=action_space, 
        gracetime=gracetime, 
        n_epochs=gens, 
        max_episode_length=max_episode_length, 
        res_filepath=res_filepath
    )







