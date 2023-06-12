'''
This script evaluates 100 random policies (associated with a maximum accuracy) on 10 episodes of 
a defined environment with 10 different accuracy values for the time-step parameter. The relevant
data (average rewards and number of steps per evaluation) are stored for later access.
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import numpy as np
import pandas as pd
from tqdm import tqdm

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def build_policy_sample(n_sample=100):
    '''Build random sample of policies.'''

    # Initialize environment.
    env = gym.make('CartPole-v1')

    # Build sample of policies.
    policy_sample=[]
    for i in range(0,n_sample):
        policy = PPO(MlpPolicy,env,seed=i)
        policy_sample.append(policy)

    return policy_sample

def evaluate_policy(policy,accuracy):
    ''' Evaluate a policy on 10 episodes of a defined environment with a given accuracy.'''

    # Initialize the evaluation environment with the specified accuracy.
    env = gym.make('CartPole-v1')
    env.env.tau = env.env.tau / accuracy
    env.env.spec.max_episode_steps = int(env.env.spec.max_episode_steps*accuracy)
    env._max_episode_steps = env.unwrapped.spec.max_episode_steps
    
    # To ensure that the same episodes are used in each call to the function.
    env.seed(0)
    obs=env.reset()
    
    all_episode_reward=[]
    all_episode_steps=[]
    for i in range(10):
        episode_rewards = 0
        episode_steps=0
        done = False
        while not done:
            action, _states = policy.predict(obs, deterministic=True)         
            obs, reward, done, info = env.step(action) 
            episode_rewards+=reward 
            episode_steps+=1

        all_episode_reward.append(episode_rewards)
        all_episode_steps.append(episode_steps)
        obs = env.reset() 
    
    return np.mean(all_episode_reward),np.mean(all_episode_steps)

def evaluate_policy_sample(policy_sample,accuracy):
    '''Evaluate a set of policies on 10 episodes of a defined environment with a specific accuracy.'''

    for i in tqdm(range(len(policy_sample))):
        reward,steps=evaluate_policy(policy_sample[i],accuracy)
        df.append([accuracy,i,reward,steps])

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Build sample of policies.
policy_sample=build_policy_sample(100)

# list of accuracies.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# Build a database with information on rewards and evaluation times obtained by evaluating the policy sample with different accuracy values.
df=[]
for accuracy in tqdm(list_acc):
    evaluate_policy_sample(policy_sample,accuracy)

# Save the database.
df=pd.DataFrame(df,columns=['accuracy','n_policy','reward','steps'])
df.to_csv('results/data/CartPole/UnderstandingAccuracy.csv')


