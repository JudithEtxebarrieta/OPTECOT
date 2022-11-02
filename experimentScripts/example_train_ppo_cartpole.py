import stable_baselines3
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from tqdm import tqdm as tqdm
import time


# Based on https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb



def evaluate(model, env, n_eval_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param n_eval_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last n_eval_episodes
    """
    all_episode_rewards = []
    for i in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    return np.mean(all_episode_rewards), np.std(all_episode_rewards)

env = gym.make('CartPole-v1')


# Use a separate environement for evaluation
eval_env = gym.make('CartPole-v1')


n_samples_reward = 100
default_tau = 0.02
default_max_episode_steps = 500
for accuracy in tqdm([1.0, 0.8, 0.6, 0.4]):
    # tau is the number of seconds between each frame
    env.env.tau = default_tau / accuracy

    # After max_episode_steps steps, stop simulation.
    env.env.spec.max_episode_steps = int(default_max_episode_steps * accuracy)
    env.env._max_episode_steps = int(default_max_episode_steps * accuracy)


    t = time.time()
    

    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)
    elapsed = time.time() - t

    mean_reward, std_reward = evaluate(model, eval_env, n_eval_episodes=100, deterministic=True)
    print(elapsed, mean_reward)








# # Save the model
# model.save("CartPole.PPOmodel")


