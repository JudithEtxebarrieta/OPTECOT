import stable_baselines3
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy


# Based on https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb

env = gym.make('CartPole-v1')
model = PPO(MlpPolicy, env, verbose=1)


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

# Use a separate environement for evaluation
eval_env = gym.make('CartPole-v1')

# Evaluate random Agent, before training
mean_reward, std_reward = evaluate(model, eval_env, n_eval_episodes=100)


model.learn(total_timesteps=10000)
# Evaluate the trained agent
mean_reward, std_reward = evaluate(model, eval_env, n_eval_episodes=100)


print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Save the model
model.save("CartPole.PPOmodel")

# # Load the trained model
# model = PPO.load("CartPole.PPOmodel")



# Generate video of model
import model_to_video
model_to_video.record_video('CartPole-v1', model)