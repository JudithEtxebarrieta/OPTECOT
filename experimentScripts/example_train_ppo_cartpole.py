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


class stopwatch:
    
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
        
n_samples_reward = 100
default_tau = 0.02
default_max_episode_steps = 500
for accuracy in tqdm([1.0, 0.8, 0.6, 0.4]):
    # tau is the number of seconds between each frame
    env.env.tau = default_tau / accuracy

    # After max_episode_steps steps, stop simulation.
    env.env.spec.max_episode_steps = int(default_max_episode_steps * accuracy)
    env.env._max_episode_steps = int(default_max_episode_steps * accuracy)



    sw = stopwatch()
    sw.reset()



    # Callback in each iteration
    def callback_in_each_iteration(self, num_timesteps: int, total_timesteps: int) -> None:

        sw.pause()    
        mean_reward, std_reward = evaluate(model, eval_env, n_eval_episodes=100, deterministic=True)
        print("steps", "time", "reward", sep=",")
        print(num_timesteps, sw.get_time(), mean_reward, sep=",")
        sw.resume()

        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps) # Esta linea la usa la funcion que sustituimos: no cambiar esta linea.

        

    import stable_baselines3.common.base_class
    stable_baselines3.common.base_class.BaseAlgorithm._update_current_progress_remaining = callback_in_each_iteration
    


    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)
    elapsed = sw.get_time()

    mean_reward, std_reward = evaluate(model, eval_env, n_eval_episodes=100, deterministic=True)
    print(elapsed, mean_reward)








# # Save the model
# model.save("CartPole.PPOmodel")


