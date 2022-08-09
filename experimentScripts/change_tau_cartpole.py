import stable_baselines3
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import time
from tqdm import tqdm as tqdm


# Based on https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb

env = gym.make('CartPole-v1')

# Load the trained model
model = PPO.load("models/CartPole.PPOmodel")


def evaluate(model, env, n_eval_episodes, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param n_eval_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last n_eval_episodes
    """
    all_episode_rewards = []
    n_steps_per_episode = []
    for i in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()

        # Need to update max episodes in each reset
        env._max_episode_steps = env.unwrapped.spec.max_episode_steps

        i = 0
        while not done:
            i += 1
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))
        n_steps_per_episode.append(i)

    # Notice that sometimes it will stop before _max_episode_steps.
    # For example, if the cart goes outside a defined area, it stops.
    print("On average, each episode had ", np.mean(n_steps_per_episode) , "steps. While env._max_episode_steps =",env._max_episode_steps)

    return all_episode_rewards



fig, ax = plt.subplots()

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
    rewards = np.array(evaluate(model, env, n_eval_episodes=n_samples_reward)).reshape(-1, 1)
    elapsed = time.time() - t

    # Gaussian KDE, based on https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#sphx-glr-auto-examples-neighbors-plot-kde-1d-py
    kde = KernelDensity(kernel="gaussian", bandwidth=10).fit(rewards)
    x_plot = np.linspace(min(rewards)[0], max(rewards)[0], 200).reshape(-1, 1)
    log_dens = kde.score_samples(x_plot)
    ax.plot(x_plot, np.exp(log_dens), label=str(accuracy) + ", in " + f"{elapsed:.2f}" + " s")

plt.xlabel("Reward")
plt.ylabel("Probability density")
plt.legend(title="Model accuracy")
plt.savefig("results/figures/model_accuracy_and_runtime.pdf")
plt.savefig("results/figures/model_accuracy_and_runtime.png")
plt.close()
