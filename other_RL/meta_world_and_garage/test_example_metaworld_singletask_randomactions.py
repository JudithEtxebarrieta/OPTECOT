from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

# For visible goal
cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]

# # For observable goal
# cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["door-open-v2-goal-hidden"]



env = cls(seed=2)

for i in range(2000):
    env.reset()  # Reset environment
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action

    env.render()