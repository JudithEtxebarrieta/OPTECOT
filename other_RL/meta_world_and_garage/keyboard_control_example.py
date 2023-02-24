"""
Modified version of https://github.com/rlworkgroup/metaworld/blob/master/scripts/keyboard_control.py

Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.
"""
import sys
import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor


# print("\n".join(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN))
cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["pick-place-v2-goal-hidden"]







pygame.init()
screen = pygame.display.set_mode((400, 300))


char_to_action = {
    'w': np.array([0, -1, 0, 0]),
    'a': np.array([1, 0, 0, 0]),
    's': np.array([0, 1, 0, 0]),
    'd': np.array([-1, 0, 0, 0]),
    'q': np.array([1, -1, 0, 0]),
    'e': np.array([-1, -1, 0, 0]),
    'z': np.array([1, 1, 0, 0]),
    'c': np.array([-1, 1, 0, 0]),
    'k': np.array([0, 0, 1, 0]),
    'j': np.array([0, 0, -1, 0]),
    'n': np.array([0, 0, 0, 1]),
    'm': np.array([0, 0, 0, -1]),
    'h': 'close',
    'l': 'open',
    'x': 'toggle',
    'r': 'reset',
    'p': 'put obj in hand',
}




env = cls()
env.max_path_length = 200000
env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
env.reset()
env._freeze_rand_vec = True
lock_action = False
random_action = False
obs = env.reset()
action = np.zeros(4)
while True:
    done = False
    if not lock_action:
        action[:] = 0
    if not random_action:
        for event in pygame.event.get():
            event_happened = True
            if event.type == QUIT:
                sys.exit()
            if event.type == KEYDOWN:
                char = event.dict['key']
                new_action = char_to_action.get(chr(char), None)
                if new_action == 'toggle':
                    lock_action = not lock_action
                elif new_action == 'reset':
                    done = True
                elif new_action == 'close':
                    action[3] = 1
                elif new_action == 'open':
                    action[3] = -1
                elif new_action is not None:
                    action[:] = new_action[:]
                else:
                    action = np.zeros(4)
                print(action)
    else:
        action = env.action_space.sample()
    print(env.action_space)
    print(action)
    ob, reward, done, infos = env.step(action)
    if done:
        obs = env.reset()
    env.render()
