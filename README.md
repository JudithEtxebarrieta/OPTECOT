# VariableModelAccuracy

### Install dependencies:

```
bash install_sb3.sh
```

### Chnge model accuracy in cart pole 

We change the accuracy of the model by modifying tau and max_episode_steps.
Tau measures the number of seconds from step to step and max_episode_steps is the stopping criterion (max steps) for the evaluation of the model.

Run the script:
```
python experimentScripts/change_tau_cartpole.py
```

First, notice that the averaage number of steps per episode was not the same as the max_episode_steps.

| max_episode_steps | average steps per episode |
|-------------------|---------------------------|
| 500               | 440                       |
| 400               | 376                       |
| 300               | 272                       |
| 200               | 126                       |

The reason is that there are other stopping criterions, such as the cart moving too much to the left or right and getting out of a certain area. 


Below we show the gaussian KDE of the reward of the model <em>models/CartPole.PPOmodel</em>, with different accuracy parameters. Note that the accuracy modifies the max_episode_steps and tau, and a lower accuracy also implies a faster computation time (less steps need to be computed).

<img src="results/figures/model_accuracy_and_runtime.png" alt="drawing" width="400"/>