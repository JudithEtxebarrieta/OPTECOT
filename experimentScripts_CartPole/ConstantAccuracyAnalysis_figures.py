'''
This script is used to graphically represent the numerical results obtained in 
"ConstantAccuracyAnalysis_data.py".
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
from scipy.stats import norm
import time
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import pandas as pd

#==================================================================================================
# FUNCTIONS
#==================================================================================================
def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
    '''
    The 95% confidence interval of a given data sample is calculated.

    Parameters
    ==========
    data (list): Data on which the range between percentiles will be calculated.
    bootstrap_iterations (int): Number of subsamples of data to be considered to calculate the percentiles of their means. 

    Return
    ======
    The mean of the original data together with the percentiles of the means obtained from the subsampling of the data. 
    '''
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def train_data_to_figure_data(df_train_acc,list_train_steps):
    '''
    The database associated with a parameter value is summarized to the data needed to construct the solution quality curves. 

    Parameters
    ==========
    df_train_acc: Database with training information associated with a time-step accuracy.
    list_train_steps: List with number of training steps to be drawn.

    Return
    ====== 
    The average of the original rewards together with the percentiles of the averages obtained 
    from the subsampling performed on these data.
    '''

    # Initialize lists for the graph.
    all_mean_rewards=[]
    all_q05_reawrds=[]
    all_q95_rewards=[]

    # Fitting lists.
    for train_steps in list_train_steps:
        # Row indexes with a number of training steps less than train_steps.
        ind_train=df_train_acc["info_steps"] <= train_steps
        
        # Group the previous rows by seed and save the row per group that has the highest reward value associated with it.
        interest_rows=df_train_acc[ind_train].groupby("seed")["mean_reward"].idxmax()

        # Calculate the mean and confidence interval of the reward.
        interest_rewards=list(df_train_acc['mean_reward'][interest_rows])
        mean_reward,q05_reward,q95_reward=bootstrap_mean_and_confidence_interval(interest_rewards)

        # Save data.
        all_mean_rewards.append(mean_reward)
        all_q05_reawrds.append(q05_reward)
        all_q95_rewards.append(q95_reward)

    return all_mean_rewards,all_q05_reawrds,all_q95_rewards

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

# Initialize graph.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.84,top=0.88,wspace=0.17,hspace=0.2)

# Read list with accuracy values considered.
grid_acc=np.load('results/data/CartPole/ConstantAccuracyAnalysis/grid_acc.npy')

# List with limits of training steps to be drawn.
df_train_acc_min=pd.read_csv("results/data/CartPole/ConstantAccuracyAnalysis/df_train_acc"+str(min(grid_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/CartPole/ConstantAccuracyAnalysis/df_train_acc"+str(max(grid_acc))+".csv", index_col=0)
max_train_steps=np.load('results/data/CartPole/ConstantAccuracyAnalysis/max_train_steps.npy')
min_steps=max(df_train_acc_max.groupby("seed")["steps"].min())
split_steps=max(df_train_acc_min.groupby("seed")["steps"].min())
list_train_steps=np.arange(min_steps,max_train_steps,split_steps)

#--------------------------------------------------------------------------------------------------
# GRAPH 1 (best results per accuracy value)
#--------------------------------------------------------------------------------------------------
ax=plt.subplot(121)

# Get data for the graph.
train_steps=[]
max_rewards=[]
for accuracy in grid_acc:
    # Read database.
    df_train_acc=pd.read_csv("results/data/CartPole/ConstantAccuracyAnalysis/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_rewards,all_q05_rewards,all_q95_rewards=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Set maximum solution quality.
    if accuracy==1:
        score_limit=all_mean_rewards[-1]

    # Find when the maximum reward is given for the first time.
    limit_rewards=list(np.array(all_mean_rewards)>=score_limit)
    if True in limit_rewards:
        ind_max=limit_rewards.index(True)
    else:
        ind_max=len(all_mean_rewards)-1
    train_steps.append(list_train_steps[ind_max])
    max_rewards.append(all_mean_rewards[ind_max])

# Draw graph.
ind_sort=np.argsort(train_steps)
train_steps_sort=[str(i) for i in sorted(train_steps)]
max_rewards_sort=[max_rewards[i] for i in ind_sort]
acc_sort=[np.arange(len(grid_acc)/10,0,-0.1)[i] for i in ind_sort]
acc_sort_str=[str(grid_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax.bar(train_steps_sort,max_rewards_sort,acc_sort,label=acc_sort_str,color=colors)
plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xlabel("Train steps")
ax.set_ylabel("Maximum reward")
ax.set_title('Best results for each model')

#--------------------------------------------------------------------------------------------------
# GRAPH 2 (general results)
#--------------------------------------------------------------------------------------------------
ax=plt.subplot(122)

# Draw a curve for each accuracy value.
for accuracy in grid_acc:
    # read database.
    df_train_acc=pd.read_csv("results/data/CartPole/ConstantAccuracyAnalysis/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_rewards,all_q05_rewards,all_q95_rewards=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Draw curve.
    ax.fill_between(list_train_steps,all_q05_rewards,all_q95_rewards, alpha=.5, linewidth=0)
    plt.plot(list_train_steps, all_mean_rewards, linewidth=2,label=str(accuracy))

# Save graph.
plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xlabel("Train steps")
ax.set_ylabel("Mean reward")
ax.set_title('Models evaluations (train 30 seeds, test 100 episodes)')
ax.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

plt.savefig('results/figures/CartPole/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()