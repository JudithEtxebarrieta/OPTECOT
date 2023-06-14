'''
This script is used to graphically represent the numerical results obtained in 
"ConstantAccuracyAnalysis_data.py".
'''

#==================================================================================================
# LIBRERIES
#==================================================================================================
import numpy as np
from scipy.stats import norm
import time
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
import math

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
    Constructing a graph formed by the solution quality curves.

    Parameters
    ==========
    df_train_acc: Database with training data.
    list_train_steps: List with limits of number of steps to be drawn in the graph.

    Returns
    =======
    all_mean: Rewards averages (of all seeds) per training step limit set in list_train_steps.
    all_q05,all_q95: Percentiles of rewards per training step limit.
    '''

    # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in lists.
    for train_steps in list_train_steps:

        # Row indexes with a number of training steps less than train_steps.
        ind_train=df_train_acc["n_steps"] <= train_steps
        
        # Group the previous rows by seed and keep the row per group that has the highest reward value associated with it.
        interest_rows=df_train_acc[ind_train].groupby("train_seed")["reward"].idxmax()

        # Calculate the mean and confidence interval of the reward.
        interest=list(df_train_acc['reward'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confidence_interval(interest)

        # save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95
 
#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

# Initialize graph.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.15,right=0.84,top=0.9,wspace=0.17,hspace=0.2)

# Read list with accuracy values considered.
list_acc=np.load('results/data/MuJoCo/ConstantAccuracyAnalysis/list_acc.npy')

# Define list with limits of training times to be drawn.
df_train_acc_min=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(max(list_acc))+".csv", index_col=0)
max_steps=np.load('results/data/MuJoCo/ConstantAccuracyAnalysis/max_steps.npy')
min_steps=max(df_train_acc_max.groupby("train_seed")["n_steps"].min())
split_steps=max(df_train_acc_min.groupby("train_seed")["n_steps"].min())
list_train_steps=np.arange(min_steps,max_steps,split_steps)

#--------------------------------------------------------------------------------------------------
# GRAPH 1 (Best results per accuracy value)
#--------------------------------------------------------------------------------------------------

# Get data for the graph.
train_stepss=[]
max_scores=[]
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Set maximum solution quality
    if accuracy==1:
        score_limit=all_mean_scores[-1]

    # Find out when the maximum solution quality is given for the first time.
    limit_scores=list(np.array(all_mean_scores)>=score_limit)
    if True in limit_scores:
        ind_min=limit_scores.index(True)
    else:
        ind_min=len(all_mean_scores)-1
    train_stepss.append(int(list_train_steps[ind_min]))
    max_scores.append(all_mean_scores[ind_min])


# Draw graph.
ind_sort=np.argsort(train_stepss)
train_stepss_sort=[str(i) for i in sorted(train_stepss)]
max_scores_sort=[max_scores[i] for i in ind_sort]
acc_sort=[np.arange(len(list_acc)/10,0,-0.1)[i] for i in ind_sort]
acc_sort_str=[str(list_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax=plt.subplot(121)
ax.bar(train_stepss_sort,max_scores_sort,acc_sort,label=acc_sort_str,color=colors)
ax.tick_params(axis='x', labelrotation = 45)
ax.set_xlabel("Train steps")
ax.set_ylabel("Reward")
ax.set_title('Best results for each accuracy')
plt.axhline(y=score_limit,color='black', linestyle='--')


#--------------------------------------------------------------------------------------------------
# GRAPH 2 (General results)
#--------------------------------------------------------------------------------------------------
ax=plt.subplot(122)

# Draw a curve for each accuracy value.
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_steps)

    # Draw curve.
    ax.fill_between(list_train_steps,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_steps, all_mean_scores, linewidth=2,label=str(accuracy))

# Save graph.
ax.set_xlabel("Train steps")
ax.set_ylabel("Reward")
ax.set_title('Solution quality curves (100 seeds for each accuracy)')
ax.legend(title="Time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='black', linestyle='--')

plt.savefig('results/figures/MuJoCo/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()

