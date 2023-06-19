
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

def train_data_to_figure_data(df_train_acc,list_train_times):
    '''
    Constructing a graph formed by the solution quality curves.

    Parameters
    ==========
    df_train_acc: Database with training data.
    list_train_steps: List with time limits to be drawn in the graph.

    Returns
    =======
    all_mean: score averages (of all seeds) per training time limit set in list_train_times.
    all_q05,all_q95: Percentiles of scores per training time limit.
    '''

    # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in lists.
    for train_time in list_train_times:

        # Row indexes with a training time less than train_time.
        ind_train=df_train_acc["elapsed_time"] <= train_time
        
        # Group the previous rows by seed and keep the row per group that has the highest score value associated with it.
        interest_rows=df_train_acc[ind_train].groupby("seed")["score"].idxmax()

        # Calculate the mean and confidence interval of the scores.
        interest=list(df_train_acc['score'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confidence_interval(interest)

        # Save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95
 
#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

# Initialize graph.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.84,top=0.88,wspace=0.17,hspace=0.2)

# Read list with accuracy values considered.
list_acc=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/list_acc.npy')

# Define list with limits of training times to be drawn.
df_train_acc_min=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis"+str(max(list_acc))+".csv", index_col=0)
max_time=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/max_time.npy')
min_time=max(df_train_acc_max.groupby('seed')['elapsed_time'].min())
split_time=max(df_train_acc_min.groupby('seed')['elapsed_time'].min())
list_train_times=np.arange(min_time,max_time,split_time)

#--------------------------------------------------------------------------------------------------
# GRAPH 1 (Best results per accuracy value)
#--------------------------------------------------------------------------------------------------
# Get data for the graph.
train_times=[]
max_scores=[]
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_times)

    # Set maximum solution quality.
    if accuracy==1:
        score_limit=all_mean_scores[-1]

    # Find when the preset score limit is given for the first time.
    limit_scores=list(np.array(all_mean_scores)>=score_limit)
    if True in limit_scores:
        ind_min=limit_scores.index(True)
    else:
        ind_min=len(all_mean_scores)-1
    train_times.append(int(list_train_times[ind_min]))
    max_scores.append(all_mean_scores[ind_min])

# Draw graph.
ind_sort=np.argsort(train_times)
train_times_sort=[str(i) for i in sorted(train_times)]
max_scores_sort=[max_scores[i] for i in ind_sort]
acc_sort=[np.arange(len(list_acc)/10,0,-0.1)[i] for i in ind_sort]
acc_sort_str=[str(list_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax=plt.subplot(121)
ax.bar(train_times_sort,max_scores_sort,acc_sort,label=acc_sort_str,color=colors)
ax.set_xlabel("Train time")
ax.set_ylabel("Score (generated power)")
ax.set_title('Best results for each accuracy')
plt.axhline(y=score_limit,color='black', linestyle='--')

#--------------------------------------------------------------------------------------------------
# GRAPH 2 (General results)
#--------------------------------------------------------------------------------------------------
# Draw a curve for each accuracy value.
ax=plt.subplot(122)
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_times)

    # Draw curve.
    ax.fill_between(list_train_times,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_times, all_mean_scores, linewidth=2,label=str(accuracy))

ax.set_xlabel("Train time")
ax.set_ylabel("Score (generated power)")
ax.set_title('Solution quality curves (100 seeds for each accuracy)')
ax.legend(title="monteCarloPts \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='black', linestyle='--')

plt.savefig('results/figures/WindFLO/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()

