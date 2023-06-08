
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
import plotly.express as px

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
    The database associated with a parameter value is summarized to the data needed to construct the solution quality curves. 

    Parameters
    ==========
    df_train_acc: Database with training information associated with a N accuracy.
    list_train_times: List with number of training times to be drawn.

    Return
    ====== 
    all_mean: Averages of the scores per training time limit set in list_train_times.
    all_q05,all_q95: Percentiles of the scores per training time limit.
    '''

    # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in lists.
    for train_time in list_train_times:

        # Indexes of rows with a training time less than train_time.
        ind_train=df_train_acc["elapsed_time"] <= train_time
        
        # Group the previous rows by seed and keep the row per group that has the highest score value associated with it.
        interest_rows=df_train_acc[ind_train].groupby("seed")["score"].idxmax()

        # Calculate the mean and confidence interval of the score.
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
# List of colors.
colors=px.colors.qualitative.D3

# Initialize graph.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.84,top=0.88,wspace=0.17,hspace=0.2)

# Read list of accuracy values.
list_acc=np.load('results/data/Turbines/ConstantAccuracyAnalysis/list_acc.npy')

# Define list with limits of training times to be drawn.
df_train_acc_min=pd.read_csv("results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(max(list_acc))+".csv", index_col=0)
max_time=np.load('results/data/Turbines/ConstantAccuracyAnalysis/max_time.npy')
min_time=max(df_train_acc_max.groupby('seed')['elapsed_time'].min())
split_time=max(df_train_acc_min.groupby('seed')['elapsed_time'].min())
list_train_times=np.arange(min_time,max_time+10,10)


#--------------------------------------------------------------------------------------------------
# GRAPH 1 (Best results per accuracy value)
#--------------------------------------------------------------------------------------------------

# Get data for the graph.
train_times=[]
max_scores=[]
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_times)

    # Set maximum quality.
    if accuracy==max(list_acc):
        score_limit=all_mean_scores[-1]

    # Find when the quality limit is given for the first time.
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
ax.set_ylabel("Score")
ax.set_title('Best results for each model')
plt.axhline(y=score_limit,color='black', linestyle='--')


#--------------------------------------------------------------------------------------------------
# GRAPH 2 (general results)
#--------------------------------------------------------------------------------------------------
ax=plt.subplot(122)

# Draw a curve for each accuracy value.
for accuracy in list_acc:
    
    # Read database.
    df_train_acc=pd.read_csv("results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_times)

    if accuracy==max(list_acc):
        score_limit=all_mean_scores[-1]

    # Draw curve.
    ax.fill_between(list_train_times,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_times, all_mean_scores, linewidth=2,label=str(accuracy))

ax.set_xlabel("Train time")
ax.set_ylabel("Score")
ax.set_title('Model evaluation \n (train 100 seeds, test N=100)')
ax.legend(title="N parameter accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xscale('log')

# Save graph.
plt.savefig('results/figures/Turbines/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()


