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

def train_data_to_figure_data(df_train_acc,list_train_n_eval):
    '''
    The database associated with an accuracy value is summarized to the data needed to construct the solution quality curves. 

    Parameters
    ==========
    df_train_acc: Database with information extracted from the surface search process for a given 
    accuracy value (a size of the initial set of points).
    list_train_n_eval: List with the number of evaluations to be drawn in the graph.

    Return
    ======
    List of means and confidence intervals associated with each number of evaluations highlighted in list_train_n_eval.
    '''

    # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in lists.
    for train_n_eval in list_train_n_eval:

        # Row indexes with a number of training evaluations less than train_n_eval.
        ind_train=df_train_acc["n_eval"] <= train_n_eval
        
        # Group the previous rows by seed and keep the row per group that has the lowest score value associated with it.
        interest_rows=df_train_acc[ind_train].groupby("train_seed")["score"].idxmin()

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
colors=px.colors.qualitative.D3+['#FFB90F']

# Initialize graph.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.84,top=0.88,wspace=0.3,hspace=0.2)

#--------------------------------------------------------------------------------------------------
# GRAPH 1 (Best results per accuracy value)
#--------------------------------------------------------------------------------------------------
ax=plt.subplot(121)

# Read list with accuracy values considered.
list_acc=np.load('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/list_acc.npy')

# List with limits on the number of training evaluations to be drawn.
df_train_acc_min=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc"+str(max(list_acc))+".csv", index_col=0)
max_n_eval=np.load('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/max_n_eval.npy')
min_n_eval=max(df_train_acc_max.groupby("train_seed")["n_eval"].min())
split_n_eval=max(df_train_acc_min.groupby("train_seed")["n_eval"].min())
list_train_n_eval=np.arange(min_n_eval,max_n_eval,split_n_eval)

# Get data for the graph.
train_times=[]
max_scores=[]
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_n_eval)

    # Set maximum solution quality.
    if accuracy==1:
        score_limit=all_mean_scores[-1]

    # Find when the maximum solution quality is given for the first time.
    limit_scores=list(np.array(all_mean_scores)<=score_limit)
    if True in limit_scores:
        ind_min=limit_scores.index(True)
    else:
        ind_min=len(all_mean_scores)-1
    train_times.append(list_train_n_eval[ind_min])
    max_scores.append(all_mean_scores[ind_min])

# Draw graph.
ind_sort=np.argsort(train_times)
train_times_sort=[str(i) for i in sorted(train_times)]
max_scores_sort=[max_scores[i] for i in ind_sort]
acc_sort=[list_acc[i] for i in ind_sort]
acc_sort_str=[str(list_acc[i]) for i in ind_sort]
colors_sort=[colors[i] for i in ind_sort]

ax.bar(train_times_sort,max_scores_sort,acc_sort,label=acc_sort_str,color=colors_sort)
ax.set_xlabel("Train evaluations")
ax.set_ylabel("Score (MAE)")
ax.set_title('Best results for each accuracy')
plt.axhline(y=score_limit,color='black', linestyle='--')

#--------------------------------------------------------------------------------------------------
# GRAPH 2 (general results)
#--------------------------------------------------------------------------------------------------
ax=plt.subplot(122)

# Draw a curve for each accuracy value.
curve=0
for accuracy in list_acc:

    # read database.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_n_eval)

    # Draw curve.
    ax.fill_between(list_train_n_eval,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean_scores, linewidth=2,label=str(accuracy),color=colors[curve])

    curve+=1

plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Solution quality curves (100 seeds for each accuracy)')
ax.legend(title="Point set size \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.xscale('log')
plt.yscale('log')

# save graph.
plt.savefig('results/figures/SymbolicRegressor/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()

