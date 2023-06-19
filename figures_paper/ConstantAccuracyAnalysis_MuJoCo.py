'''
This script is an adaptation of script experimentScripts_MuJoCo/ConstantAccuracyAnalysis_figures.py. 
The design of the original graphs is modified to insert them in the paper.
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
    '''The 95% confidence interval of a given data sample is calculated.'''
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def train_data_to_figure_data(df_train_acc,list_train_steps):
    '''Constructing a graph formed by the solution quality curves.'''

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

        # Save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95
 
#==================================================================================================
# MAIN PROGRAM
#==================================================================================================

# Define relationship between accuracy (term used in the code) and cost (term used in the paper).
def a_c(a,a_0,a_1):
    c=(a-a_0)/(a_1-a_0)
    return c
total_list_acc=np.load('results/data/MuJoCo/ConstantAccuracyAnalysis/list_acc.npy')
a_0=min(total_list_acc)
a_1=max(total_list_acc)
list_acc=[1.0,0.9,0.6,0.3,0.2,0.1]

# List with limits of training steps to be drawn.
df_train_acc_min=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(max(list_acc))+".csv", index_col=0)
max_steps=np.load('results/data/MuJoCo/ConstantAccuracyAnalysis/max_steps.npy')
min_steps=max(df_train_acc_max.groupby("train_seed")["n_steps"].min())
split_steps=max(df_train_acc_min.groupby("train_seed")["n_steps"].min())
list_train_steps=np.arange(min_steps,max_steps,split_steps)

# Build and save graphs.
plt.figure(figsize=[7,6])
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\boldmath'
plt.subplots_adjust(left=0.18,bottom=0.11,right=0.85,top=0.88,wspace=0.17,hspace=0.2)
ax=plt.subplot(111)

list_markers = ["o", "^", "s", "P", "v", 'D', "x", "1", "|", "+"]

markers_ind=0
ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
for accuracy in list_acc:
    # Read database.
    df_train_acc=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_steps)

    # Set maximum solution quality.
    if accuracy==1:
        score_limit=all_mean_scores[-1]

    # Draw curve.
    ax.fill_between(list_train_steps,all_q05_scores,all_q95_scores, alpha=.2, linewidth=0)
    plt.plot(list_train_steps, all_mean_scores, linewidth=1,label=r'\textbf{'+str(round(a_c(accuracy,a_0,a_1),2))+'}',marker=list_markers[markers_ind],markevery=0.1)
    markers_ind+=1

ax.set_xlabel("$t$",fontsize=23)
ax.set_ylabel(r"\textbf{Solution quality}",fontsize=23)
ax.set_title(r'\textbf{Swimmer}',fontsize=23)
leg=ax.legend(title="$c$",fontsize=17,title_fontsize=17,labelspacing=0.1,handlelength=0.8)
for line in leg.get_lines():
    line.set_linewidth(1.0)
plt.axhline(y=score_limit,color='black', linestyle='--')
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_MuJoCo.pdf')
plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_MuJoCo.png')
plt.show()
plt.close()

