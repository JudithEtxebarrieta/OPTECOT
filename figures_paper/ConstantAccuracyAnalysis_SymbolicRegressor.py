'''
This script is an adaptation of script experimentScripts_SymbolicRegressor/ConstantAccuracyAnalysis_figures.py. 
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
import plotly.express as px

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def bootstrap_mean_and_confiance_interval(data,bootstrap_iterations=1000):
    '''The 95% confidence interval of a given data sample is calculated.'''
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def train_data_to_figure_data(df_train_acc,list_train_n_eval):
    '''
    The database associated with a parameter value is summarized to the data needed to construct 
    the solution quality curves. 
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
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

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
total_list_acc=np.load('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/list_acc.npy')
a_0=min(total_list_acc)
a_1=max(total_list_acc)
list_acc=[1.0,0.9,0.5,0.3,0.2,0.1]

# List with limits on the number of training evaluations to be drawn.
df_train_acc_min=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc"+str(max(list_acc))+".csv", index_col=0)
max_n_eval=np.load('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/max_n_eval.npy')
min_n_eval=max(df_train_acc_max.groupby("train_seed")["n_eval"].min())
split_n_eval=max(df_train_acc_min.groupby("train_seed")["n_eval"].min())
list_train_n_eval=np.arange(min_n_eval,max_n_eval,split_n_eval)

# Build and save graphs.
plt.figure(figsize=[7,6])
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\boldmath'
plt.subplots_adjust(left=0.18,bottom=0.11,right=0.85,top=0.88,wspace=0.3,hspace=0.2)
ax=plt.subplot(111)

colors=px.colors.qualitative.D3+['#FFB90F']
list_markers = ["o", "^", "s", "P", "v", 'D', "x", "1", "|", "+"]

ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
curve=0
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_n_eval)

    # Set maximum solution quality.
    if accuracy==1:
        score_limit=all_mean_scores[-1]

    # Draw curve.
    ax.fill_between(list_train_n_eval,all_q05_scores,all_q95_scores, alpha=.2, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean_scores, linewidth=1,label=r'\textbf{'+str(round(a_c(accuracy,a_0,a_1),2))+'}',color=colors[curve],marker=list_markers[curve],markevery=0.1)

    curve+=1

plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xlabel("$t$",fontsize=23)
ax.set_ylabel(r"\textbf{Solution quality}",fontsize=23)
ax.set_title(r'\textbf{SR}',fontsize=23)
ax.invert_yaxis()
leg=ax.legend(title="$c$",fontsize=17,title_fontsize=17,labelspacing=0.1,handlelength=0.8)
for line in leg.get_lines():
    line.set_linewidth(1.0)
plt.xscale('log')
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_SymbolicRegressor.pdf')
plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_SymbolicRegressor.png')
plt.show()
plt.close()

