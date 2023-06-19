'''
This script is an adaptation of script experimentScripts_Turbines/ConstantAccuracyAnalysis_figures.py. 
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
import plotly.express as px
from matplotlib.patches import Rectangle

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

def train_data_to_figure_data(df_train_acc,list_train_times):
    '''
    The database associated with a parameter value is summarized to the data needed to 
    construct the solution quality curves. 
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
# Define relationship between accuracy (term used in the code) and cost (term used in the paper).
def a_c(a,a_0,a_1):
    c=(a-a_0)/(a_1-a_0)
    return c
total_list_acc=np.load('results/data/Turbines/ConstantAccuracyAnalysis/list_acc.npy')
a_0=min(total_list_acc)
a_1=max(total_list_acc)
list_acc=[1.0,0.8,0.6,0.4,0.3,0.1]

# Define list with limits of training times to be drawn.
df_train_acc_min=pd.read_csv("results_Hipatia/df_ConstantAccuracyAnalysis_acc"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results_Hipatia/df_ConstantAccuracyAnalysis_acc"+str(max(list_acc))+".csv", index_col=0)
max_time=np.load('results_Hipatia/max_time.npy')
min_time=max(df_train_acc_max.groupby('seed')['elapsed_time'].min())
split_time=max(df_train_acc_min.groupby('seed')['elapsed_time'].min())
list_train_times=np.arange(min_time,60*60,10)

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
    df_train_acc=pd.read_csv("results_Hipatia/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_times)

    # Set maximum solution quality.
    if accuracy==max(list_acc):
        score_limit=all_mean_scores[-1]

    # Draw curve.
    ax.fill_between(list_train_times,all_q05_scores,all_q95_scores, alpha=.2, linewidth=0,color=colors[curve])
    plt.plot(list_train_times, all_mean_scores, linewidth=1,label=r'\textbf{'+str(round(a_c(accuracy,a_0,a_1),2))+'}',color=colors[curve],marker=list_markers[curve],markevery=0.1)
    curve+=1

plt.gca().add_patch(Rectangle((2000, .516), 3600-2000, 0.526-0.516,facecolor='white',edgecolor='black'))
ax.set_ylim([0.42,0.55])
ax.set_xlabel("$t$",fontsize=23)
ax.set_ylabel(r"\textbf{Solution quality}",fontsize=23)
ax.set_title(r'\textbf{Turbines}',fontsize=23)
leg=ax.legend(title="$c$",fontsize=16.5,title_fontsize=16.5,labelspacing=0.1,handlelength=0.8,loc='upper left')
plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xscale('log')
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_Turbines1.png')
plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_Turbines1.pdf')
plt.show()
plt.close()

# Zoom.
plt.figure(figsize=[7,4])
plt.subplots_adjust(left=0.57,bottom=0.44,right=0.93,top=0.88,wspace=0.3,hspace=0.2)
ax=plt.subplot(111)
ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
curve=0
for accuracy in list_acc:

    # Read database.
    df_train_acc=pd.read_csv("results_Hipatia/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extract relevant information from the database.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_times)

    # Set maximum solution quality.
    if accuracy==max(list_acc):
        score_limit=all_mean_scores[-1]

    # Draw curve.
    ax.fill_between(list_train_times,all_q05_scores,all_q95_scores, alpha=.2, linewidth=0,color=colors[curve])
    plt.plot(list_train_times, all_mean_scores, linewidth=1,label=str(round(a_c(accuracy,a_0,a_1),2)),color=colors[curve],marker=list_markers[curve],markevery=0.1)
    curve+=1

ax.set_yticks(np.arange(0.520,0.524,0.002),rotation=0)
ax.set_xticks(np.arange(2200,3600,500))
ax.ticklabel_format(style='sci', axis='x', useOffset=True, scilimits=(0,0))
ax.set_title('Zoom',fontsize=23)
ax.set_xlim([2000,3750])
ax.set_ylim([0.518,0.526])
plt.axhline(y=score_limit,color='black', linestyle='--')
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_Turbines2.png')
plt.savefig('figures_paper/figures/ConstantAccuracyAnalysis/ConstantAccuracyAnalysis_Turbines2.pdf')
plt.show()
plt.close()