'''
This script is an adaptation of script experimentScripts_SymbolicRegressor/OptimalAccuracyAnalysis_figures.py. 
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

def train_data_to_figure_data(df_train,type_eval):
    '''
    The database associated with the heuristic execution information is summarized to the data
    needed to construct the solution quality curve. 
    '''

    # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in lists.
    for train_n_eval in list_train_n_eval:

        # Row indexes with a number of training evaluations less than train_n_eval.
        ind_train=df_train[type_eval] <= train_n_eval
  
        # Group the previous rows by seed and keep the row per group that has the lowest score value associated with it.
        interest_rows=df_train[ind_train].groupby("train_seed")["score"].idxmin()

        # Calculate the mean and confidence interval of the score.
        interest=list(df_train['score'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confidence_interval(interest)

        # Save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95,list_train_n_eval

def draw_accuracy_behaviour(df_train,type_n_eval):
    '''Construction of curves showing the behavior of the accuracy during training.'''

    # Define relationship between accuracy (term used in the code) and cost (term used in the paper).
    def a_c(a,a_0,a_1):
        c=(a-a_0)/(a_1-a_0)
        return c
    a_0=0.1 
    a_1=1

    # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in lists.
    for train_n_eval in list_train_n_eval:

        # Row indexes with the number of training evaluations closest to train_n_eval.
        ind_down=df_train[type_n_eval] <= train_n_eval
        ind_per_seed=df_train[ind_down].groupby('train_seed')[type_n_eval].idxmax()

        # Group the previous rows by seeds and keep the accuracy values.
        list_acc=list(df_train[ind_down].loc[ind_per_seed]['acc'])

        # Calculate the mean and confidence interval of the accuracy.
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list_acc)

        # Save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Draw graph.
    all_q05=[round(a_c(i,a_0,a_1),2)for i in all_q05]
    all_q95=[round(a_c(i,a_0,a_1),2)for i in all_q95]
    all_mean=[round(a_c(i,a_0,a_1),2)for i in all_mean]
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[1])
    plt.plot(list_train_n_eval, all_mean, linewidth=1,label='OPTECOT',color=colors[1],marker=list_markers[1],markevery=0.1)

def draw_and_save_figures_per_heuristic():
    '''
    Function to draw and save the graphs (with solution quality curves and accuracy behavior) 
    according to the selected heuristic.
    '''
    
    global ax

    # Initialize graph.
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'
    plt.figure(figsize=[6,10])
    plt.subplots_adjust(left=0.21,bottom=0.09,right=0.9,top=0.94,wspace=0.24,hspace=0.17)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: Solution quality during training using the total number of evaluations.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(211)

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')

    # Constant use of accuracy (default situation).
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[0])
    plt.plot(list_train_n_eval, all_mean, linewidth=1,label=r'\textbf{Original}',color=colors[0],marker=list_markers[0],markevery=0.1)

    # Optimal use of accuracy (heuristic application).
    df=df_optimal_acc[df_optimal_acc['heuristic_param']=='[5, 3]']
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[1])
    plt.plot(list_train_n_eval, all_mean, linewidth=1,label=r'\textbf{OPTECOT}',color=colors[1],marker=list_markers[1],markevery=0.1)

    ax.set_ylabel(r"\textbf{Solution quality}",fontsize=23)
    ax.legend(fontsize=18,title_fontsize=18,labelspacing=0.1,handlelength=0.8)
    ax.set_title(r'\textbf{SR}',fontsize=23)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    ax.invert_yaxis()
    ax.set_xscale('log')

    #----------------------------------------------------------------------------------------------
    # GRAPH 2: Graphical representation of the accuracy behavior.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(212)
    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')

    # Draw curves.
    df=df_optimal_acc[df_optimal_acc['heuristic_param']=='[5, 3]']
    draw_accuracy_behaviour(df,'n_eval')

    ax.set_xlabel("$t$",fontsize=23)
    ax.ticklabel_format(style='sci', axis='x', useOffset=True, scilimits=(3010,0))
    ax.set_ylabel("$c$",fontsize=23)
    ax.set_xticks(range(200000,1000000,250000))
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    ax.set_title('')
    ax.set_xscale('log')

    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/OptimalAccuracyAnalysis_SymbolicRegressor.pdf')
    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/OptimalAccuracyAnalysis_SymbolicRegressor.png')
    plt.show()
    plt.close()

def draw_heuristic_effectiveness():
    '''Construct graph showing the increase in quality and time savings obtained after applying the heuristic.'''

    # Initialize graph.
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'
    plt.figure(figsize=[4,4])
    plt.subplots_adjust(left=0.27,bottom=0.14,right=0.95,top=0.94,wspace=0.3,hspace=0.07)

    all_mean_acc_max,_,_,_=train_data_to_figure_data(df_max_acc,'n_eval')
    all_mean_h,_,_,_=train_data_to_figure_data(df_optimal_acc[df_optimal_acc['heuristic_param']=='[5, 3]'],'n_eval')

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: Curve of quality increment.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(411)

    quality_percentage=list((np.array(all_mean_acc_max)/np.array(all_mean_h))*100)
    print('Quality above 100:',sum(np.array(quality_percentage)>=100)/len(quality_percentage))
    print('Best quality:',max(quality_percentage),np.mean(quality_percentage),np.std(quality_percentage))

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    plt.plot(list_train_n_eval,quality_percentage,color=colors2[3],label='Quality', linewidth=1.2)
    plt.axhline(y=100,color='black', linestyle='--')
    ax.set_title(r'\textbf{SR}',fontsize=15)
    ax.set_ylabel(r"\textbf{QI}",fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks([])
    ax.set_xscale('log')
    ax.set_ylim([88,132])
    ax.set_yticks([100,115,130])

    #----------------------------------------------------------------------------------------------
    # GRAPH 2: Difference between the mean quality curves associated with heuristic application 
    # and original situation.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(4,1,(2,3))

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    #plt.gca().add_patch(Rectangle((400000, .200), 320000-400000, 0.180-0.200,facecolor='white',edgecolor='black'))
    plt.plot(list_train_n_eval, all_mean_acc_max, linewidth=1,label=r'\textbf{Original}',color=colors[0])
    plt.plot(list_train_n_eval, all_mean_h, linewidth=1,label=r'\textbf{OPTECOT}',color=colors[1],linestyle = '--')
    ax.fill_between(list_train_n_eval, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)<np.array(all_mean_acc_max),facecolor='green', alpha=.2,interpolate=True,label=r'\textbf{Improvement}')
    ax.fill_between(list_train_n_eval, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)>np.array(all_mean_acc_max),facecolor='red', alpha=.2,interpolate=True)#,label=r'\textbf{Worsening}')

    ax.set_ylabel(r"\textbf{Solution quality}",fontsize=15)
    ax.legend(fontsize=11,title_fontsize=11,labelspacing=0.01,loc='lower right',handlelength=1)
    plt.xticks([])
    plt.yticks(fontsize=15)
    ax.invert_yaxis()
    ax.set_ylim([0.45,0.08])
    ax.set_yticks(np.arange(0.1,0.4,0.1))
    ax.set_xscale('log')

    #----------------------------------------------------------------------------------------------
    # GRAPH 3: Curve of time saving.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(414)

    list_time_y=[]
    counter=0
    for i in all_mean_acc_max:
        aux_list=list(np.array(all_mean_h)<=i)
        if True in aux_list:
            ind=aux_list.index(True)
            list_time_y.append((list_train_n_eval[ind]/list_train_n_eval[counter])*100)
        else:
            list_time_y.append((max(list_train_n_eval)/list_train_n_eval[counter])*100)
        counter+=1
    print('Time below 100:',sum(np.array(list_time_y)<=100)/len(list_time_y))
    print('Best time:',min(list_time_y),np.mean(list_time_y),np.std(list_time_y))

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    plt.plot(list_train_n_eval, list_time_y,color=colors2[3],label='Time', linewidth=1.2)
    plt.axhline(y=100,color='black', linestyle='--')

    ax.set_xscale('log')
    ax.set_ylabel(r"\textbf{TR}",fontsize=15)
    ax.set_xlabel("$t$",fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.set_ylim([38,117])
    ax.set_yticks([40,70,100])

    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/HeuristicEffectiveness_SymbolicRegressor.pdf')
    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/HeuristicEffectiveness_SymbolicRegressor.png')
    plt.show()
    plt.close()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# List of colors and markers.
colors=px.colors.qualitative.D3
colors2=['#C7C9C8' ,'#AFB0AF','#F4E47B','#DBC432' ]
list_markers = ["o", "^", "s", "P", "v", 'D', "x", "1", "|", "+"]

# Define the number of evaluations to be drawn.
df_max_acc=pd.read_csv('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc1.0.csv', index_col=0)
df_optimal_acc=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristicII.csv', index_col=0)
min_time_acc_max=max(df_max_acc.groupby('train_seed')['n_eval'].min())
min_time_h=max(df_optimal_acc.groupby('train_seed')['n_eval'].min())
max_time_acc_max=min(df_max_acc.groupby('train_seed')['n_eval'].max())
max_time_h=min(df_optimal_acc.groupby('train_seed')['n_eval'].max())

df_cost_per_acc=pd.read_csv('results/data/SymbolicRegressor/UnderstandingAccuracy/df_Bisection.csv',index_col=0)
time_split=list(df_cost_per_acc['cost_per_eval'])[0]*1000

list_train_n_eval=np.arange(max(min_time_acc_max,min_time_h),min(max_time_acc_max,max_time_h)+time_split,time_split)

# Build and save graph.
draw_and_save_figures_per_heuristic()
draw_heuristic_effectiveness()

