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

def train_data_to_figure_data(df_train,type_eval):
    '''
    The database associated with the heuristic execution information is summarized to the data
    needed to construct the solution quality curve. 

    Parameters
    ==========
    df_train: Database with the heuristic execution information.
    type_eval: Type of evaluation to be represented in the graph ('n_eval': all evaluations; 
    'n_eval_proc': only the evaluations spent for the calculation of the scores per generation 
    without the extra evaluations used for the accuracy adjustment).

    Return
    ====== 
    all_mean: Averages of the scores per limit of number of training evaluations set in list_train_n_eval.
    all_q05,all_q95: Percentiles of the scores by limit of training evaluations.
    list_train_n_eval: List with number of evaluations to be plotted in the graph.
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

def draw_accuracy_behaviour(df_train,type_n_eval,curve):
    '''
    Construction of curves showing the behavior of the accuracy during training.

    Parameters
    ==========
    df_train: Database with training data.
    type_n_eval: Type of evaluation to be represented in the graph ('n_eval': all evaluations; 
    n_eval_proc': only the evaluations spent for the calculation of the scores per generation without 
    the extra evaluations used for the accuracy adjustment). 
    curve: Number indicating which curve is being drawn.
    '''

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
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,color=colors[curve])


def draw_and_save_figures_per_heuristic(heuristic):
    '''
    Function to draw and save the graphs (with solution quality curves and accuracy behavior) 
    according to the selected heuristic.
    '''

    global ax

    # Initialize graph.
    plt.figure(figsize=[20,5])
    plt.subplots_adjust(left=0.08,bottom=0.11,right=0.76,top=0.88,wspace=0.4,hspace=0.76)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: Solution quality during training using the total number of evaluations.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(132)

    # read databases.
    df_max_acc=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc1.0.csv", index_col=0) # Accuracy constant 1 (default situation).
    df_optimal_acc=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0) # Optimal accuracy (heuristic application).

    # Initialize number of curves.
    curve=0

    # Constant use of accuracy (default situation).
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Default',color=colors[curve])
    curve+=1

    # Optimal use of accuracy (heuristic application).
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df,'n_eval')
        ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train total evaluations")
    ax.set_ylabel("Mean score (MAE)")
    ax.set_title('Comparison between optimal and constant accuracy')
    ax.set_xscale('log')

    #----------------------------------------------------------------------------------------------
    # GRAPH 2: Solution quality during training without considering the number of extra 
    # evaluations needed to readjust the accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(133)

    # Initialize number of curves.
    curve=0

    # Constant use of accuracy (default situation).
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Default',color=colors[curve])
    curve+=1

    # Optimal use of accuracy (heuristic application).
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df,'n_eval_proc')
        ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train evaluations (without extra)")
    ax.set_ylabel("Mean score (MAE)")
    ax.set_title('Comparison between optimal and constant accuracy')
    ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')
    ax.set_xscale('log')

    #----------------------------------------------------------------------------------------------
    # GRAPH 3: Graphical representation of the accuracy behavior.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(131)

    # Initialize number of curves.
    curve=1

    # Draw curve.
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        draw_accuracy_behaviour(df,'n_eval',curve)
        curve+=1
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    ax.set_xticks(range(200000,800000,200000))
    ax.set_title('Behavior of optimal accuracy')

    # Save graph.
    plt.savefig('results/figures/SymbolicRegressor/OptimalAccuracyAnalysis/OptimalAccuracy_h'+str(heuristic)+'.png')
    plt.show()
    plt.close()


def draw_comparative_figure(heuristic_param_list):
    '''Draw comparative graph (comparison between heuristics I and II).'''
    global ax

    # Initialize graph.
    plt.figure(figsize=[15,5])
    plt.subplots_adjust(left=0.12,bottom=0.11,right=0.73,top=0.88,wspace=0.4,hspace=0.76)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: Solution quality during training using the total number of evaluations.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(122)

    # Constant use of accuracy (default situation).
    df_max_acc=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc1.0.csv", index_col=0)
    curve=0
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Default',color=colors[curve])
    curve+=1

    # Optimal use of accuracy (heuristic application).
    for heuristic_param in heuristic_param_list:
        heuristic=heuristic_param[0]
        param=heuristic_param[1]
        df=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0)
        all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df[df['heuristic_param']==param],'n_eval')
        ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1
    
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Mean score (MAE)")
    ax.set_title('Comparison of heuristics and default case')
    ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')
    ax.set_xscale('log')

    #----------------------------------------------------------------------------------------------
    # GRAPH 3: Graphical representation of the accuracy behavior.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(121)

    # Initialize number of curves.
    curve=1

    # Draw curves.
    for heuristic_param in heuristic_param_list:
        heuristic=heuristic_param[0]
        param=heuristic_param[1]
        df=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0)
        draw_accuracy_behaviour(df[df['heuristic_param']==param],'n_eval',curve)
        curve+=1
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    ax.set_xticks(range(200000,800000,200000))
    ax.set_title('Behavior of optimal accuracy')

    plt.savefig('results/figures/SymbolicRegressor/OptimalAccuracyAnalysis/OptimalAccuracyAnalysis_comparison.png')
    plt.show()
    plt.close()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# List of colors.
colors=px.colors.qualitative.D3
    
# Define number of evaluations to be drawn.
df_max_acc=pd.read_csv('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc1.0.csv', index_col=0)
df_hI=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristicI.csv', index_col=0)
df_hII=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristicII.csv', index_col=0)
min_time_acc_max=max(df_max_acc.groupby('train_seed')['n_eval'].min())
min_time_hI=max(df_hI.groupby('train_seed')['n_eval_proc'].min())
min_time_hII=max(df_hII.groupby('train_seed')['n_eval_proc'].min())
max_time_acc_max=min(df_max_acc.groupby('train_seed')['n_eval'].max())
max_time_hI=min(df_hI.groupby('train_seed')['n_eval_proc'].max())
max_time_hII=min(df_hII.groupby('train_seed')['n_eval_proc'].max())

df_cost_per_acc=pd.read_csv('results/data/SymbolicRegressor/UnderstandingAccuracy/df_Bisection.csv',index_col=0)
time_split=list(df_cost_per_acc['cost_per_eval'])[0]*1000

list_train_n_eval=np.arange(max(min_time_acc_max,min_time_hI,min_time_hII),min(max_time_acc_max,max_time_hI,max_time_hII)+time_split,time_split)


# Independent graphs per heuristic.
list_heuristics=['I','II']
for heuristic in list_heuristics:
    draw_and_save_figures_per_heuristic(heuristic)

# Build comparative graph.
heuristic_param_list=[['I',0.95],['II',"[5, 3]"]]
draw_comparative_figure(heuristic_param_list)