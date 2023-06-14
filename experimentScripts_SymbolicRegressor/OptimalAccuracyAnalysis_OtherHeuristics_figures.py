'''
This script is used to graphically represent the numerical results obtained in 
"OptimalAccuracyAnalysis_OtherHeuristics_data.py".
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

    # Fill in list.
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

    # Fill in list.
    for train_n_eval in list_train_n_eval:

        # Row indexes with the number of training evaluations closest to train_n_eval.
        ind_down=df_train[type_n_eval] <= train_n_eval
        ind_per_seed=df_train[ind_down].groupby('train_seed')[type_n_eval].idxmax()

        # Group the previous rows by seeds and keep the accuracy values.
        list_acc=list(df_train[ind_down].loc[ind_per_seed]['acc'])

        # Calculate the mean and the confidence interval of the accuracy.
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list_acc)

        # Save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Draw graph.
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,color=colors[curve])

def draw_heuristic1_acc_split_functions(list_params):
    '''Graphical representation of the functions that define the accuracy split in heuristic 1.'''

    def acc_split(corr,acc_rest,param):
        if param=='logistic':
            split=(1/(1+np.exp(12*(corr-0.5))))*acc_rest
        else:
            if corr<=param[0]:
                split=acc_rest
            else:
                split=-acc_rest*(((corr-param[0])/(1-param[0]))**(1/param[1]))+acc_rest
        return split

    acc_rest=1
    all_x=np.arange(-1,1.1,0.1)
    curve=1
    for params in list_params:
        if params!='logistic':
            params=params.translate({ord('('):'',ord(')'):''})
            params=tuple(float(i) for i in params.split(','))        
        all_y=[]
        for x in all_x:
            all_y.append(acc_split(x,acc_rest,params))

        ax=plt.subplot2grid((3, 6), (2,curve-1), colspan=1)
        plt.plot(all_x, all_y, linewidth=2,color=colors[curve])
        ax.set_xlabel("Spearman correlation")
        ax.set_ylabel("Accuracy ascent")
        ax.set_title('Heuristic 1 f'+str(curve))
        ax.set_yticks([0,1])
        ax.set_yticklabels([0,"1-accuracy"],rotation=90)

        curve+=1

def draw_heuristic13_14_threshold_shape(df_train,type_n_eval,curve):
    '''Drawing threshold behavior of the bisection method for heuristics 13 and 14.'''

    # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in list.
    for train_n_eval in list_train_n_eval:

        # Row indexes with the number of training evaluations closest to train_n_eval.
        ind_down=df_train[type_n_eval] <= train_n_eval
        ind_per_seed=df_train[ind_down].groupby('train_seed')[type_n_eval].idxmax()

        # Group the previous rows by seeds and keep the threshold values.
        list_threshold=list(df_train[ind_down].loc[ind_per_seed]['threshold'])

        # Calculate the mean and confidence interval of the threshold.
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list_threshold)

        # Save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Draw graph.
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,color=colors[curve])

def draw_and_save_figures_per_heuristic(heuristic):
    '''Draw and save graphs according to selected heuristic.'''

    global ax

    # Initialize graph.
    if heuristic==1:
        plt.figure(figsize=[20,9])
        plt.subplots_adjust(left=0.08,bottom=0.11,right=0.82,top=0.88,wspace=0.98,hspace=0.76)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12]:
        plt.figure(figsize=[20,5])
        plt.subplots_adjust(left=0.08,bottom=0.11,right=0.76,top=0.88,wspace=0.4,hspace=0.76)
    if heuristic in [13,14]:
        plt.figure(figsize=[22,5])
        plt.subplots_adjust(left=0.04,bottom=0.11,right=0.87,top=0.88,wspace=0.37,hspace=0.76)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: Solution quality during training using the total number of evaluations.
    #----------------------------------------------------------------------------------------------

    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,2), colspan=2,rowspan=2)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12]:
        ax=plt.subplot(132)
    if heuristic in [13,14]:
        ax=plt.subplot(143)

    # Reading of databases to be used.
    df_max_acc=pd.read_csv("results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc1.0.csv", index_col=0) # Default (accuracy 1).
    df_optimal_acc=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis_OtherHeuristics/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0) # Heuristic.

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
        if heuristic==1:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' f'+str(list_params.index(param)+1),color=colors[curve])
        else:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train total evaluations")
    ax.set_ylabel("Mean score (MAE)")
    ax.set_xscale('log')
    if heuristic in [7,8,9,10,11,12,13,14]:
        ax.set_title('Comparison between optimal and constant accuracy')
    else:
        ax.set_title('Comparison between ascendant and constant accuracy')

    #----------------------------------------------------------------------------------------------
    # GRAPH 2: Solution quality during training without considering the number of extra 
    # evaluations needed to readjust the accuracy.
    #----------------------------------------------------------------------------------------------
    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,4), colspan=2,rowspan=2)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12]:
        ax=plt.subplot(133)
    if heuristic in [13,14]:
        ax=plt.subplot(144)

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
        if heuristic==1:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' f'+str(list_params.index(param)+1),color=colors[curve])
        else:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train evaluations (without extra)")
    ax.set_ylabel("Mean score (MAE)")
    ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')
    ax.set_xscale('log')
    if heuristic in [7,8,9,10,11,12,13,14]:
        ax.set_title('Comparison between optimal and constant accuracy')
    else:
        ax.set_title('Comparison between ascendant and constant accuracy')

    #----------------------------------------------------------------------------------------------
    # GRAPH 3: Graphical representation of the accuracy behavior.
    #----------------------------------------------------------------------------------------------
    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,0), colspan=2,rowspan=2)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12]:
        ax=plt.subplot(131)
    if heuristic in [13,14]:
        ax=plt.subplot(142)

    # Initialize number of curves.
    curve=1

    # Draw curves.
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        draw_accuracy_behaviour(df,'n_eval',curve)
        curve+=1
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    ax.set_xticks(range(200000,800000,200000))
    if heuristic in [7,8,9,10,11,12,13,14]:
        ax.set_title('Behavior of optimal accuracy')
    else:
        ax.set_title('Ascendant behavior of accuracy')

    #----------------------------------------------------------------------------------------------
    # GRAPH 4: Graphical representation of the correlation threshold behavior.
    #----------------------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                           
    if heuristic==1:
        # Draw functions considered to define the rise of the accuracy.
        draw_heuristic1_acc_split_functions(list_params)

    if heuristic in[13,14]:
        ax=plt.subplot(141)

        # Initialize number of curves.
        curve=1

        # Draw curves.
        for param in list_params:
            df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
            draw_heuristic13_14_threshold_shape(df,'n_eval',curve)
            curve+=1
        ax.set_xlabel("Train evaluations")
        ax.set_ylabel("Threshold value")
        ax.set_title('Behavior of bisection method threshold')

    plt.savefig('results/figures/SymbolicRegressor/OptimalAccuracyAnalysis_OtherHeuristics/OptimalAccuracy_h'+str(heuristic)+'.png')
    plt.show()
    plt.close()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# List of colors.
colors=px.colors.qualitative.D3

# Draw and save graphs.
list_heuristics=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
for heuristic in list_heuristics:
    # Define the number of evaluations to be drawn.
    list_train_n_eval=range(50000,1000000,10000)
    # Call function.
    draw_and_save_figures_per_heuristic(heuristic)
