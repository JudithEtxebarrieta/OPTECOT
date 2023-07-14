'''
This script is an adaptation of script experimentScripts_WindFLO/OptimalAccuracyAnalysis_figures.py. 
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
    for train_time in list_train_time:

        # Indexes of rows with training time less than train_time.
        ind_train=df_train[type_eval] <= train_time
  
        # Group the previous rows by seed and keep the row per group that has the highest score value associated with it.
        interest_rows=df_train[ind_train].groupby("seed")["score"].idxmax()

        # Calculate the mean and confidence interval of the score.
        interest=list(df_train[ind_train].loc[interest_rows]['score'])
        mean,q05,q95=bootstrap_mean_and_confidence_interval(interest)

        # Save data.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95,list_train_time

def draw_accuracy_behaviour(df_train,type_time):
    '''Construct curve showing the accuracy behavior during the execution process.'''

    # Define relationship between accuracy (term used in the code) and cost (term used in the paper).
    def a_c(a,a_0,a_1):
        c=(a-a_0)/(a_1-a_0)
        return c
    a_0=0.001 
    a_1=1

    # # Initialize lists for the graph.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Fill in lists.
    for train_time in list_train_time:

        # Row indexes with the closest training time to train_time.
        ind_down=df_train[type_time] <= train_time
        ind_per_seed=df_train[ind_down].groupby('seed')[type_time].idxmax()

        # Group the previous rows by seeds and keep the accuracy values.
        list_acc=list(df_train[ind_down].loc[ind_per_seed]['accuracy'])

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
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[1])
    plt.plot(list_train_time, all_mean, linewidth=1,label=r'\textbf{OPTECOT}',color=colors[1],marker=list_markers[1],markevery=0.1)


def draw_and_save_figures_per_heuristic():
    '''Draw and save graphs associated with the selected heuristic.'''

    global ax

    # Initialize graph.
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'
    plt.figure(figsize=[6,10])
    plt.subplots_adjust(left=0.21,bottom=0.09,right=0.9,top=0.94,wspace=0.24,hspace=0.17)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: best solution quality during execution process.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(211)
    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')

    # Constant use of accuracy (default situation).
    all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df_max_acc,'elapsed_time')
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[0])
    plt.plot(list_train_time, all_mean, linewidth=1,label=r'\textbf{Original}',color=colors[0],marker=list_markers[0],markevery=0.1)

    # Optimal use of accuracy (heuristic application).
    df=df_optimal_acc[df_optimal_acc['heuristic_param']=='[5, 3]']
    all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df,'elapsed_time')
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[1])
    plt.plot(list_train_time, all_mean, linewidth=1,label=r'\textbf{OPTECOT}',color=colors[1],marker=list_markers[1],markevery=0.1)


    ax.set_ylabel(r"\textbf{Solution quality}",fontsize=23)
    ax.legend(fontsize=18,title_fontsize=18,labelspacing=0.1,handlelength=0.8,loc='lower right')
    ax.set_title(r'\textbf{Turbines}',fontsize=23)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    ax.set_xscale('log')

    #----------------------------------------------------------------------------------------------
    # GRAPH 2: Graphical representation of the accuracy behavior.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(212)
    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')

    # Draw curve.
    df=df_optimal_acc[df_optimal_acc['heuristic_param']=='[5, 3]']
    draw_accuracy_behaviour(df,'elapsed_time')

    ax.ticklabel_format(style='sci', axis='x', useOffset=True, scilimits=(0,0))
    ax.set_xticks(range(500,4000,1000))
    ax.set_xlabel("$t$",fontsize=23)
    ax.set_ylabel("$c$",fontsize=23)
    ax.set_title('')
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    ax.set_xscale('log')

    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/OptimalAccuracyAnalysis_Turbines.pdf')
    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/OptimalAccuracyAnalysis_Turbines.png')
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

    all_mean_acc_max,_,_,_=train_data_to_figure_data(df_max_acc,'elapsed_time')
    all_mean_h,_,_,_=train_data_to_figure_data(df_optimal_acc[df_optimal_acc['heuristic_param']=='[5, 3]'],'elapsed_time')

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: Curve of quality increment.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(411)

    quality_percentage=list((np.array(all_mean_h)/np.array(all_mean_acc_max))*100)
    print('Quality above 100:',sum(np.array(quality_percentage)>100)/len(quality_percentage))
    print('Best quality:',max(quality_percentage),np.mean(quality_percentage),np.std(quality_percentage))

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    plt.plot(list_train_time,quality_percentage,color=colors2[3],linewidth=1.2)
    plt.axhline(y=100,color='black', linestyle='--')
    ax.set_title(r'\textbf{Turbines}',fontsize=15)
    ax.set_ylabel(r"\textbf{QI}",fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xscale('log')
    ax.set_ylim([99.5,103.5])
    ax.set_yticks([100,101,102,103])

    #----------------------------------------------------------------------------------------------
    # GRAPH 2: Difference between the mean quality curves associated with heuristic application 
    # and original situation.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(4,1,(2,3))
    plt.gca().add_patch(Rectangle((2000, .52), 3600-2000, 0.524-0.52,facecolor='white',edgecolor='black'))
    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    plt.plot(list_train_time, all_mean_acc_max, linewidth=1,label=r'\textbf{Original}',color=colors[0])
    plt.plot(list_train_time, all_mean_h, linewidth=1,label=r'\textbf{OPTECOT}',color=colors[1],linestyle = '--')
    ax.fill_between(list_train_time, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)>np.array(all_mean_acc_max),facecolor='green', alpha=.2,interpolate=True,label=r'\textbf{Improvement}')
    ax.fill_between(list_train_time, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)<np.array(all_mean_acc_max),facecolor='red', alpha=.2,interpolate=True,label=r'\textbf{Worsening}')

    ax.set_ylabel(r"\textbf{Solution quality}",fontsize=15)
    ax.legend(fontsize=11,title_fontsize=11,labelspacing=0.01,handlelength=1)
    plt.yticks(fontsize=15)
    ax.set_xscale('log')

    #----------------------------------------------------------------------------------------------
    # GRAPH 3: Curve of time saving.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(414)

    list_time_y=[]
    counter=0
    for i in all_mean_acc_max:
        aux_list=list(np.array(all_mean_h)>=i)
        if True in aux_list:
            ind=aux_list.index(True)
            list_time_y.append((list_train_time[ind]/list_train_time[counter])*100)
        else:
            list_time_y.append(100)
        counter+=1
    print('Time below 100:',(sum(np.array(list_time_y)<100)+1)/len(list_time_y))
    print('Best time:',min(list_time_y),np.mean(list_time_y),np.std(list_time_y))

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    plt.plot(list_train_time, list_time_y,color=colors2[3],label='Time',linewidth=1.2)
    plt.axhline(y=100,color='black', linestyle='--')

    ax.set_ylabel(r"\textbf{TS}",fontsize=15)
    ax.set_xlabel("$t$",fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.set_xscale('log')
    ax.set_ylim([45,130])
    ax.set_yticks([50,75,100,125])

    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/HeuristicEffectiveness_Turbines.pdf')
    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/HeuristicEffectiveness_Turbines.png')
    plt.show()
    plt.close()

    #----------------------------------------------------------------------------------------------
    # ZOOM OF GRAPH 2
    #----------------------------------------------------------------------------------------------
    plt.figure(figsize=[2,1.5])
    plt.subplots_adjust(left=0.28,bottom=0.62,right=0.6,top=0.87,wspace=0.3,hspace=0.07)
    ax=plt.subplot(111)

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    plt.plot(list_train_time, all_mean_acc_max, linewidth=1,label='Original',color=colors[0])
    plt.plot(list_train_time, all_mean_h, linewidth=1,label='Heuristic',color=colors[1],linestyle = '--')
    ax.fill_between(list_train_time, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)>np.array(all_mean_acc_max),facecolor='green', alpha=.2,interpolate=True,label='Improvement')
    ax.fill_between(list_train_time, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)<np.array(all_mean_acc_max),facecolor='red', alpha=.2,interpolate=True,label='Worsening')
    
    ax.set_xticks([2500])
    ax.ticklabel_format(style='sci', axis='x', useOffset=True, scilimits=(0,0))
    plt.yticks(fontsize=11)
    plt.xticks(fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim([2000,3600])
    ax.set_ylim([0.52,0.524])

    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/HeuristicEffectiveness_TurbinesZoom.pdf')
    plt.savefig('figures_paper/figures/OptimalAccuracyAnalysis/HeuristicEffectiveness_TurbinesZoom.png')
    plt.show()
    plt.close()


#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# List of scores and markers.
colors=px.colors.qualitative.D3
list_markers = ["o", "^", "s", "P", "v", 'D', "x", "1", "|", "+"]
colors1=['#C7C9C8' ,'#AFB0AF' ]
colors2=['#C7C9C8' ,'#AFB0AF','#F4E47B','#DBC432' ]

# Define training times to be drawn.
df_max_acc=pd.read_csv('results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc1.0.csv', index_col=0)
df_optimal_acc=pd.read_csv('results/data/Turbines/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_hII.csv', index_col=0)
min_time_acc_max=max(df_max_acc.groupby('seed')['elapsed_time'].min())
min_time_hII=max(df_optimal_acc.groupby('seed')['elapsed_time'].min())
max_time_acc_max=min(df_max_acc.groupby('seed')['elapsed_time'].max())
max_time_hII=min(df_optimal_acc.groupby('seed')['elapsed_time'].max())

df_cost_per_acc=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/df_Bisection.csv',index_col=0)
time_split=list(df_cost_per_acc['cost_per_eval'])[0]*20

list_train_time=np.arange(max(min_time_acc_max,min_time_hII),3600+time_split,time_split)

# Build and save graphs.
draw_and_save_figures_per_heuristic()
draw_heuristic_effectiveness()
