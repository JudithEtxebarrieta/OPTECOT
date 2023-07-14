'''
This script is an adaptation of script experimentScripts_Turbines/UnderstandingAccuracy_figures.py. 
The design of the original graphs is modified to insert them in the paper.
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
import matplotlib as mpl
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from itertools import combinations
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

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

def from_argsort_to_ranking(list):
    '''Obtain ranking from a list get after applying "np.argsort" on an original list.'''
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

def custom_sort(list,argsort):
    '''Sort list according to the order for each position defined in argsort.'''
    new_list=[]
    for index in argsort:
        new_list.append(list[index])
    return new_list

def from_data_to_figures_paper(df,blade_number=None):
    '''Construct the desired graphs from the database.'''

    # Initialize graph.
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'
    plt.figure(figsize=[5,5],constrained_layout=True)

    def convert_textbf(x):
        return [r'\textbf{'+str(i)+'}' for i in x]

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: evaluation time per accuracy and extra evaluations.
    #----------------------------------------------------------------------------------------------
    # Define relationship between accuracy (term used in the code) and cost (term used in the paper).
    def a_c(a,a_0,a_1):
        c=(a-a_0)/(a_1-a_0)
        return c
    list_acc=list(set(df['accuracy']))
    list_acc.sort()
    a_0=0.1 
    a_1=1
    list_acc_str=[str(round(a_c(acc,a_0,a_1),2)) for acc in list_acc]
    list_acc_str.reverse()
    list_acc_float=[round(a_c(acc,a_0,a_1),2) for acc in list_acc]

    # Execution times per evaluation.
    all_means=[]
    all_q05=[]
    all_q95=[]

    for accuracy in list_acc:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list(df[df['accuracy']==accuracy]['time']))
        all_means.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    # Extra evaluations.
    list_extra_eval=[]
    for i in range(0,len(all_means)):
        # Time saving.
        save_time=all_means[-1]
        # Number of extra evaluations that can be done to exhaust the default time needed to evaluate the entire sample.
        extra_eval=save_time/all_means[i]
        list_extra_eval.append(extra_eval)

    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,3))

    ax=plt.subplot(221)
    ax.set_title(r'\textbf{Turbines}',fontsize=16)
    y=[i*(-1) for i in all_means]
    y.reverse()
    plt.barh(convert_textbf(list_acc_str), y, align='center',color=color[0])
    plt.ylabel("$c$",fontsize=16)
    plt.xlabel("$t_c$",fontsize=16)
    ax.set_xticks(np.arange(0,-4,-1))
    ax.set_xticklabels(convert_textbf(np.arange(0,4,1)),rotation=0,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(b=True, color=color[0],linestyle='-', linewidth=0.8,alpha=0.2,axis='x')
    

    ax=plt.subplot(222)
    for i in range(len(list_extra_eval)):
        if list_extra_eval[i]<0:
            list_extra_eval[i]=0
    list_extra_eval.reverse()
    plt.barh(list_acc_str, list_extra_eval, align='center',color=color[1])
    plt.yticks([])
    ax.set_xticks(range(0,4,1))
    plt.xlabel("$t_1/t_c$",fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(b=True, color=color[1],linestyle='-', linewidth=0.8,alpha=0.2,axis='x')

    #----------------------------------------------------------------------------------------------
    # GRAPH 2: Loss of quality in the evaluations when considering lower accuracies and existence 
    # of a lower accuracy with which the maximum quality is obtained.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(2,2,(3,4))

    def ranking_matrix_sorted_by_max_acc(ranking_matrix):
        '''Reorder ranking matrix according to the order established by the original ranking.'''

        # Indexes associated with the maximum accuracy ranking ordered from lowest to highest.
        argsort_max_acc=np.argsort(ranking_matrix[-1])

        # Reorder matrix rows.
        sorted_ranking_list=[]
        for i in range(len(ranking_list)):
            sorted_ranking_list.append(custom_sort(ranking_list[i],argsort_max_acc))

        return sorted_ranking_list

    def absolute_distance_matrix_between_rankings(ranking_matrix):
        '''Calculate matrix of normalized absolute distances between the positions of the rankings.'''
        abs_dist_matrix=[]
        for i in range(len(ranking_matrix)):
            abs_dist_matrix.append(np.abs(list(np.array(ranking_matrix[i])-np.array(ranking_matrix[-1]))))

        max_value=max(max(i) for i in abs_dist_matrix)
        norm_abs_dist_matrix=list(np.array(abs_dist_matrix)/max_value)
        
        return norm_abs_dist_matrix

    ranking_list=[]

    for accuracy in list_acc:
        df_acc=df[df['accuracy']==accuracy]
        ranking=from_argsort_to_ranking(list(df_acc['score'].argsort()))
        ranking_list.append(ranking)

    ranking_matrix=np.matrix(absolute_distance_matrix_between_rankings(ranking_matrix_sorted_by_max_acc(ranking_list)))
    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)

    ax = sns.heatmap(ranking_matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    colorbar=ax.collections[0].colorbar
    colorbar.set_label(r'\textbf{Normalized distance}',fontsize=16)
    colorbar.ax.set_yticklabels(colorbar.ax.get_yticklabels(), fontsize=12,rotation=90, va='center')

    ax.set_xlabel(r'\textbf{Solution}',fontsize=16)
    ax.set_xticks(range(0,ranking_matrix.shape[1],10))
    ax.set_xticklabels(convert_textbf(range(1,ranking_matrix.shape[1]+1,10)),rotation=0,fontsize=16)
    plt.ylabel("$c$",fontsize=16)
    ax.set_yticks(np.arange(0.5,len(list_acc)+0.5,1))
    ax.set_yticklabels(convert_textbf(list_acc_float),rotation=0,fontsize=16)

    # Save graph.
    plt.savefig('figures_paper/figures/UnderstandingAccuracy/UnderstandingAccuracy_Turbines.png')
    plt.savefig('figures_paper/figures/UnderstandingAccuracy/UnderstandingAccuracy_Turbines.pdf')

    plt.show()
    plt.close()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
df=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII.csv',index_col=0)
from_data_to_figures_paper(df)

