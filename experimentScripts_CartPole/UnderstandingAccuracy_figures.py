'''
This script is used to graphically represent the numerical results obtained in 
"UnderstandingAccuracy_data.py".
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
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import scipy as sc
from itertools import combinations

#==================================================================================================
# FUNCTIONS
#==================================================================================================
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

def remove_element_in_idx(list_elements,list_idx):
    '''Delete elements at certain positions in a list.'''
    new_list=[]
    for i in range(len(list_elements)):
        if i not in list_idx:
            new_list.append(list_elements[i])
    return new_list

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

def inverse_normalized_tau_kendall(x,y):
    '''Calculation of the normalized inverse tau kendall distance between two rankings.'''
    # Number of pairs with reverse order.
    pairs_reverse_order=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            case1 = x[i] < x[j] and y[i] > y[j]
            case2 = x[i] > x[j] and y[i] < y[j]

            if case1 or case2:
                pairs_reverse_order+=1  
    
    # Number of total pairs.
    total_pairs=len(list(combinations(x,2)))
    # Normalized tau kendall distance.
    tau_kendall=pairs_reverse_order/total_pairs

    return 1-tau_kendall

def from_data_to_figures(df):
    '''Construct the desired graphs from the database.'''

    # Initialize graph.
    plt.figure(figsize=[15,10])
    plt.subplots_adjust(left=0.06,bottom=0.11,right=0.95,top=0.88,wspace=0.3,hspace=0.69)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: evaluation time per accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot2grid((4, 4), (0,0), colspan=2,rowspan=2)

    list_acc=list(set(df['accuracy']))
    list_acc.sort()

    all_means=[]
    all_q05=[]
    all_q95=[]

    for accuracy in list_acc:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list(df[df['accuracy']==accuracy]['steps']))
        all_means.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    ax.fill_between(list_acc,all_q05,all_q95, alpha=.5, linewidth=0)
    plt.plot(list_acc, all_means, linewidth=2) 
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Steps per evaluation")
    ax.set_title('Evaluation cost depending on accuracy')


    #----------------------------------------------------------------------------------------------
    # GRAPH 2: extra evaluations.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot2grid((4, 4), (0,2), colspan=2,rowspan=2)

    list_total_times=list(df.groupby('accuracy')['steps'].sum())

    list_extra_eval=[]
    for i in range(0,len(list_total_times)):
        # Time saving.
        save_time=list_total_times[-1]-list_total_times[i]
        # Number of extra evaluations that can be done to exhaust the default time needed to evaluate the entire sample.
        extra_eval=int(save_time/all_means[i])
        list_extra_eval.append(extra_eval)

    plt.bar([str(acc) for acc in list_acc],list_extra_eval)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Number of extra evaluations')
    plt.xticks(rotation = 0)
    ax.set_title('Extra evaluations in the same amount of time required for maximum accuracy')


    #----------------------------------------------------------------------------------------------
    # GRAPH 3: comparison of rankings associated with the 100 solutions with each accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(223)

    def ranking_matrix_sorted_by_max_acc(ranking_matrix):
        '''Reorder ranking matrix according to the order established by the original ranking.'''

        # Indexes associated with the maximum accuracy ranking ordered from lowest to highest.
        argsort_max_acc=np.argsort(ranking_matrix[-1])

        # Reorder matrix rows.
        sorted_ranking_list=[]
        for i in range(len(ranking_list)):
            sorted_ranking_list.append(custom_sort(ranking_list[i],argsort_max_acc))

        return sorted_ranking_list

    ranking_list=[]
    for accuracy in list_acc:
        df_acc=df[df['accuracy']==accuracy]
        ranking=from_argsort_to_ranking(list(df_acc['reward'].argsort()))
        ranking_list.append(ranking)
    ranking_matrix=np.matrix(ranking_matrix_sorted_by_max_acc(ranking_list))

    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,ranking_matrix.shape[1]))
    color[:1, :]=np.array([248/256, 67/256, 24/256, 1])# Red (rgb code)
    color = ListedColormap(color)

    ax = sns.heatmap(ranking_matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    colorbar=ax.collections[0].colorbar
    colorbar.set_label('Ranking position')
    colorbar.set_ticks(range(0,ranking_matrix.shape[1],10))
    colorbar.set_ticklabels(range(1,ranking_matrix.shape[1]+1,10))

    ax.set_xlabel('Solution')
    ax.set_xticks(range(0,ranking_matrix.shape[1],10))
    ax.set_xticklabels(range(1,ranking_matrix.shape[1]+1,10),rotation=0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparing rankings depending on accuracy')
    ax.set_yticks(np.arange(0.5,len(list_acc)+0.5,1))
    ax.set_yticklabels(list_acc,rotation=0)

    #----------------------------------------------------------------------------------------------
    # GRAPH 4: solution quality loss.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot2grid((4, 4), (2,2), colspan=2,rowspan=2)

    # Similarity between entire rankings.
    list_corr=[]
    for i in range(ranking_matrix.shape[0]):
        list_corr.append(inverse_normalized_tau_kendall(ranking_list[-1],ranking_list[i]))

    plt.plot(list_acc, list_corr, linewidth=2,label='All ranking') 

    # Similarity between the initial 50% of the rankings.
    list_ind=[]
    for i in range(int(ranking_matrix.shape[1]*0.5),ranking_matrix.shape[1]):
        ind=ranking_list[-1].index(i)
        list_ind.append(ind)

    list_corr=[]
    for i in range(ranking_matrix.shape[0]):
        best_ranking=remove_element_in_idx(ranking_list[-1],list_ind)
        lower_ranking=remove_element_in_idx(ranking_list[i],list_ind)
        list_corr.append(inverse_normalized_tau_kendall(best_ranking,lower_ranking))

    plt.plot(list_acc, list_corr, linewidth=2,label='The best 50%') 

    # Similarity between the initial 10% of the rankings.
    list_ind=[]
    for i in range(int(ranking_matrix.shape[1]*0.1),ranking_matrix.shape[1]):
        ind=ranking_list[-1].index(i)
        list_ind.append(ind)

    list_corr=[]
    for i in range(ranking_matrix.shape[0]):
        best_ranking=remove_element_in_idx(ranking_list[-1],list_ind)
        lower_ranking=remove_element_in_idx(ranking_list[i],list_ind)
        list_corr.append(inverse_normalized_tau_kendall(best_ranking,lower_ranking))

    plt.plot(list_acc, list_corr, linewidth=2,label='The best 10%') 
    ax.legend(title="Similarity between")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("1 - normalized tau Kendall")
    ax.set_title('Comparing the similarity between the best and the rest rankings')

    # Save graphs.
    plt.savefig('results/figures/CartPole/UnderstandingAccuracy.png')
    plt.show()
    plt.close()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Read database.
df=pd.read_csv('results/data/CartPole/UnderstandingAccuracy.csv',index_col=0)

# Build the graph.
from_data_to_figures(df)
