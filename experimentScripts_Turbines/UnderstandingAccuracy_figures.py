'''
This script is used to graphically represent the numerical results obtained in 
"UnderstandingAccuracy_data.py".
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

def form_str_col_to_float_list_col(str_col,type_elems):
    '''
    Convert string elements of a database column to list form.

    Parameters
    ==========
    str_col: Database column whose elements are originally lists of numbers (float or integer), but are 
    considered as strings when the database is saved and read. 
    type_elems: Type of numbers that keep the lists of the columns, float or integer.

    Return
    ======
    Database column transformed to the appropriate format.

    '''

    float_list_col=[]
    for str in str_col:
        # Remove string brackets.
        str=str.replace('[','')
        str=str.replace(']','')

        # Convert str to float list.
        float_list=str.split(", ")
        if len(float_list)==1:
            float_list=str.split(" ")
        
        # Accumulate conversions.
        if type_elems=='float':
            float_list=[float(i) for i in float_list]
        if type_elems=='int':
            float_list=[int(i) for i in float_list]
        float_list_col.append(float_list)

    return float_list_col

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
    
    # Total number of pairs.
    total_pairs=len(list(combinations(x,2)))
    # Normalized tau Kendall distance.
    tau_kendall=pairs_reverse_order/total_pairs

    return 1-tau_kendall

def remove_element_in_idx(list_elements,list_idx):
    '''Delete items at certain positions in a list.'''
    new_list=[]
    for i in range(len(list_elements)):
        if i not in list_idx:
            new_list.append(list_elements[i])
    return new_list


def from_data_to_figuresI(df):
    '''Build first motivation graph.'''

    # Change columns containing lists to proper formatting.
    df['all_scores']=form_str_col_to_float_list_col(df['all_scores'],'float')
    df['ranking']=form_str_col_to_float_list_col(df['ranking'],'int')
    df['all_times']=form_str_col_to_float_list_col(df['all_times'],'float')

    # Initialize figure (size and margins).
    plt.figure(figsize=[12,9])
    plt.subplots_adjust(left=0.09,bottom=0.11,right=0.95,top=0.88,wspace=0.3,hspace=0.4)

    #--------------------------------------------------------------------------------------------------
    # GRAPH 1: Impact of the accuracy of N on the time required to make an evaluation.
    #--------------------------------------------------------------------------------------------------
    x=df['N']
    y_mean=[]
    y_q05=[]
    y_q95=[]
    for times in df['all_times']:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(times)
        y_mean.append(mean)
        y_q05.append(q05)
        y_q95.append(q95)

    ax1=plt.subplot(221)
    ax1.fill_between(x,y_q05,y_q95,alpha=.5,linewidth=0)
    plt.plot(x,y_mean,linewidth=2)
    ax1.set_xlabel('N')
    ax1.set_ylabel('Time per evaluation')
    ax1.set_title('Evaluation time depending on N')

    #Zoom
    axins = zoomed_inset_axes(ax1,6,loc='upper left',borderpad=1)
    axins.fill_between(x,y_q05,y_q95,alpha=.5,linewidth=0)
    plt.plot(x,y_mean,linewidth=2)
    axins.set_xlim(0,60)
    axins.set_ylim(2,7)
    axins.yaxis.tick_right()

    #--------------------------------------------------------------------------------------------------
    # GRAPH 2: Impact of the accuracy of N on the optimization problem solution quality 
    # (comparing rankings).
    #--------------------------------------------------------------------------------------------------
    x=range(1,len(df['all_scores'][0])+1)
    ax2=plt.subplot(224)

    matrix=[]
    for i in df['ranking']:
        matrix.append(i)
    matrix=np.matrix(matrix)

    color = sns.color_palette("deep", len(df['ranking'][0]))
    ax2 = sns.heatmap(matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    colorbar=ax2.collections[0].colorbar
    colorbar.set_label('Ranking position', rotation=270)
    colorbar.set_ticks(np.arange(.5,len(df['ranking'][0])-.9,.9))
    colorbar.set_ticklabels(range(len(df['ranking'][0]),0,-1))

    ax2.set_xlabel('Turbine design')
    ax2.set_xticklabels(range(1,len(df['ranking'][0])+1))

    ax2.set_ylabel('N')
    ax2.set_yticks(np.arange(0.5,df.shape[0]+0.5))
    ax2.set_yticklabels(df['N'],rotation=0)

    ax2.set_title('Comparing rankings depending on N')

    #--------------------------------------------------------------------------------------------------
    # GRAPH 3: Impact of the accuracy of N on the optimization problem solution quality 
    # (comparing score losses).
    #--------------------------------------------------------------------------------------------------
    x=[str(i) for i in df['N']]
    y=[]
    best_scores=df['all_scores'][0]
    for i in range(0,df.shape[0]):
        ind_best_turb=df['ranking'][i].index(max(df['ranking'][i]))
        quality_loss=(max(best_scores)-best_scores[ind_best_turb])/max(best_scores)
        y.append(quality_loss)


    ax3=plt.subplot(223)
    plt.scatter(x,y)
    ax3.set_xlabel('N')
    ax3.set_ylabel('Score loss (%)')
    plt.xticks(rotation = 45)
    ax3.set_title('Comparing loss of score quality depending on N')


    #--------------------------------------------------------------------------------------------------
    # GRAPH 4: Extra evaluations that can be made when considering a less accurate N.
    #--------------------------------------------------------------------------------------------------
    x=[str(i) for i in df['N']]
    y=[]
    for i in range(0,df.shape[0]):
        extra_eval=(df['total_time'][0]-df['total_time'][i])/df['time_per_eval'][i]
        y.append(extra_eval)

    ax4=plt.subplot(222)
    plt.bar(x,y)
    ax4.set_xlabel('N')
    ax4.set_ylabel('Number of extra evaluations')
    plt.xticks(rotation = 45)
    ax4.set_title('Extra evaluations in the same time required by maximum N')

    plt.savefig('results/figures/Turbines/UnderstandingAccuracyAnalysis/UnderstandingAccuracyI.png')
    plt.show()


def from_data_to_figuresII(df,blade_number):
    '''Build second motivation graph.'''

    # Initialize graph.
    plt.figure(figsize=[20,10])
    plt.subplots_adjust(left=0.06,bottom=0.11,right=0.95,top=0.88,wspace=0.3,hspace=0.69)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: evaluation time per accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(311)

    list_acc=list(set(df['accuracy']))
    list_acc.sort()


    all_means=[]
    all_q05=[]
    all_q95=[]

    for accuracy in list_acc:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list(df[df['accuracy']==accuracy]['time']))
        all_means.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    ax.fill_between(list_acc,all_q05,all_q95, alpha=.5, linewidth=0)
    plt.plot(list_acc, all_means, linewidth=2) 
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Time per evaluation")
    ax.set_title('Evaluation cost depending on accuracy')


    #----------------------------------------------------------------------------------------------
    # GRAPH 2: extra evaluations.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(312)

    list_extra_eval=[]
    for i in range(0,len(all_means)):
        # Time saving.
        save_time=all_means[-1]-all_means[i]
        # Number of extra evaluations that could be done to exhaust the default time needed.
        extra_eval=save_time/all_means[i]
        if extra_eval<0:
            extra_eval=0
        list_extra_eval.append(extra_eval)

    plt.bar([str(acc) for acc in list_acc],list_extra_eval)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Number of extra evaluations')
    plt.xticks(rotation = 45)
    ax.set_title('Extra evaluations in the same amount of time required for maximum accuracy')

    #----------------------------------------------------------------------------------------------
    # GRAPH 3: null solutions per accuracy value.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(313)

    list_null_scores=[]
    for accuracy in list_acc:
        list_null_scores.append(sum(np.array(list(df[df['accuracy']==accuracy]['score']))==0)/100)

    plt.bar([str(acc) for acc in list_acc],list_null_scores)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Percentaje of null solutions')
    plt.xticks(rotation = 45)
    ax.set_title('Presence of null solutions')


    # Save graph.
    plt.savefig('results/figures/Turbines/UnderstandingAccuracyAnalysis/UnderstandingAccuracyII_nullsolutions_bladenumber'+str(blade_number)+'.png')
    plt.show()
    plt.close()


def from_data_to_figures_bladenumber(df,blade_number):
    '''Construct graphs to help decide the blade-number value to be considered.'''

    # Initialize graph.
    fig=plt.figure(figsize=[15,15],constrained_layout=True)
    subfigs = fig.subfigures(1, 4, width_ratios=[2,2,0.1,4], wspace=0)

    #----------------------------------------------------------------------------------------------
    # GRAPH 1: execution time per accuracy and extra evaluations.
    #----------------------------------------------------------------------------------------------
    list_acc=list(set(df['accuracy']))
    list_acc.sort()
    list_acc_str=[str(acc) for acc in list_acc]
    list_acc_str.reverse()

    # Evaluation times.
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
        save_time=all_means[-1]-all_means[i]
        # Number of extra evaluations that could be done to exhaust the default time needed.
        extra_eval=save_time/all_means[i]
        list_extra_eval.append(extra_eval)

    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,3))

    ax=subfigs[0].subplots()
    y=[i*(-1) for i in all_means]
    y.reverse()
    plt.barh(list_acc_str, y, align='center',color=color[0])
    plt.title("\n Cost per evaluation")
    plt.ylabel("Accuracy")
    plt.xlabel("Time")
    ax.set_xticks(np.arange(0,-4,-1))
    ax.set_xticklabels(np.arange(0,4,1),rotation=0)
    

    ax=subfigs[1].subplots()
    for i in range(len(list_extra_eval)):
        if list_extra_eval[i]<0:
            list_extra_eval[i]=0
    list_extra_eval.reverse()
    plt.barh(list_acc_str, list_extra_eval, align='center',color=color[1])
    plt.yticks([])
    plt.title("Extra evaluations in the same amount of \n time required for maximum accuracy")
    plt.xlabel("Evaluations")


    #----------------------------------------------------------------------------------------------
    # GRAPH 2: loss of quality in the evaluations when considering minor accuracys and existence
    # of a minor accuracy with which maximum quality is obtained
    #----------------------------------------------------------------------------------------------
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
        abs_dist_matrix=[]
        for i in range(len(ranking_matrix)):
            abs_dist_matrix.append(np.abs(list(np.array(ranking_matrix[i])-np.array(ranking_matrix[-1]))))

        max_value=max(max(i) for i in abs_dist_matrix)
        norm_abs_dist_matrix=list(np.array(abs_dist_matrix)/max_value)
        
        return norm_abs_dist_matrix

    ax=subfigs[3].subplots()

    ranking_list=[]

    for accuracy in list_acc:
        df_acc=df[df['accuracy']==accuracy]
        ranking=from_argsort_to_ranking(list(df_acc['score'].argsort()))
        ranking_list.append(ranking)

    ranking_matrix=np.matrix(absolute_distance_matrix_between_rankings(ranking_matrix_sorted_by_max_acc(ranking_list)))
    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)

    ax = sns.heatmap(ranking_matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    colorbar=ax.collections[0].colorbar
    colorbar.set_label('Normalized distance')

    ax.set_xlabel('Solution (sorted by the ranking of maximum accuracy)')
    ax.set_xticks(range(0,ranking_matrix.shape[1],10))
    ax.set_xticklabels(range(1,ranking_matrix.shape[1]+1,10),rotation=0)

    ax.set_ylabel('Accuracy')
    ax.set_title('Normalized absolute distances between the respective positions \n of the best and the rest rankings')
    ax.set_yticks(np.arange(0.5,len(list_acc)+0.5,1))
    ax.set_yticklabels(list_acc,rotation=0)

    # Save graph.
    plt.savefig('results/figures/Turbines/UnderstandingAccuracyAnalysis/UnderstandingAccuracy_bladenumber'+str(blade_number)+'.png')
    plt.show()
    plt.close()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Read databases.
df1=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyI.csv',index_col=0)
df2=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII.csv',index_col=0)
df2_bladenumber3=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber3.csv',index_col=0)
df2_bladenumber5=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber5.csv',index_col=0)
df2_bladenumber7=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber7.csv',index_col=0)
df2_bladenumberAll=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumberAll.csv',index_col=0)

# Draw graphs.
from_data_to_figuresI(df1)
from_data_to_figuresII(df2_bladenumberAll,'All')
from_data_to_figuresII(df2_bladenumber3,3)
from_data_to_figuresII(df2_bladenumber5,5)
from_data_to_figuresII(df2_bladenumber7,7)
from_data_to_figures_bladenumber(df2_bladenumberAll,blade_number='All')
from_data_to_figures_bladenumber(df2_bladenumber3,blade_number=3)
from_data_to_figures_bladenumber(df2_bladenumber5,blade_number=5)
from_data_to_figures_bladenumber(df2_bladenumber7,blade_number=7)


