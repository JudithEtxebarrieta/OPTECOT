
'''
This script is used to graphically represent the numerical results obtained in 
"ConstantAccuracy_PopulationInfluence_data.py".
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
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import math

#==================================================================================================
# FUNCTIONS
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# General functions.
#--------------------------------------------------------------------------------------------------
def bootstrap_median_and_confidence_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def common_n_gen_per_seed(blade_number,list_seeds):
    n_gen_per_seed=math.inf

    for seed in list_seeds:
        df=pd.read_csv('results/data/Turbines/PopulationInfluence/df_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv',index_col=0)
        max_n_gen=int(df['n_gen'].max())

        if max_n_gen<n_gen_per_seed:
            n_gen_per_seed=max_n_gen

    return n_gen_per_seed
#--------------------------------------------------------------------------------------------------
# Functions for ranking charts.
#--------------------------------------------------------------------------------------------------
def from_argsort_to_ranking(list):
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

def ranking_matrix(df,gen):

    # Reduce database to rows of interest.
    df=df[df['n_gen']==gen]

    # Store rankings of each N in a row matrix.
    matrix=[]
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)
 
    for N in list_N:
        df_N=df[df['N']==N]
        list_scores=df_N.sort_values('n_eval')['score']
        ranking=from_argsort_to_ranking(np.argsort(-np.array(list_scores)))# Descending order.
        matrix.append(ranking)

    # Reorder rows of the matrix so that the row with the highest precision represents a perfectly ordered ranking (from best to worst).
    matrix_sort=[]
    ind_sort=np.argsort(matrix[0])
    for row in range(len(matrix)):

        new_row=[]
        for i in ind_sort:
            new_row.append(matrix[row][i])

        matrix_sort.append(new_row)
    return matrix_sort

def draw_rankings_per_generation(size,blade_number,seed,figure):

    # Read database.
    df=pd.read_csv('results/data/Turbines/PopulationInfluence/df_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv',index_col=0)

    # Number of total evaluations.
    total_n_eval=df['n_eval'].max()

    # N values considered.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # For each generation draw the ranking matrix.
    for gen in range(1,size[1]+1):
        ax=plt.subplot(size[0],size[1],gen+(seed-1)*size[1])

        # Create a ranking matrix.
        matrix=np.matrix(ranking_matrix(df,(figure-1)*size[1]+gen))

        # Draw the matrix.
        color = sns.color_palette("deep",total_n_eval )
        color = ListedColormap(color)
        ax = sns.heatmap(matrix, cmap=color,linewidths=.5, linecolor='lightgray',cbar=False)
        plt.xticks([])

        if seed==size[0]-1:
            ax.set_xlabel('Generation '+str((figure-1)*size[1]+gen))

        if gen ==1:
            ax.set_ylabel('Seed '+str(seed)+'\n N')
            ax.set_yticks(np.arange(.5,matrix.shape[0]+.5))
            ax.set_yticklabels(list_N, rotation=0)

        if gen !=1:
            plt.yticks([])
    return color

#--------------------------------------------------------------------------------------------------
# Functions for evaluation time graphing and ranking comparison.
#--------------------------------------------------------------------------------------------------
def pearson_corr(x,y):
    return sc.stats.pearsonr(x,y)[0]
def spearman_corr(x,y):
    return sc.stats.spearmanr(x,y)[0]

def join_df(blade_number,list_seeds):

    # First database.
    df=pd.read_csv('results/data/Turbines/PopulationInfluence/df_blade_number'+str(blade_number)+'_seed'+str(list_seeds[0])+'.csv',index_col=0)
    
    # Others.
    for seed in list_seeds[1:]:
        new_df=pd.read_csv('results/data/Turbines/PopulationInfluence/df_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv',index_col=0)
        df=pd.concat([df,new_df],ignore_index=True)    
    return df

def score_matrix(df_seed,gen):

    # Reduce df to rows of interest.
    df_seed_gen=df_seed[df_seed['n_gen']==gen]

    # Store scores of each N in a row matrix.
    matrix=[]
    list_N=list(set(df_seed_gen['N']))
    list_N.sort(reverse=True)

    for N in list_N:
        df_N=df_seed_gen[df_seed_gen['N']==N]
        list_scores=df_N.sort_values('n_eval')['score']
        matrix.append(list(list_scores))

    return matrix

def draw_ranking_comparison_per_gen(position,legend,blade_number,list_seeds,type,what):

    # Join all the databases into one.
    df=join_df(blade_number,list_seeds)

    # N values considered.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Number of generations to consider.
    n_gen_considered=min(df.groupby('seed')['n_gen'].max())

    # Initialize graph.
    ax=plt.subplot(position)

    # For each value of N draw a curve.
    for ind_N in range(len(list_N)):
        # Initialize lists.
        all_mean=[]
        all_q05=[]
        all_q95=[]

        # Per generation update the previous lists.
        for gen in range(1,n_gen_considered+1):
            list_metric=[]
            for seed in list_seeds:
                df_seed=df[df['seed']==seed]
                if what=='scores':
                    matrix=score_matrix(df_seed,gen)
                if what=='positions':
                    matrix=ranking_matrix(df_seed,gen)
                if type=='pearson':
                    list_metric.append(pearson_corr(matrix[0],matrix[ind_N]))
                if type=='spearman':
                    list_metric.append(spearman_corr(matrix[0],matrix[ind_N]))

            mean,q05,q95=bootstrap_median_and_confidence_interval(list_metric)
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)
        
        # Draw curve.
        ax.fill_between(range(1,n_gen_considered+1),all_q05,all_q95,alpha=.5,linewidth=0)
        plt.plot(range(1,n_gen_considered+1),all_mean,linewidth=2,label=str(list_N[ind_N]))

    # Details of the graph.
    ax.set_ylabel(str(type)+' correlation')
    ax.set_xlabel('Generation')
    ax.set_title('Comparing similarity between the perfect \n and each '+str(what)+' ranking (blade-number '+str(blade_number)+')')
    if legend==True:
        ax.legend(title="N",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

def draw_total_eval_time_per_gen(position,legend,blade_number,list_seeds):
    # Join all the databases into one.
    df=join_df(blade_number,list_seeds)

    # N values considered.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Number of generations to consider.
    n_gen_considered=min(df.groupby('seed')['n_gen'].max())

    # Initialize graph.
    ax=plt.subplot(position)

    # For each value of N draw a curve.
    for N in list_N:
        # Initialize lists.
        all_mean=[]
        all_q05=[]
        all_q95=[]

        # Per generation update the previous lists.
        for gen in range(1,n_gen_considered+1):
            list_times=list(df[(df['N']==N) & (df['n_gen']==gen)].groupby('seed')['time'].sum())

            mean,q05,q95=bootstrap_median_and_confidence_interval(list_times)
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)

        # Draw curve.
        ax.fill_between(range(1,n_gen_considered+1),all_q05,all_q95,alpha=.5,linewidth=0)
        plt.plot(range(1,n_gen_considered+1),all_mean,linewidth=2,label=str(N))

    # Graph details.
    ax.set_ylabel('Evaluation time (seconds)')
    ax.set_xlabel('Generation')
    ax.set_title('Evaluation time depending on N \n (blade-number '+str(blade_number)+')')
    if legend==True:
        ax.legend(title="N",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

def draw_best_turbine_score_per_gen(position,legend,blade_number,list_seeds):
    # Join all the databases into one.
    df=join_df(blade_number,list_seeds)    

    # N values considered.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Number of generations to consider.
    n_gen_considered=min(df.groupby('seed')['n_gen'].max())

    # Initialize graph.
    ax=plt.subplot(position)

    # For each value of N draw a curve.
    for N in list_N:
        # Initialize lists.
        all_mean=[]
        all_q05=[]
        all_q95=[]

        # Per generation update the previous lists.
        for gen in range(1,n_gen_considered+1):
            df_N_gen=df[(df['N']==N) & (df['n_gen']==gen)]
            ind_max_score_per_seed=df_N_gen.groupby('seed')['score'].idxmax()
            best_turbine_per_seed=list(df_N_gen.loc[ind_max_score_per_seed]['n_eval'])
            
            list_real_scores=[]
            df_best_N_gen=df[(df['N']==50) & (df['n_gen']==gen)]
            seed=1
            for ind_best_turbine in best_turbine_per_seed:
                list_real_scores.append(float(df_best_N_gen[(df_best_N_gen['seed']==seed)&(df_best_N_gen['n_eval']==ind_best_turbine)]['score']))
                seed+=1

            mean,q05,q95=bootstrap_median_and_confidence_interval(list_real_scores)
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)

        # Draw curve.
        ax.fill_between(range(1,n_gen_considered+1),all_q05,all_q95,alpha=.5,linewidth=0)
        plt.plot(range(1,n_gen_considered+1),all_mean,linewidth=2,label=str(N))

    # graph details.
    ax.set_ylabel('Real score')
    ax.set_xlabel('Generation')
    ax.set_title('Real score depending on N \n (blade-number '+str(blade_number)+')')
    if legend==True:
        ax.legend(title="N",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

def draw_n_null_score_turb_per_gen(position,legend,blade_number,list_seeds):
    # Join all the databases into one.
    df=join_df(blade_number,list_seeds)    

    # N values considered.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Number of generations to consider.
    n_gen_considered=min(df.groupby('seed')['n_gen'].max())

    # Initialize graph.
    ax=plt.subplot(position)

    # For each value of N draw a curve.
    for N in list_N:
        # Initialize lists.
        all_mean=[]
        all_q05=[]
        all_q95=[]

        # Per generation update the previous lists.
        for gen in range(1,n_gen_considered+1):
            df_N_gen=df[(df['N']==N) & (df['n_gen']==gen)]
            n_seed=len(set(df_N_gen['seed']))
            list_n_null_per_seed=list(df_N_gen[df_N_gen['score']==0].groupby('seed').size())
            list_n_null_per_seed=list_n_null_per_seed+[0]*(n_seed-len(list_n_null_per_seed))

            mean,q05,q95=bootstrap_median_and_confidence_interval(list_n_null_per_seed)
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)

        # Draw curve.
        ax.fill_between(range(1,n_gen_considered+1),all_q05,all_q95,alpha=.5,linewidth=0)
        plt.plot(range(1,n_gen_considered+1),all_mean,linewidth=2,label=str(N))

    # Graph details.
    ax.set_ylabel('Number of null scores')
    ax.set_xlabel('Generation')
    ax.set_title('Number of null score turbines per generation \n depending on N (blade-number '+str(blade_number)+')')
    if legend==True:
        ax.legend(title="N",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

#--------------------------------------------------------------------------------------------------
# Functions for graphical illustration of turbine designs per generation for each seed.
#--------------------------------------------------------------------------------------------------
def define_bounds():
    # Blade-number values.
    bn_values=[3,5,7]

    # Define ranges of parameters defining the turbine design.
    sigma_hub = [0.4, 0.7]# Hub solidity gene.
    sigma_tip = [0.4, 0.7]# Tip solidity gene.
    nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
    tip_clearance=[0,3]# Tip-clearance gene.	  
    airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

    # List with ranges.
    rest_bounds=np.array([
    [sigma_hub[0]    , sigma_hub[1]],
    [sigma_tip[0]    , sigma_tip[1]],
    [nu[0]           , nu[1]],
    [tip_clearance[0], tip_clearance[1]],
    [0               , 26]
    ])

    return bn_values, rest_bounds

def scale_turb_params(x_bn,x_rest):
    bn_values, rest_bounds=define_bounds()
    x_bn=(1/len(bn_values))*(bn_values.index(x_bn)+1)
    x_rest=(x_rest-rest_bounds[:,0]) * (1/(rest_bounds[:,1] - rest_bounds[:,0]))

    return [x_bn]+list(x_rest[:-1])+[round(x_rest[-1])]

def turb_params_matrix(df_seed,gen,popsize):
    matrix=df_seed[popsize*(gen-1):popsize*gen]
    scaled_matrix=[]
    for ind_turb in range(matrix.shape[0]):
        turb_params=np.array(matrix.iloc[ind_turb])
        scaled_matrix.append(scale_turb_params(turb_params[0],turb_params[1:]))
    return scaled_matrix

def draw_turb_params_per_generation(size,blade_number,seed,popsize,figure):

    # Read database.
    df=pd.read_csv('results/data/Turbines/PopulationInfluence/df_turb_params_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv',index_col=0)

    # For each generation draw the turbine design matrix.
    for gen in range(1,size[1]+1):
        ax=plt.subplot(size[0],size[1],gen+(seed-1)*size[1])

        # Create matrix of turbine parameters.
        matrix=np.matrix(turb_params_matrix(df,(figure-1)*size[1]+gen,popsize))

        # Draw matrix.
        color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        ax = sns.heatmap(matrix, cmap=color,linewidths=.5, linecolor='lightgray',cbar=False,vmin=0,vmax=1)
        
        if seed==1:
            ax.set_title('Generation '+str((figure-1)*size[1]+gen))

        if seed==size[0]-2:
            ax.set_xlabel('Parameter')
            ax.set_yticks(np.arange(.5,matrix.shape[1]+.5))
            ax.set_yticklabels(range(1,matrix.shape[1]+1), rotation=0)

        if seed!=size[0]-2:
            plt.xticks([])

        if gen ==1:
            ax.set_ylabel('Seed '+str(seed)+'\n Turbine')
            ax.set_yticks(np.arange(.5,matrix.shape[0]+.5))
            ax.set_yticklabels(range(1,matrix.shape[0]+1), rotation=0)

        if gen !=1:
            plt.yticks([])
   

    return color


#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# GRAPH 1: rankings for blade-number=3.
#--------------------------------------------------------------------------------------------------
# Read database.
list_seeds=np.load('results/data/Turbines/PopulationInfluence/list_seeds.npy')

# Blade-number.
blade_number=3

# Number of generations in common for all seeds.
n_gen_per_seed=common_n_gen_per_seed(blade_number,list_seeds)
n_gen_per_figure=int(np.load("results/data/Turbines/PopulationInfluence/popsize.npy"))

# Initialize graph structure.
for figure in range(1,int(n_gen_per_seed/n_gen_per_figure+1)):
    plt.figure(figsize=[15,9])
    plt.subplots_adjust(left=0.1,bottom=0.05,right=0.95,top=0.89,wspace=0.14,hspace=0.3)
    for seed in list_seeds:
        color=draw_rankings_per_generation([len(list_seeds)+1,n_gen_per_figure],blade_number,seed,figure)

    # Color bar.
    ax = plt.subplot(len(list_seeds)+1,n_gen_per_seed,(len(list_seeds)*n_gen_per_seed+2,(len(list_seeds)+1)*n_gen_per_seed-1))
    mpl.colorbar.ColorbarBase(ax, cmap=color,orientation='horizontal')
    ax.set_xlabel('Ranking position')
    ax.set_xticks(np.arange(.05,1.05,0.1))
    ax.set_xticklabels(range(1,10+1,1))

    plt.figtext(0.5, 0.95, 'Rankings per generation (blade-number '+str(blade_number)+') (FIGURE '+str(figure)+')', ha='center', va='center')
    plt.savefig('results/figures/Turbines/PopulationInfluence/rankings'+str(figure)+'.png')
    plt.show()
    plt.close()

#--------------------------------------------------------------------------------------------------
# GRAPH 2: compare turbine designs considered in each generation for blade-number=3.
#--------------------------------------------------------------------------------------------------
# Create graph.
for figure in range(1,int(n_gen_per_seed/n_gen_per_figure+1)):
    plt.figure(figsize=[15,10])
    plt.subplots_adjust(left=0.1,bottom=0.05,right=0.95,top=0.89,wspace=0.14,hspace=0.4)

    for seed in list_seeds:
        color=draw_turb_params_per_generation([len(list_seeds)+2,n_gen_per_figure],blade_number,seed,n_gen_per_figure,figure)

    # Color bar.
    ax = plt.subplot(len(list_seeds)+2,n_gen_per_seed,((len(list_seeds)+1)*n_gen_per_seed+2,(len(list_seeds)+2)*n_gen_per_seed-1))
    mpl.colorbar.ColorbarBase(ax, cmap=color,orientation='horizontal')
    ax.set_xlabel('Scaled parameter value')

    plt.figtext(0.5, 0.95, 'Turbine designs per generation (blade-number '+str(blade_number)+') (FIGURE '+str(figure)+')', ha='center', va='center')
    plt.savefig('results/figures/Turbines/PopulationInfluence/turbines'+str(figure)+'.png')
    plt.show()
    plt.close()

#--------------------------------------------------------------------------------------------------
# GRAPH 3: comparison of evaluation times and rankings obtained per generation when considering 
# different accuracy values.
#--------------------------------------------------------------------------------------------------
# Initialize graph.
plt.figure(figsize=[12,10])
plt.subplots_adjust(left=0.12,bottom=0.05,right=0.84,top=0.87,wspace=0.22,hspace=0.3)

# SUBGRAPH 1: comparison of rankings (using the positions) per generation for blade-number=3.
draw_ranking_comparison_per_gen(222,True,blade_number,list_seeds,'spearman','positions')

# SUBGRAPH 2: time saved per generation by considering less accurate values of N for blade-number=3.
draw_total_eval_time_per_gen(223,False,blade_number,list_seeds)

# SUBGRAPH 3: comparison of actual scores obtained per generation for the best selected turbines with different N values.
draw_best_turbine_score_per_gen(221,False,blade_number,list_seeds)

# SUBGRAPH 4: number of turbine designs per generation with associated null score.
draw_n_null_score_turb_per_gen(224,False,blade_number,list_seeds)

# Save graph.
plt.savefig('results/figures/Turbines/PopulationInfluence/eval_time_and_ranking_comparison.png')
plt.show()
plt.close()