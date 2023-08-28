#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
import random
import time
import scipy as sc
import pandas as pd
import math
import cma
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import plotly.express as px
from matplotlib.patches import Rectangle
import sys
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

#==================================================================================================
# CLASSES
#==================================================================================================
class AuxiliaryFunctions:

    @staticmethod
    def from_argsort_to_ranking(list):
        '''Obtain ranking from a list get after applying "np.argsort" on an original list.'''
        new_list=[0]*len(list)
        i=0
        for j in list:
            new_list[j]=i
            i+=1
        return new_list

    @staticmethod
    def custom_sort(list,argsort):
        '''Sort list according to the order for each position defined in argsort.'''
        new_list=[]
        for index in argsort:
            new_list.append(list[index])
        return new_list
    
    @staticmethod
    def spearman_corr(x,y):
        '''Calculation of Spearman's correlation between two lists.'''
        return sc.stats.spearmanr(x,y)[0]

    @staticmethod
    def from_scores_to_ranking(list_scores):
        '''Convert score list to ranking list.'''

        list_pos_ranking=np.argsort(np.array(list_scores))
        ranking=[0]*len(list_pos_ranking)
        i=0
        for j in list_pos_ranking:
            ranking[j]=i
            i+=1
        return ranking
    
    @staticmethod
    def from_cost_to_theta(cost,theta0,theta1):
        if theta1>theta0:
            acc=theta0/theta1+cost*(1-theta0/theta1)
            return acc,int(theta1*acc)
        else:
            acc=theta1/theta0+cost*(1-theta1/theta0)
            return acc,int(theta1/acc)

    @staticmethod
    def extract_info_evaluating_set_with_equidistant_accuracies(set_solutions,objective_function,theta0,theta1,path):
        '''Build data frames from information obtained after evaluating a set o solutions with 10 different theta values associated with equidistant accuracies.'''

        df_info_set=[]

        list_cost=np.arange(0,1+1/10,1/9) # As cost and theta accuracy are linearly proportional, to equidistant values of cost corresponds equidistant values of accuracy.

        total_it=len(list_cost)*len(set_solutions)
        it=0
        for cost in list_cost:
            acc,theta=AuxiliaryFunctions.from_cost_to_theta(cost,theta0,theta1)
            for n_sol in range(len(set_solutions)):
                t=time.time()
                score=objective_function(set_solutions[n_sol],theta=theta)
                elapsed_time=time.time()-t
                df_info_set.append([acc,n_sol,score,elapsed_time])
                print('    Computing |S| and t^{period}... '+colored(str(round((it/total_it)*100,2))+'%','light_cyan'),end='\r')
                sys.stdout.flush()
                it+=1

        df_info_set=pd.DataFrame(df_info_set,columns=['accuracy','n_solution','score','cost_per_eval'])
        df_acc_time=df_info_set[['accuracy','cost_per_eval']]
        df_acc_time=df_acc_time.groupby('accuracy').mean()
        df_acc_time=df_acc_time.reset_index()

        df_info_set.to_csv(path+'/df_info_set.csv')
        df_acc_time.to_csv(path+'/df_acc_time.csv')
        return df_acc_time

    @staticmethod
    def SampleSize_TimePeriod_bisection_method(popsize,min_sample_size,perc_cost,df_acc_time):

        '''
        Calculate the sample size for the bisection method and the frequency with which the indications 
        of the heuristic should be accepted.
        '''

        def bisection_middle_points(lower,upper,type_cost='max'):
            '''
            Obtain list of midpoints that are selected in 4 iterations of the bisection method, 
            in the worst case and in the mean case.
            '''
            list=[] 
            stop_threshold=(upper-lower)*0.1
            max_value=upper

            # Mean case.
            if type_cost=='mean':
                first=True
                while abs(lower-upper)>stop_threshold:       
                    middle=(lower+upper)/2
                    list.append(middle)
                    if first:
                        lower=middle
                        first=False
                    else:
                        upper=middle

            # Worst case: In all the iterations, the interval is bounded below.
            if type_cost=='max':
                while abs(lower-upper)>stop_threshold:       
                    middle=(lower+upper)/2
                    list.append(middle)
                    lower=middle

            return list+[max_value]

        time_list=bisection_middle_points(min(df_acc_time['cost_per_eval']),max(df_acc_time['cost_per_eval']))

        # Cost of evaluating a population with maximum accuracy.
        cost_max_acc=time_list[-1]*popsize

        # Cost of applying the bisection method on a population.
        cost_bisection=min_sample_size*(sum(time_list)-time_list[-2])

        # Real percentage.
        perc=cost_bisection/cost_max_acc

        # If applying the bisection method exceeds the predefined cost percentage, define how often 
        # the heuristic should be heeded (when to readjust the accuracy).
        if perc>perc_cost:
            sample_size=min_sample_size
            freq_gen=math.ceil(perc/perc_cost)

        # Otherwise.
        else:
            sample_size=int((perc_cost*cost_max_acc)/(sum(time_list)-time_list[-2]))
            freq_gen=1
        time_period=freq_gen*cost_max_acc

        return sample_size, time_period

    @staticmethod
    def interruption_threshold(popsize,sample_size,df_acc_time,path):
        
        df=[]
        list_sequences=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

        # Read database with evaluation time of each accuracy.
        list_acc=list(df_acc_time['accuracy'])
        list_time=list(df_acc_time['cost_per_eval'])

        for sequence in list_sequences:
            upper=max(list_time)
            lower=min(list_time)

            # Cost of the bisection method when it bounded interval like specified in sequence.
            middle=(lower+upper)/2
            j=1 # Fist midpoint
            cost=upper
            cost+=middle
            for i in sequence:       
                if i==0:
                    lower=middle
                else:
                    upper=middle
                middle=(lower+upper)/2
                j+=1 # Update midpoints counter.
                if j<=len(sequence):
                    cost+=middle
            cost=cost*sample_size

            # Cost of evaluate by default a population.
            default_cost=popsize*max(list_time)

            # Cost of evaluate with optimal accuracy a population.
            opt_cost=middle*popsize

            # Table content.
            opt_acc=np.interp(middle,list_time,list_acc)
            bisec_cost_perc=cost/default_cost
            pop_eval_save_perc=1-opt_cost/default_cost

            # Update database.
            df.append([opt_acc,cost,pop_eval_save_perc])

        # interruption threshold.
        df_ExtraCost_SavePopEvalCost=pd.DataFrame(df,columns=['opt_acc','bisec_cost','pop_eval_save_perc'])
        df_ExtraCost_SavePopEvalCost.to_csv(path+'/df_ExtraCost_SavePopEvalCost.csv')
        interruption_threshold=float(max(df_ExtraCost_SavePopEvalCost['opt_acc']))

        return interruption_threshold


class ExperimentalGraphs:

    @staticmethod
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
    
    @staticmethod
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
            mean,q05,q95=ExperimentalGraphs.bootstrap_mean_and_confidence_interval(interest)

            # Save data.
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)

        return all_mean,all_q05,all_q95
    
    @staticmethod
    def draw_accuracy_behaviour(ax,df_train,type_time,list_train_time,colors,list_markers):
        '''Construct curve showing the accuracy behavior during the execution process.'''

        # Define relationship between accuracy (term used in the code) and cost (term used in the paper).
        def a_c(a,a_0,a_1):
            c=(a-a_0)/(a_1-a_0)
            return c
        a_0=0.001 
        a_1=1

        # Initialize lists for the graph.
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
            mean,q05,q95=ExperimentalGraphs.bootstrap_mean_and_confidence_interval(list_acc)

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

    @staticmethod
    def illustrate_approximate_objective_functions_use(optecot):

        print('Drawing results obtained after executing CMA-ES using approximate objective functions:')
        sys.stdout.flush()

        list_acc=[]
        for cost in optecot.list_costs:
            acc,_=AuxiliaryFunctions.from_cost_to_theta(cost,optecot.theta0,optecot.theta1)
            list_acc.append(acc)

        # Obtener base de datos a partir de la cual se construira la grafica.
        df=pd.read_csv(optecot.auxiliary_data_path+'/df_info_set.csv',index_col=0)

        #------------------------------------------------------------------------------------------
        # Funciones auxiliares
        #------------------------------------------------------------------------------------------
        def convert_textbf(x):
            return [r'\textbf{'+str(i)+'}' for i in x]
        
        # Define relationship between accuracy (term used in the code) and cost (term used in the paper).
        def a_c(a,a_0,a_1):
            c=(a-a_0)/(a_1-a_0)
            return c
        list_acc=list(set(df['accuracy']))
        list_acc.sort()
        a_0=min(list_acc) 
        a_1=max(list_acc)       

        #------------------------------------------------------------------------------------------
        # Construir gráficas asociadas a la evaluacion del conjunto de soluciones usando 
        # las diferentes aproximaciones.
        #------------------------------------------------------------------------------------------

        print('    Generating figure of evaluation times, extra evaluation proportions and ranking preservations (Figure 3 in the paper)...',end='\r')
        sys.stdout.flush()

        # Initialize graph.
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = r'\boldmath'
        plt.figure(figsize=[5,5],constrained_layout=True)

        #__________________________________________________________________________________________
        # GRAPH 1: evaluation time per accuracy and extra evaluations.

        # Define accuracy list.
        list_acc_str=[str(round(a_c(acc,a_0,a_1),2)) for acc in list_acc]
        list_acc_str.reverse()
        list_acc_float=[round(a_c(acc,a_0,a_1),2) for acc in list_acc]

        # Execution times per evaluation.
        all_means=[]
        all_q05=[]
        all_q95=[]

        for accuracy in list_acc:
            mean,q05,q95=ExperimentalGraphs.bootstrap_mean_and_confidence_interval(list(df[df['accuracy']==accuracy]['cost_per_eval']))
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

        #__________________________________________________________________________________________
        # GRAPH 2: Loss of quality in the evaluations when considering lower accuracies and existence 
        # of a lower accuracy with which the maximum quality is obtained.

        ax=plt.subplot(2,2,(3,4))

        def ranking_matrix_sorted_by_max_acc(ranking_matrix):
            '''Reorder ranking matrix according to the order established by the original ranking.'''

            # Indexes associated with the maximum accuracy ranking ordered from lowest to highest.
            argsort_max_acc=np.argsort(ranking_matrix[-1])

            # Reorder matrix rows.
            sorted_ranking_list=[]
            for i in range(len(ranking_list)):
                sorted_ranking_list.append(AuxiliaryFunctions.custom_sort(ranking_list[i],argsort_max_acc))

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
            ranking=AuxiliaryFunctions.from_argsort_to_ranking(list(df_acc['score'].argsort()))
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
        plt.savefig(optecot.figure_path+'/ApproximateObjectiveFunctions_SolutionSet.png')
        plt.savefig(optecot.figure_path+'/ApproximateObjectiveFunctions_SolutionSet.pdf')

        plt.close()

        print('    Generating figure of evaluation times, extra evaluation proportions and ranking preservations (Figure 3 in the paper)'+ colored('    DONE','light_cyan'))
        sys.stdout.flush()

        #------------------------------------------------------------------------------------------
        # Construir gráfica asociada a la ejecucion del RBEA usando diferentes aproximaciones con 
        # una precision de theta constante durante toda la ejecucion.
        #------------------------------------------------------------------------------------------
        print('    Generating figure of the solution quality curves associated with the use of approximate objective functions with constant cost (Figure 4 in the paper)...',end='\r')
        sys.stdout.flush()

        # Define list with limits of training times to be drawn.
        df_train_acc_min=pd.read_csv('library_OPTECOT/results/data/df_ConstantAnalysis_cost'+'{:.02f}'.format(min(optecot.list_costs))+".csv", index_col=0)
        df_train_acc_max=pd.read_csv('library_OPTECOT/results/data/df_ConstantAnalysis_cost'+'{:.02f}'.format(max(optecot.list_costs))+".csv", index_col=0)
        min_time=max(df_train_acc_max.groupby('seed')['elapsed_time'].min())
        split_time=max(df_train_acc_min.groupby('seed')['elapsed_time'].min())
        list_train_times=np.arange(min_time,optecot.max_time,split_time)

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
        for cost in optecot.list_costs:

            # Read database.
            df_train_acc=pd.read_csv('library_OPTECOT/results/data/df_ConstantAnalysis_cost'+'{:.02f}'.format(cost)+".csv", index_col=0)

            # Extract relevant information from the database.
            all_mean_scores,all_q05_scores,all_q95_scores =ExperimentalGraphs.train_data_to_figure_data(df_train_acc,list_train_times)

            # Set maximum solution quality.
            if cost==max(optecot.list_costs):
                score_limit=all_mean_scores[-1]

            # Draw curve.
            ax.fill_between(list_train_times,all_q05_scores,all_q95_scores, alpha=.2, linewidth=0,color=colors[curve])
            plt.plot(list_train_times, all_mean_scores, linewidth=1,label=r'\textbf{'+str(cost)+'}',color=colors[curve],marker=list_markers[curve],markevery=0.1)
            curve+=1

        ax.set_ylim([0.42,0.55])
        ax.set_xlabel("$t$",fontsize=23)
        ax.set_ylabel(r"\textbf{Solution quality}",fontsize=23)
        ax.set_title(r'\textbf{Turbines}',fontsize=23)
        leg=ax.legend(title="$c$",fontsize=16.5,title_fontsize=16.5,labelspacing=0.1,handlelength=0.8,loc='upper left')
        plt.axhline(y=score_limit,color='black', linestyle='--')
        ax.set_xscale('log')
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)

        plt.savefig(optecot.figure_path+'/ApproximateObjectiveFunctions_ConstantAnalysis.png')
        plt.savefig(optecot.figure_path+'/ApproximateObjectiveFunctions_ConstantAnalysis.pdf')

        plt.close()

        print('    Generating figure of the solution quality curves associated with the use of approximate objective functions with constant cost (Figure 4 in the paper)'+colored('    DONE','light_cyan'))
        print(colored('Results drawn.','light_yellow',attrs=["bold"]))
        sys.stdout.flush()

    @staticmethod
    def illustrate_OPTECOT_application_results(optecot):

        print('Drawing results obtained after executing CMA-ES appliying OPTECOT:')
        sys.stdout.flush()

        # List of scores and markers.
        colors=px.colors.qualitative.D3
        list_markers = ["o", "^", "s", "P", "v", 'D', "x", "1", "|", "+"]
        colors1=['#C7C9C8' ,'#AFB0AF' ]
        colors2=['#C7C9C8' ,'#AFB0AF','#F4E47B','#DBC432' ]

        # Define training times to be drawn.
        df_max_acc=pd.read_csv(optecot.data_path+'/df_ConstantAnalysis_cost1.00.csv', index_col=0)
        df_optimal_acc=pd.read_csv(optecot.data_path+'/df_OPTECOT.csv', index_col=0)
        min_time_acc_max=max(df_max_acc.groupby('seed')['elapsed_time'].min())
        min_time_h=max(df_optimal_acc.groupby('seed')['elapsed_time'].min())

        df_cost_per_acc=pd.read_csv(optecot.auxiliary_data_path+'/df_acc_time.csv',index_col=0)
        time_split=list(df_cost_per_acc['cost_per_eval'])[0]*optecot.popsize

        list_train_time=np.arange(max(min_time_acc_max,min_time_h),optecot.max_time+time_split,time_split)

        #------------------------------------------------------------------------------------------
        # Curva de calidad y comportamiento de coste optimo.
        #------------------------------------------------------------------------------------------
        print('    Generating figure of solution quality curves (Figure 5 in the paper) and optimal cost behavior (Figure 7 in the paper)...',end='\r')
        sys.stdout.flush()

        # Initialize graph.
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = r'\boldmath'


        #__________________________________________________________________________________________
        # GRAPH 1: best solution quality during execution process.

        plt.figure(figsize=[7,12])
        plt.subplots_adjust(left=0.18,bottom=0.11,right=0.85,top=0.93,wspace=0.32,hspace=0.35)
        ax=plt.subplot(211)
        ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')

        # Constant use of accuracy (default situation).
        all_mean,all_q05,all_q95=ExperimentalGraphs.train_data_to_figure_data(df_max_acc,list_train_time)
        ax.fill_between(list_train_time,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[0])
        plt.plot(list_train_time, all_mean, linewidth=1,label=r'\textbf{Original}',color=colors[0],marker=list_markers[0],markevery=0.1)

        # Optimal use of accuracy (heuristic application).
        all_mean,all_q05,all_q95=ExperimentalGraphs.train_data_to_figure_data(df_optimal_acc,list_train_time)
        ax.fill_between(list_train_time,all_q05,all_q95, alpha=.2, linewidth=0,color=colors[1])
        plt.plot(list_train_time, all_mean, linewidth=1,label=r'\textbf{OPTECOT}',color=colors[1],marker=list_markers[1],markevery=0.1)


        ax.set_ylabel(r"\textbf{Solution quality}",fontsize=23)
        ax.set_xlabel("$t$",fontsize=23)
        ax.legend(fontsize=18,title_fontsize=18,labelspacing=0.1,handlelength=0.8,loc='lower right')
        ax.set_title(r'\textbf{Turbines}',fontsize=23)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        ax.set_xscale('log')

        #__________________________________________________________________________________________
        # GRAPH 2: Graphical representation of the accuracy behavior.

        ax=plt.subplot(212)
        ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')

        # Draw curve.
        ExperimentalGraphs.draw_accuracy_behaviour(ax,df_optimal_acc,'elapsed_time',list_train_time,colors,list_markers)

        ax.ticklabel_format(style='sci', axis='x', useOffset=True, scilimits=(0,0))
        ax.set_xticks(range(500,4000,1000))
        ax.set_xlabel("$t$",fontsize=23)
        ax.set_ylabel("$\widetilde{c}$",fontsize=23)
        ax.set_title(r'\textbf{Turbines}',fontsize=23)
        plt.xticks(fontsize=23)
        plt.yticks(fontsize=23)
        ax.set_xscale('log')
        ax.set_ylim([0,1])

        plt.savefig(optecot.figure_path+'/OPTECOT_QualityCurve+OptimalCostbehaviour.png')
        plt.savefig(optecot.figure_path+'/OPTECOT_QualityCurve+OptimalCostbehaviour.pdf')
        plt.close()

        print('    Generating figure of solution quality curves (Figure 5 in the paper) and optimal cost behavior (Figure 7 in the paper)'+colored('    DONE','light_cyan'))
        sys.stdout.flush()

        #------------------------------------------------------------------------------------------
        # Analisis de efectividad de OPTECOT.
        #------------------------------------------------------------------------------------------
        print('    Generating figure of quality improvement curves and time saving curves (Figure 6 in the paper)... ',end='\r')
        sys.stdout.flush()

        # Initialize graph.
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = r'\boldmath'
        plt.figure(figsize=[4,4])
        plt.subplots_adjust(left=0.27,bottom=0.14,right=0.95,top=0.94,wspace=0.3,hspace=0.07)

        all_mean_acc_max,_,_=ExperimentalGraphs.train_data_to_figure_data(df_max_acc,list_train_time)
        all_mean_h,_,_=ExperimentalGraphs.train_data_to_figure_data(df_optimal_acc,list_train_time)

        #__________________________________________________________________________________________
        # GRAPH 1: Curve of quality increment.

        ax=plt.subplot(411)

        quality_percentage=list((np.array(all_mean_h)/np.array(all_mean_acc_max))*100)
        # print('Quality above 100:',sum(np.array(quality_percentage)>100)/len(quality_percentage))
        # print('Best quality:',max(quality_percentage),np.mean(quality_percentage),np.std(quality_percentage))

        ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
        plt.plot(list_train_time,quality_percentage,color=colors2[3],linewidth=1.2)
        plt.axhline(y=100,color='black', linestyle='--')
        ax.set_title(r'\textbf{Turbines}',fontsize=15)
        ax.set_ylabel(r"\textbf{QI}",fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_xscale('log')
        ax.set_ylim([99.5,103.5])
        ax.set_yticks([100,101,102,103])

        #__________________________________________________________________________________________
        # GRAPH 2: Difference between the mean quality curves associated with heuristic application 
        # and original situation.

        ax=plt.subplot(4,1,(2,3))
        ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
        plt.plot(list_train_time, all_mean_acc_max, linewidth=1,label=r'\textbf{Original}',color=colors[0])
        plt.plot(list_train_time, all_mean_h, linewidth=1,label=r'\textbf{OPTECOT}',color=colors[1],linestyle = '--')
        ax.fill_between(list_train_time, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)>np.array(all_mean_acc_max),facecolor='green', alpha=.2,interpolate=True,label=r'\textbf{Improvement}')
        ax.fill_between(list_train_time, all_mean_acc_max, all_mean_h, where=np.array(all_mean_h)<np.array(all_mean_acc_max),facecolor='red', alpha=.2,interpolate=True,label=r'\textbf{Worsening}')

        ax.set_ylabel(r"\textbf{Solution quality}",fontsize=15)
        ax.legend(fontsize=11,title_fontsize=11,labelspacing=0.01,handlelength=1)
        plt.yticks(fontsize=15)
        ax.set_xscale('log')

        #__________________________________________________________________________________________
        # GRAPH 3: Curve of time saving.
   
        ax=plt.subplot(414)

        list_time_y=[]
        counter=0
        for i in all_mean_acc_max:
            aux_list=list(np.array(all_mean_h)>=i)
            if True in aux_list:
                ind=aux_list.index(True)
                list_time_y.append((list_train_time[ind]/list_train_time[counter])*100)

            counter+=1
        # print('Time below 100:',(sum(np.array(list_time_y)<100)+1)/len(list_train_time))
        # print('Best time:',min(list_time_y),np.mean(list_time_y),np.std(list_time_y))

        plt.gca().add_patch(Rectangle((list_train_time[len(list_time_y)], 45), max(list_train_time)-list_train_time[len(list_time_y)], 135-45,facecolor='grey',edgecolor='white',alpha=.2))
        ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
        plt.plot(list_train_time[:len(list_time_y)], list_time_y,color=colors2[3],label='Time',linewidth=1.2)
        plt.axhline(y=100,color='black', linestyle='--')

        ax.set_ylabel(r"\textbf{TR}",fontsize=15)
        ax.set_xlabel("$t$",fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        ax.set_xscale('log')
        ax.set_ylim([45,135])
        ax.set_yticks([50,75,100,125])

        plt.savefig(optecot.figure_path+'/OPTECOT_effectiveness.png')
        plt.savefig(optecot.figure_path+'/OPTECOT_effectiveness.pdf')
        plt.close()

        print('    Generating figure of quality improvement curves and time saving curves (Figure 6 in the paper)'+colored('    DONE','light_cyan'))
        print(colored('Results drawn.','light_yellow',attrs=["bold"]))
        sys.stdout.flush()

        



class OPTECOT:

    def __init__(self,popsize,xdim,max_time,theta0,theta1,objective_min,objective_function,set_solutions,scaled_solution_transformer,
                 alpha=0.95,beta=5,kappa=3,min_sample_size=10,perc_cost=0.25,
                 customized_df_acc_time=None,auxiliary_data_path=None,data_path=None,figure_path=None,list_costs=None):
        
    
        print('\nCreating an instance of OPTECOT:')
        sys.stdout.flush()
        print('    Setting explicitly indicated parameters... ',end='\r')
        sys.stdout.flush()

        # Initialize arguments taking into account giving data.
        self.popsize=popsize
        self.xdim=xdim
        self.max_time=max_time
        self.theta0=theta0
        self.theta1=theta1
        self.objective_min=objective_min
        self.objective_function=objective_function
        self.scaled_solution_transformer=scaled_solution_transformer
        self.alpha=alpha
        self.beta=beta
        self.kappa=kappa
        self.min_sample_size=min_sample_size
        self.perc_cost=perc_cost

        if auxiliary_data_path== None:
            self.auxiliary_data_path=os.path.abspath(__file__).split('/')[-2]+'/results/auxiliary_data'
            # os.makedirs(self.auxiliary_data_path)
        else:
            self.auxiliary_data_path=auxiliary_data_path

        if data_path== None:
            self.data_path=os.path.abspath(__file__).split('/')[-2]+'/results/data'
            # os.makedirs(self.data_path)
        else:
            self.data_path=data_path

        if figure_path== None:
            self.figure_path=os.path.abspath(__file__).split('/')[-2]+'/results/figures'
            # os.makedirs(self.figure_path)
        else:
            self.figure_path=figure_path

        print('    Setting explicitly indicated parameters'+ colored('    DONE','light_cyan'))
        print('    Computing |S| and t^{period}... ',end='\r')
        sys.stdout.flush()

        # Compute the rest of arguments using given data.
        if customized_df_acc_time is not None: # ESTE IF Y ELSE ES PROVISIONAL (puedo poner algo asi tb, pero lo definitivo de momento seria solo lo del else)
            df_acc_time=customized_df_acc_time
        else:
            df_acc_time=AuxiliaryFunctions.extract_info_evaluating_set_with_equidistant_accuracies(set_solutions,objective_function,theta0,theta1,self.auxiliary_data_path)
        
        if list_costs is not None:
            self.list_costs=list_costs
        
        
        self.interpolation_acc=list(df_acc_time['accuracy'])
        self.interpolation_time=list(df_acc_time['cost_per_eval'])
        self.lower_time=min(self.interpolation_time)
        self.upper_time=max(self.interpolation_time)

        self.sample_size,self.heuristic_freq=AuxiliaryFunctions.SampleSize_TimePeriod_bisection_method(popsize,min_sample_size,perc_cost,df_acc_time)
        self.interruption_threshold=AuxiliaryFunctions.interruption_threshold(popsize,self.sample_size,df_acc_time,self.auxiliary_data_path)
        
        print('    Computing |S| and t^{period}'+ colored('   DONE     ','light_cyan'))
        print(colored('Instance of OPTECOT created.','light_yellow',attrs=["bold"]))
        sys.stdout.flush()
        print('\n')
        

    def evaluate_population(self,objective_function,population,accuracy,count_time_acc=True,count_time_gen=False,readjustment=False):
        '''
        Generate a list of scores associated with each design in the population.

        Parameters
        ==========
        population: List of solutions forming the population.
        accuracy: Accuracy set as optimal in the previous population.
        count_time_acc: True or False if you want to add or not respectively the evaluation time as additional time to adjust the accuracy.
        count_time_gen: True or False if you want to add or not respectively the evaluation time as natural time for the population evaluation.

        Return
        ======
        list_scores: List with the scores associated to each solution that forms the population.
        '''

        # Obtain scores and times per evaluation.
        list_scores=[]
        elapsed_time=0
        for solution in population:
            t=time.time()
            score=objective_function(solution, theta=int(self.theta1*accuracy))
            elapsed_time+=time.time()-t
            list_scores.append(score)

            if self.print_message is not False:

                if elapsed_time+self.time_acc+self.time_proc>self.max_time:
                    print("    Processing execution with seed "+str(self.print_message-1)+'...   '+colored('100.00%','light_cyan'),end='\r')
                    sys.stdout.flush()
                else:
                    print("    Processing execution with seed "+str(self.print_message-1)+'...   '+colored('{:.2f}'.format(((elapsed_time+self.time_acc+self.time_proc)/self.max_time)*100)+'%','light_cyan'),end='\r')
                    sys.stdout.flush()

        # Update time counters.
        if count_time_acc and not count_time_gen:
            self.time_acc+=elapsed_time
        if count_time_gen:
            self.time_proc+=elapsed_time
            if readjustment:
                self.last_time_heuristic_accepted=self.time_proc+self.time_acc

        return list_scores


    def bisection_method(self,population,train_seed):
        '''Adapted implementation of bisection method.'''

        # Initialize lower and upper limit.
        time0=self.lower_time
        time1=self.upper_time   

        # Midpoint.
        prev_m=time0
        m=(time0+time1)/2

        # Function to calculate the correlation between the rankings of the sample_size random solution using the current and maximum accuracy.
        def similarity_between_current_best_acc(acc,population,train_seed,first_iteration):
            global time_acc

            # Randomly select sample_size solutions forming the generation.
            random.seed(train_seed)
            ind_sol=random.sample(range(len(population)),self.sample_size)
            list_solutions=list(np.array(population)[ind_sol])

            # Save the scores associated with each selected solution.
            t=time.time()
            best_scores=self.evaluate_population(self.objective_function,list_solutions,1,count_time_acc=first_iteration)# Maximum accuracy. 
            new_scores=self.evaluate_population(self.objective_function,list_solutions,acc)# new accuracy. 
            last_time_acc_increase=time.time()-t

            # Obtain associated rankings.
            new_ranking=AuxiliaryFunctions.from_scores_to_ranking(new_scores)# New accuracy. 
            best_ranking=AuxiliaryFunctions.from_scores_to_ranking(best_scores)# Maximum accuracy. 
                    
            # Compare two rankings.
            metric_value=AuxiliaryFunctions.spearman_corr(new_ranking,best_ranking)

            return metric_value,last_time_acc_increase

        # Reset interval limits until the interval has a sufficiently small range.
        first_iteration=True
        stop_threshold=(time1-time0)*0.1
        while time1-time0>stop_threshold:
            metric_value,last_time_acc_increase=similarity_between_current_best_acc(np.interp(m,self.interpolation_time,self.interpolation_acc),population,train_seed,first_iteration)
            if metric_value>=self.alpha:
                time1=m
            else:
                time0=m

            prev_m=m
            m=(time0+time1)/2

            first_iteration=False
        return np.interp(prev_m,self.interpolation_time,self.interpolation_acc),last_time_acc_increase

    def execute_OPTECOT(self,gen,acc,population,train_seed,list_accuracies,list_variances):
        '''
        Running heuristics during the training process.

        Parameters
        ==========
        gen: Generation/population number in the CMA-ES algorithm.
        acc: Accuracy associated with the previous generation.
        population: List of solutions forming the population.
        train_seed: Training seed.
        list_accuracies: List with the optimal accuracies of the previous populations.
        list_variances: List with the variances of the scores of the previous populations.

        Returns
        =======
        acc: Accuracy value selected as optimal.
        time_best_acc: Evaluation time consumed in the last iteration of the bisection method.
        '''

        time_best_acc=0

        # Heuristic application: The accuracy is updated when it is detected that the variance of the scores of the last 
        # population is significantly different from the previous ones.
        if gen==0: 
            acc,time_best_acc=self.bisection_method(population,train_seed)
        else:
            # If the last kappa optimal accuracies is equal to the maximum possible optimal cost, the maximum accuracy will be considered as optimal from now on.
            if len(list_accuracies)>=self.kappa+1:    
                if self.stop_heuristic==False:
                    prev_acc=list_accuracies[-self.kappa:]
                    prev_acc_high=np.array(prev_acc)==self.interruption_threshold
                    if sum(prev_acc_high)==self.kappa:
                        self.stop_heuristic=True
                        acc=1
            
            if len(list_variances)>=self.beta+1 and self.stop_heuristic==False:
                # Compute the confidence interval.
                variance_q05=np.mean(list_variances[(-1-self.beta):-1])-2*np.std(list_variances[(-1-self.beta):-1])
                variance_q95=np.mean(list_variances[(-1-self.beta):-1])+2*np.std(list_variances[(-1-self.beta):-1])
                last_variance=list_variances[-1]
                
                # Compute the minimum accuracy with which the maximum quality is obtained.
                if last_variance<variance_q05 or last_variance>variance_q95:

                    if (self.time_proc+self.time_acc)-self.last_time_heuristic_accepted>=self.heuristic_freq:   
                        self.unused_bisection_executions+=int((self.time_proc+self.time_acc-self.last_time_heuristic_accepted)/self.heuristic_freq)-1

                        acc,time_best_acc=self.bisection_method(population,train_seed)
                    else:
                        if self.unused_bisection_executions>0:
                            acc,time_best_acc=self.bisection_method(population,train_seed)
                            self.unused_bisection_executions-=1

        # Subtract population sample evaluation time with optimum accuracy (this time will be counted in time_proc).
        self.time_acc-=time_best_acc

        return acc,bool(time_best_acc)
    
    # Como resolver el problema con un RBEA (en este caso CMA-ES) usando una funcion objetivo 
    # aproximada definida con un valor del parametro theta asociado a una precision concreta 
    def execute_CMAES_with_approximation(self,cost,seed_index,seed,n_seeds,accuracy,df):
        '''Run the CMA-ES algorithm with specific seed using approximate objective function defined by specified theta accuracy value.'''

        # Initialize CMA-ES.
        np.random.seed(seed)
        es = cma.CMAEvolutionStrategy(np.random.random(self.xdim), 0.33,inopts={'bounds': [0, 1],'seed':seed,'popsize':self.popsize,'verbose':-9})

        # Initialize time counters.
        eval_time = 0

        # Continue executing CMA-ES until the maximum time is exhausted.
        n_gen=0
        while eval_time<self.max_time:
            # New population.
            solutions = es.ask()

            # Transform the scaled values of the parameters to the real values.
            list_turb_params=[self.scaled_solution_transformer(x) for x in solutions]

            # Obtain scores and compute elapsed time.
            list_scores=[]
            for turb_params in list_turb_params:
                t=time.time()
                score=self.objective_function(turb_params, theta=int(self.theta1*accuracy))
                eval_time+=time.time()-t

                if eval_time>self.max_time:
                    print("    Processing execution with approximation of cost "+str(cost)+'...  '+colored('100.00% of the '+str(seed_index)+'/'+str(n_seeds)+' seed executed','light_cyan'),end='\r')
                    sys.stdout.flush()
                else: 
                    print("    Processing execution with approximation of cost "+str(cost)+'...   '+colored('{:.2f}'.format(round((eval_time/self.max_time)*100,2))+'% of the '+str(seed_index)+'/'+str(n_seeds)+' seed executed','light_cyan'),end='\r')
                    sys.stdout.flush()

                if self.objective_min:
                    list_scores.append(score)
                else:
                    list_scores.append(-score)

            # To build the next generation.
            es.tell(solutions, list_scores)

            # Accumulate data of interest.
            test_score=self.objective_function(self.scaled_solution_transformer(es.result.xbest))
            df.append([accuracy,seed,n_gen,test_score,eval_time])

            n_gen+=1

    def execute_CMAES_with_approximations(self,n_seeds,list_costs):

        self.list_costs=list_costs

        list_seeds=range(2,2+n_seeds)
        
        list_acc=[]
        for cost in list_costs:
            acc,_=AuxiliaryFunctions.from_cost_to_theta(cost,self.theta0,self.theta1)
            list_acc.append(round(float(acc),1))

        print("Executing CMA-ES using approximate objective functions:")
        df=[]
        for i in range(len(list_acc)):
            sys.stdout.flush()
            df=[]
            accuracy=list_acc[i]
            for j in range(len(list_seeds)):
                seed=list_seeds[j]
                self.execute_CMAES_with_approximation(self.list_costs[i],j+1,seed,n_seeds,accuracy,df)

            # Save database.
            df=pd.DataFrame(df,columns=['accuracy','seed','n_gen','score','elapsed_time'])
            df.to_csv(self.data_path+'/df_ConstantAnalysis_cost'+'{:.02f}'.format(self.list_costs[i])+'.csv')
            print("    Processing execution with approximation of cost "+str(self.list_costs[i])+colored('    DONE'+' '*29,'light_cyan'))
            sys.stdout.flush()
        print(colored('CMA-ES executed with all approximations.','light_yellow',attrs=["bold"]))
        sys.stdout.flush()
        print('\n')



    # Como resolvemos el problema con un RBEA (en este caso CMA-ES) aplicando el heuristico OPTECOT
    def execute_CMAES_with_OPTECOT(self,n_seeds):
        '''Run the CMA-ES algorithm with specific seed applying OPTECOT.'''

        print("Executing CMA-ES appliying OPTECOT:")
        sys.stdout.flush()

        list_seeds=range(2,2+n_seeds,1) 
        df=[] 

        for seed in list_seeds:
            self.print_message=seed
            self.time_proc=0
            self.time_acc=0
            self.last_time_heuristic_accepted=0
            self.stop_heuristic=False
            self.unused_bisection_executions=0
            
            # Initialize CMA-ES.
            np.random.seed(seed)
            es = cma.CMAEvolutionStrategy(np.random.random(self.xdim), 0.33,inopts={'bounds': [0, 1],'seed':seed,'popsize':self.popsize,'verbose':-9})


            # Until the maximum time is exhausted continue with CMA-ES execution and stored data of interest per population.
            n_gen=0
            while self.time_acc+self.time_proc<self.max_time:

                print("    Processing execution with seed "+str(seed-1)+'...   '+colored('{:.2f}'.format(((self.time_acc+self.time_proc)/self.max_time)*100)+'%','light_cyan'),end='\r')
                sys.stdout.flush()

                # New population.
                solutions = es.ask()

                # Transform the scaled values of the parameters to the real values.
                list_turb_params=[self.scaled_solution_transformer(x) for x in solutions]

                # Apply OPTECOT to compute the optimal accuracy.
                if n_gen==0:
                    list_accuracies=[]
                    list_variances=[]
                    accuracy=None
                else:
                    df_seed=pd.DataFrame(df)
                    df_seed=df_seed[df_seed[1]==seed]
                    list_accuracies=list(df_seed[5])
                    list_variances=list(df_seed[6])


                accuracy,readjustment=self.execute_OPTECOT(n_gen,accuracy,list_turb_params,seed,list_accuracies,list_variances)

                # Evaluate population with the optimal accuracy.
                list_scores=self.evaluate_population(self.objective_function,list_turb_params,accuracy,count_time_gen=True,readjustment=readjustment)

                if self.objective_min==False:
                    list_scores=[-score for score in list_scores]


                # To build the following population.
                es.tell(solutions, list_scores)

                # Accumulate data of interest.
                test_score= self.objective_function(self.scaled_solution_transformer(es.result.xbest))
                df.append([seed,n_gen,test_score,readjustment,accuracy,np.var(list_scores),self.time_proc,self.time_acc,self.time_acc+self.time_proc])

                n_gen+=1

            print("    Processing execution with seed "+str(seed-1)+colored('    DONE       ','light_cyan'))
            sys.stdout.flush()

        df=pd.DataFrame(df,columns=['seed','n_gen','score','readjustment','accuracy','variance','elapsed_time_proc','elapsed_time_acc','elapsed_time'])
        df.to_csv(self.data_path+'/df_OPTECOT.csv')
        print(colored('CMA-ES executed appliying OPTECOT with all seeds.','light_yellow',attrs=["bold"]))
        sys.stdout.flush()
        print('\n')



    

