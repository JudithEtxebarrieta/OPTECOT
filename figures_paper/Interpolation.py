'''
This script is an adaptation of script experimentScripts_general/interval_interpolation_bisection_method.py
The design of the original graphs is modified to insert them in the paper.
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==================================================================================================
# FUNCTIONS
#==================================================================================================
def draw_linear_interpolation(x,y,env_name):
    '''Graph the polygonal line that interpolates a set of points.'''

    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'
    plt.figure(figsize=[3,3])
    plt.subplots_adjust(left=0.28,bottom=0.25,right=0.93,top=0.88,wspace=0.2,hspace=0.2)

    ax=plt.subplot(111)
    plt.plot(x,y,color='blue',label='Interpolation',linewidth=1,marker='o',mfc='red',mec='red')
    if env_name=='SymbolicRegressor':
        plt.title(r'\textbf{SR}',fontsize=22)
    elif env_name=='MuJoCo':
        plt.title(r'\textbf{Swimmer}',fontsize=22)
    else:
        plt.title(r'\textbf{'+str(env_name)+'}',fontsize=22)

    ax.grid(b=True,color='black',linestyle='--', linewidth=0.8,alpha=0.2,axis='both')
    ax.set_ylabel('$c$',fontsize=22)
    ax.set_xlabel('$t_c$',fontsize=22)
    if env_name=='MuJoCo':
        plt.xticks(np.arange(200,1000,300),fontsize=22)
    else:
        plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.savefig('figures_paper/figures/Interpolation/Interpolation_'+str(env_name)+'.png')
    plt.savefig('figures_paper/figures/Interpolation/Interpolation_'+str(env_name)+'.pdf')
    plt.show()

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
list_env_names=['SymbolicRegressor','WindFLO','MuJoCo','Turbines']
df=[]

def a_c(a,a_0,a_1):
    c=(a-a_0)/(a_1-a_0)
    return c

for env_name in list_env_names:
    if env_name=='WindFLO':
        a_0=0.001
    else:
        a_0=0.1 
    a_1=1
    df_acc_eval_cost=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_Bisection.csv')
    list_acc= [round(a_c(i,a_0,a_1),2) for i in df_acc_eval_cost['accuracy']]
    draw_linear_interpolation(df_acc_eval_cost['cost_per_eval'],list_acc,env_name)
