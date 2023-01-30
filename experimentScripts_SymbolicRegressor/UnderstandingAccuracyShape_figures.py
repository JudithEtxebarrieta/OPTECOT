#==================================================================================================
# LIBRERÍAS
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
# FUNCIONES
#==================================================================================================
def bootstrap_mean_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def from_data_to_figure(list_threshold_corr):
    # Lista de colores.
    colors=px.colors.qualitative.D3

    # Inicializar imagen.
    plt.figure(figsize=[13,5])
    plt.subplots_adjust(left=0.09,bottom=0.11,right=0.85,top=0.88,wspace=0.40,hspace=0.76)

    # SUBGRÁFICA 1
    ax=plt.subplot(121)

    curve=1
    for threshold_corr in list_threshold_corr:

        df=pd.read_csv('results/data/SymbolicRegressor/UnderstandingAccuracyShape/df_Ascendant_acc_shape_tc'+str(threshold_corr)+'.csv')

        all_mean=[]
        all_q05=[]
        all_q95=[]
        for gen in range(1,max(df['gen'])+1):
            mean,q05,q95=bootstrap_mean_and_confiance_interval(df[df['gen']==gen]['acc'])
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)

        ax.fill_between(range(1,max(df['gen'])+1),all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(range(1,max(df['gen'])+1), all_mean, linewidth=2,color=colors[curve],label=str(threshold_corr))  
        
        curve+=1
    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    ax.set_title('Ascendant accuracy shape depending \n on correlation threshold')

    # SUBGRÁFICA 2
    ax=plt.subplot(122)

    curve=1
    for threshold_corr in list_threshold_corr:

        df=pd.read_csv('results/data/SymbolicRegressor/UnderstandingAccuracyShape/df_Optimal_acc_shape_tc'+str(threshold_corr)+'.csv')

        all_mean=[]
        all_q05=[]
        all_q95=[]
        for gen in range(1,max(df['gen'])+1):
            mean,q05,q95=bootstrap_mean_and_confiance_interval(df[df['gen']==gen]['acc'])
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)

        ax.fill_between(range(1,max(df['gen'])+1),all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(range(1,max(df['gen'])+1), all_mean, linewidth=2,color=colors[curve],label=str(threshold_corr))  
        
        curve+=1
    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    ax.set_title('Optimal accuracy shape depending \n on correlation threshold')

    ax.legend(title="Correlation \n threshold",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
    plt.savefig('results/figures/SymbolicRegressor/AccuracyShape.png')
    plt.show()
    plt.close()
    

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

list_threshold_corr=np.load('results/data/SymbolicRegressor/UnderstandingAccuracyShape/list_threshold_corr.npy')
from_data_to_figure(list_threshold_corr)

