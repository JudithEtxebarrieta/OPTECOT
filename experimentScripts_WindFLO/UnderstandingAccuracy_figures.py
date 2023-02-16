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
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import scipy as sc
from itertools import combinations

#==================================================================================================
# FUNCIONES
#==================================================================================================
def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def from_argsort_to_ranking(list):
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

def inverse_normalized_tau_kendall(x,y):
    # Número de pares con orden inverso.
    pairs_reverse_order=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            case1 = x[i] < x[j] and y[i] > y[j]
            case2 = x[i] > x[j] and y[i] < y[j]

            if case1 or case2:
                pairs_reverse_order+=1  
    
    # Número de pares total.
    total_pairs=len(list(combinations(x,2)))

    # Distancia tau Kendall normalizada.
    tau_kendall=pairs_reverse_order/total_pairs

    return 1-tau_kendall

def from_data_to_figures(df):

    # Inicializar gráfica.
    plt.figure(figsize=[15,10])
    plt.subplots_adjust(left=0.06,bottom=0.11,right=0.95,top=0.88,wspace=0.2,hspace=0.37)

    #----------------------------------------------------------------------------------------------
    # GRÁFICA 1: tiempo de ejecución por accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(221)

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
    ax.set_title('Evaluation time depending on accuracy')


    #----------------------------------------------------------------------------------------------
    # GRÁFICA 2: extra de evaluaciones.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(222)

    list_total_times=list(df.groupby('accuracy')['time'].sum())

    list_extra_eval=[]
    for i in range(0,len(list_total_times)):
        # Ahorro de tiempo.
        save_time=list_total_times[-1]-list_total_times[i]
        # Número de evaluaciones extra que se podrian hacer para agotar el tiempo que se necesita por defecto.
        extra_eval=int(save_time/all_means[i])
        list_extra_eval.append(extra_eval)

    plt.bar([str(acc) for acc in list_acc],list_extra_eval)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Number of extra evaluations')
    plt.xticks(rotation = 0)
    ax.set_title('Extra evaluations in the same time required by maximum accuracy')


    #----------------------------------------------------------------------------------------------
    # GRÁFICA 3: ranking de las 100 soluciones con cada accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(223)

    ranking_list=[]

    for accuracy in list_acc:
        df_acc=df[df['accuracy']==accuracy]
        ranking=from_argsort_to_ranking(list(df_acc['score'].argsort()))
        ranking_list.append(ranking)
    
    ranking_matrix=np.matrix(ranking_list)

    
    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,ranking_matrix.shape[1]))
    color[:1, :]=np.array([248/256, 67/256, 24/256, 1])# Rojo (código rgb)
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
    # GRÁFICA 4: perdida de calidad.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(224)

    list_corr=[]
    for i in range(ranking_matrix.shape[0]):
        list_corr.append(inverse_normalized_tau_kendall(ranking_list[-1],ranking_list[i]))

    plt.plot(list_acc, list_corr, linewidth=2) 
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("1 - normalized tau Kendall")
    ax.set_title('Comparing the similarity between the best and the rest rankings')

    # Guardar imagen.
    plt.savefig('results/figures/WindFLO/UnderstandingAccuracy.png')
    plt.show()
    plt.close()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Leer base de datos.
df=pd.read_csv('results/data/WindFLO/df_UnderstandingAccuracy.csv',index_col=0)

# Construir gráfica.
from_data_to_figures(df)