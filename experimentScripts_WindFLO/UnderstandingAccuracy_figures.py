
# Mediante este script se representan gráficamente los resultados numéricos calculados por 
# "UnderstandingAccuracy_data.py".

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
# FUNCIÓN 1
# Parámetros:
#   >data: datos sobre los cuales se calculará el rango entre percentiles.
#   >bootstrap_iterations: número de submuestras que se considerarán de data para poder calcular el 
#    rango entre percentiles de sus medias.
# Devolver: la media de los datos originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre data.

def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

# FUNCIÓN 2 (Obtener ranking a partir de lista conseguida tras aplicar "np.argsort" sobre una lista original)
def from_argsort_to_ranking(list):
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

# FUNCIÓN 3 (Ordenar lista según el orden para cada posición definido en argsort)
def custom_sort(list,argsort):
    new_list=[]
    for index in argsort:
        new_list.append(list[index])
    return new_list

# FUNCIÓN 4 (Cálculo de la distancia inversa normalizada de tau kendall entre dos rankings)
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

# FUNCIÓN 5 (Eliminar elementos en ciertas posiciones de una lista)
def remove_element_in_idx(list_elements,list_idx):
    new_list=[]
    for i in range(len(list_elements)):
        if i not in list_idx:
            new_list.append(list_elements[i])
    return new_list

# FUNCIÓN 6 (Construir las gráficas deseadas a partir de la base de datos)
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
        # Número de evaluaciones extra que se podrían hacer para agotar el tiempo que se necesita por defecto.
        extra_eval=int(save_time/all_means[i])
        list_extra_eval.append(extra_eval)

    plt.bar([str(acc) for acc in list_acc],list_extra_eval)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Number of extra evaluations')
    plt.xticks(rotation = 0)
    ax.set_title('Extra evaluations in the same amount of time required for maximum accuracy')


    #----------------------------------------------------------------------------------------------
    # GRÁFICA 3: ranking de las 100 soluciones con cada accuracy.
    #----------------------------------------------------------------------------------------------
    def ranking_matrix_sorted_by_max_acc(ranking_matrix):
        # Argumentos asociados al ranking del accuracy 1 ordenado de menor a mayor.
        argsort_max_acc=np.argsort(ranking_matrix[-1])

        # Reordenar filas de la matriz.
        sorted_ranking_list=[]
        for i in range(len(ranking_list)):
            sorted_ranking_list.append(custom_sort(ranking_list[i],argsort_max_acc))

        return sorted_ranking_list
        
    ax=plt.subplot(223)

    ranking_list=[]

    for accuracy in list_acc:
        df_acc=df[df['accuracy']==accuracy]
        ranking=from_argsort_to_ranking(list(df_acc['score'].argsort()))
        ranking_list.append(ranking)
    
    ranking_matrix=np.matrix(ranking_matrix_sorted_by_max_acc(ranking_list))

    
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

    # Similitud entre rankings enteros.
    list_corr=[]
    for i in range(ranking_matrix.shape[0]):
        list_corr.append(inverse_normalized_tau_kendall(ranking_list[-1],ranking_list[i]))

    plt.plot(list_acc, list_corr, linewidth=2,label='All ranking') 

    # Similitud entre el 50% inicial de los rankings.
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

    # Similitud entre el 10% inicial de los rankings.
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
    ax.set_xscale('log')

    # Guardar imagen.
    plt.savefig('results/figures/WindFLO/UnderstandingAccuracy.png')
    plt.show()
    plt.close()

# FUNCIÓN 7 (Gráficas para el paper)
def from_data_to_figures_paper(df):

    # Inicializar gráfica.
    fig=plt.figure(figsize=[15,3.5],constrained_layout=True)
    # plt.subplots_adjust(left=0.06,bottom=0.21,right=0.95,top=0.78,wspace=0.1,hspace=0.69)
    subfigs = fig.subfigures(1, 4, width_ratios=[2,2,0.1,4], wspace=0)

    #----------------------------------------------------------------------------------------------
    # GRÁFICA 1: tiempo de ejecución por accuracy y extra de evaluaciones.
    #----------------------------------------------------------------------------------------------
    list_acc=list(set(df['accuracy']))
    list_acc.sort()
    list_acc_str=[str(acc) for acc in list_acc]
    list_acc_str.reverse()

    # Tiempos de ejecución por evaluación.
    all_means=[]
    all_q05=[]
    all_q95=[]

    for accuracy in list_acc:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list(df[df['accuracy']==accuracy]['time']))
        all_means.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    # Evaluaciones extra.
    list_extra_eval=[]
    for i in range(0,len(all_means)):
        # Ahorro de tiempo.
        save_time=all_means[-1]-all_means[i]
        # Número de evaluaciones extra que se podrían hacer para agotar el tiempo que se necesita por defecto.
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
    ax.set_xticks(np.arange(0,-0.1,-0.02))
    ax.set_xticklabels(np.arange(0,0.1,0.02),rotation=0)

    ax=subfigs[1].subplots()
    for i in range(len(list_extra_eval)):
        if list_extra_eval[i]<0:
            list_extra_eval[i]=0
    list_extra_eval.reverse()
    plt.barh(list_acc_str, list_extra_eval, align='center',color=color[1])
    plt.yticks([])
    plt.title("Extra evaluations in the same amount of \n time required for maximum accuracy")
    # plt.ylabel("Accuracy")
    plt.xlabel("Evaluations")



    #----------------------------------------------------------------------------------------------
    # GRÁFICA 2: perdida de calidad en las evaluaciones considerar accuracys menores y existencia 
    # de un accuracy menor con el que se obtiene la máxima calidad.
    #----------------------------------------------------------------------------------------------
    def ranking_matrix_sorted_by_max_acc(ranking_matrix):
        # Argumentos asociados al ranking del accuracy 1 ordenado de menor a mayor.
        argsort_max_acc=np.argsort(ranking_matrix[-1])

        # Reordenar filas de la matriz.
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

    # Guardar imagen.
    plt.savefig('results/figures/WindFLO/UnderstandingAccuracy_paper.png')
    plt.show()
    plt.close()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Leer base de datos.
df=pd.read_csv('results/data/WindFLO/UnderstandingAccuracy/df_UnderstandingAccuracy.csv',index_col=0)

# Construir gráfica.
from_data_to_figures(df)
from_data_to_figures_paper(df)
