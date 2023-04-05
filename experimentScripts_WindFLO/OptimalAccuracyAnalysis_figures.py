# Mediante este script se representan graficamente los resultados numericos calculados por 
# "OptimaltAccuracyAnalysis_data.py".

#==================================================================================================
# LIBRERIAS
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

# FUNCION 1
# Parametros:
#   >data: datos sobre los cuales se calculara el rango entre percentiles.
#   >bootstrap_iterations: numero de submuestras que se consideraran de data para poder calcular el 
#    rango entre percentiles de sus medias.
# Devolver: la media de los datos originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre data.

def bootstrap_mean_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

# FUNCION 2 (construccion de la grafica de scores)
def train_data_to_figure_data(df_train,type_eval):

    # Inicializar listas para la grafica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_time in list_train_time:

        # Indices de filas con el tiempo de entrenamiento menor que train_time.
        ind_train=df_train[type_eval] <= train_time
  
        # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
        # que mayor valor de score tiene asociado.
        interest_rows=df_train[ind_train].groupby("seed")["score"].idxmax()

        # Calcular la media y el intervalo de confianza del score.
        interest=list(df_train[ind_train].loc[interest_rows]['score'])
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95,list_train_time

# FUNCION 3 (construccion de las curvas que muestran el comportamiento del accuracy durante el entrenamiento)
def draw_accuracy_behaviour(df_train,type_time,curve):
    # Inicializar listas para la grafica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_time in list_train_time:

        # Indices de filas con el tiempo de entrenamiento mas cercano a train_time.
        ind_down=df_train[type_time] <= train_time
        ind_per_seed=df_train[ind_down].groupby('seed')[type_time].idxmax()

        # Agrupar las filas anteriores por semillas y quedarnos con los valores de accuracy.
        list_acc=list(df_train[ind_down].loc[ind_per_seed]['accuracy'])

        # Calcular la media y el intervalo de confianza del accuracy.
        mean,q05,q95=bootstrap_mean_and_confiance_interval(list_acc)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Dibujar grafica
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,color=colors[curve])

# FUNCION 4 (construccion de las curvas que muestran el comportamiento del umbral del metodo de biseccion durante el entrenamiento)
def draw_heuristic13_14_threshold_shape(df_train,type_time,curve):
    # Inicializar listas para la grafica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_time in list_train_time:

        # Indices de filas con el tiempo de evaluacion de entrenamiento mas cercano a train_time.
        ind_down=df_train[type_time] <= train_time
        ind_per_seed=df_train[ind_down].groupby('seed')[type_time].idxmax()

        # Agrupar las filas anteriores por semillas y quedarnos con los valores del umbral.
        list_threshold=list(df_train[ind_down].loc[ind_per_seed]['threshold'])

        # Calcular la media y el intervalo de confianza del umbral.
        mean,q05,q95=bootstrap_mean_and_confiance_interval(list_threshold)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Dibujar grafica
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,color=colors[curve])

# FUNCION 5 (dibujar y guardar las graficas segun el heuristico seleccionada)
def draw_and_save_figures_per_heuristic(heuristic):

    global ax

    # Inicializar grafica.
    plt.figure(figsize=[20,5])
    plt.subplots_adjust(left=0.08,bottom=0.11,right=0.76,top=0.88,wspace=0.4,hspace=0.76)

    #__________________________________________________________________________________________________
    # SUBGRAFICA 1: scores durante el entrenamiento.

    ax=plt.subplot(132)

    # Lectura de bases de datos que se emplearan.
    df_optimal_acc=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h'+str(heuristic)+'_'+str(sample_size_freq)+'.csv', index_col=0) # Accuracy ascendente.

    # Inicializar numero de curvas.
    curve=0

    # Uso constante de la precision.
    all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df_max_acc,'elapsed_time')
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precision.
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df,'elapsed_time')
        ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(list_train_time, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train total time")
    ax.set_ylabel("Score")
    ax.set_title('Comparison between optimal and constant accuracy')

    #__________________________________________________________________________________________________
    # SUBGRAFICA 2: scores durante el entrenamiento sin considerar el tiempo de evaluacion extra
    # necesarios para reajustar el accuracy.

    ax=plt.subplot(133)

    # Inicializar numero de curvas.
    curve=0

    # Uso constante de la precision.
    all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df_max_acc,'elapsed_time')
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precision.
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df,'elapsed_time_proc')
        ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(list_train_time, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train time (without extra)")
    ax.set_ylabel("Score")
    ax.set_title('Comparison between optimal and constant accuracy')
    ax.legend(title="montecraloPts \n parameter accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')



    #__________________________________________________________________________________________________
    # SUBGRAFICA 3: representacion grafica del comportamiento del accuracy.

    ax=plt.subplot(131)

    # Inicializar numero de curvas.
    curve=1

    # Dibujar curvas.
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        draw_accuracy_behaviour(df,'elapsed_time',curve)
        curve+=1
    ax.set_xlabel("Train time")
    ax.set_ylabel("Accuracy value")
    ax.set_title('Behavior of optimal accuracy')


    plt.savefig('results/figures/WindFLO/OptimalAccuracyAnalysis_h'+str(heuristic)+'_'+str(sample_size_freq)+'.png')
    plt.show()
    plt.close()

# FUNCION 6 (dibujar grafica comparativa)
def draw_comparative_figure(heuristic_param_list):
    global ax

    # Inicializar grafica.
    plt.figure(figsize=[15,5])
    plt.subplots_adjust(left=0.12,bottom=0.11,right=0.73,top=0.88,wspace=0.4,hspace=0.76)

    #__________________________________________________________________________________________________
    # SUBGRAFICA 1: scores durante el entrenamiento.

    ax=plt.subplot(122)

    # Caso por defecto.
    df_max_acc=pd.read_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv', index_col=0)
    curve=0
    all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df_max_acc,'elapsed_time')
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Heuristicos seleccionados.
    for heuristic_param in heuristic_param_list:
        heuristic=heuristic_param[0]
        param=heuristic_param[1]
        df=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h'+str(heuristic)+'_'+str(sample_size_freq)+'.csv', index_col=0)
        all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df[df['heuristic_param']==param],'elapsed_time')
        ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(list_train_time, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1
    
    ax.set_xlabel("Train time")
    ax.set_ylabel("Score")
    ax.set_title('Comparison of heuristics and default case')
    ax.legend(title="montecraloPts \n parameter accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')

    #__________________________________________________________________________________________________
    # SUBGRAFICA 2: representacion grafica del comportamiento del accuracy.

    ax=plt.subplot(121)

    # Inicializar numero de curvas.
    curve=1

    # Dibujar curvas.
    for heuristic_param in heuristic_param_list:
        heuristic=heuristic_param[0]
        param=heuristic_param[1]
        df=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h'+str(heuristic)+'_'+str(sample_size_freq)+'.csv', index_col=0)
        draw_accuracy_behaviour(df[df['heuristic_param']==param],'elapsed_time',curve)
        curve+=1
    ax.set_xlabel("Train time")
    ax.set_ylabel("Accuracy value")
    ax.set_title('Behavior of accuracy')

    plt.savefig('results/figures/WindFLO/OptimalAccuracyAnalysis_comparison_'+str(sample_size_freq)+'.png')
    plt.show()
    plt.close()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Lista de colores.
colors=px.colors.qualitative.D3

# Parametro.
# sample_size_freq='BisectionAndPopulation'
sample_size_freq='BisectionOnly'

# Definir tiempos de entrenamiento que se desean dibujar.
df_max_acc=pd.read_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv', index_col=0)
df_hI=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_hI_'+str(sample_size_freq)+'.csv', index_col=0)
df_hII=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_hII_'+str(sample_size_freq)+'.csv', index_col=0)
min_time_acc_max=max(df_max_acc.groupby('seed')['elapsed_time'].min())
min_time_hI=max(df_hI.groupby('seed')['elapsed_time'].min())
min_time_hII=max(df_hII.groupby('seed')['elapsed_time'].min())
max_time_acc_max=min(df_max_acc.groupby('seed')['elapsed_time'].max())
max_time_hI=min(df_hI.groupby('seed')['elapsed_time_proc'].max())
max_time_hII=min(df_hII.groupby('seed')['elapsed_time_proc'].max())

df_cost_per_acc=pd.read_csv('results/data/WindFLO/UnderstandingAccuracy/df_Bisection.csv',index_col=0)
time_split=list(df_cost_per_acc['cost_per_eval'])[0]*50

list_train_time=np.arange(max(min_time_acc_max,min_time_hI,min_time_hII),min(max_time_acc_max,max_time_hI,max_time_hII)+time_split,time_split)

# Llamar a la funcion.
list_heuristics=['I','II']
heuristic_param_list=[['I',0.95],['II',5]]
for heuristic in list_heuristics:
    draw_and_save_figures_per_heuristic(heuristic)
draw_comparative_figure(heuristic_param_list)