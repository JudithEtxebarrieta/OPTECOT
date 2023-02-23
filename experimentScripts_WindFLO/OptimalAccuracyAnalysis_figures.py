# Mediante este script se representan gráficamente los resultados numéricos calculados por 
# "OptimaltAccuracyAnalysis_data.py".

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

# FUNCIÓN 1
# Parámetros:
#   >data: datos sobre los cuales se calculará el rango entre percentiles.
#   >bootstrap_iterations: número de submuestras que se considerarán de data para poder calcular el 
#    rango entre percentiles de sus medias.
# Devolver: la media de los datos originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre data.

def bootstrap_mean_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

# FUNCIÓN 2 (construcción de la gráfica de scores)

def train_data_to_figure_data(df_train,type_eval):

    # Inicializar listas para la gráfica.
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
        interest=list(df_train['score'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95,list_train_time

# FUNCIÓN 3 (construcción de las curvas que muestran el comportamiento del accuracy durante el entrenamiento)

def draw_accuracy_behaviour(df_train,type_time,curve):
    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_time in list_train_time:

        # Indices de filas con el tiempo de entrenamiento más cercano a train_time.
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
    
    # Dibujar gráfica
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,color=colors[curve])


def draw_heuristic13_14_threshold_shape(df_train,type_time,curve):
    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_time in list_train_time:

        # Indices de filas con el tiempo de evaluación de entrenamiento más cercano a train_time.
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
    
    # Dibujar gráfica
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,color=colors[curve])



#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
colors=px.colors.qualitative.D3

# Definir tiempos de entrenamiento que se desean dibujar.
max_time=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/max_time.npy')
df_acc_max=pd.read_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv', index_col=0)
df_h7=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h7.csv', index_col=0)
df_h9=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h9.csv', index_col=0)
df_h14=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h14.csv', index_col=0)
min_time_acc_max=max(df_acc_max.groupby('seed')['elapsed_time'].min())
min_time_h7=max(df_h7.groupby('seed')['elapsed_time'].min())
min_time_h9=max(df_h9.groupby('seed')['elapsed_time'].min())
min_time_h14=max(df_h14.groupby('seed')['elapsed_time'].min())
list_train_time=np.arange(max(min_time_acc_max,min_time_h7,min_time_h9,min_time_h14),max_time+0.03*50,0.03*50)

# Función para dibujar y guardar las gráficas según el heurístico seleccionada.
def draw_and_save_figures_per_heuristic(heuristic):

    global ax

    # Inicializar gráfica.
    if heuristic in [7,9]:
        plt.figure(figsize=[20,5])
        plt.subplots_adjust(left=0.08,bottom=0.11,right=0.76,top=0.88,wspace=0.4,hspace=0.76)
    if heuristic==14:
        plt.figure(figsize=[22,5])
        plt.subplots_adjust(left=0.04,bottom=0.11,right=0.87,top=0.88,wspace=0.37,hspace=0.76)

    #__________________________________________________________________________________________________
    # SUBGRÁFICA 1: scores durante el entrenamiento.

    if heuristic in [7,9]:
        ax=plt.subplot(132)
    if heuristic ==14:
        ax=plt.subplot(143)

    # Lectura de bases de datos que se emplearán.
    df_max_acc=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv", index_col=0) # Accuracy constante 1 (situación por defecto).
    df_optimal_acc=pd.read_csv('results/data/WindFLO/OptimalAccuracyAnalysis/df_train_OptimalAccuracyAnalysis_h'+str(heuristic)+'.csv', index_col=0) # Accuracy ascendente.

    # Inicializar número de curvas.
    curve=0

    # Uso constante de la precisión.
    all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df_max_acc,'elapsed_time')
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precisión.
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df,'elapsed_time')
        ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        if heuristic==1:
            plt.plot(list_train_time, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' f'+str(list_params.index(param)+1),color=colors[curve])
        else:
            plt.plot(list_train_time, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train total time")
    ax.set_ylabel("Score")
    ax.set_title('Comparison between optimal and constant accuracy')

    #__________________________________________________________________________________________________
    # SUBGRÁFICA 2: scores durante el entrenamiento sin considerar el tiempo de evaluación extra
    # necesarios para reajustar el accuracy.
    if heuristic in [7,9]:
        ax=plt.subplot(133)
    if heuristic ==14:
        ax=plt.subplot(144)

    # Inicializar número de curvas.
    curve=0

    # Uso constante de la precisión.
    all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df_max_acc,'elapsed_time')
    ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_time, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precisión.
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_time=train_data_to_figure_data(df,'elapsed_time_proc')
        ax.fill_between(list_train_time,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        if heuristic==1:
            plt.plot(list_train_time, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' f'+str(list_params.index(param)+1),color=colors[curve])
        else:
            plt.plot(list_train_time, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train time (without extra)")
    ax.set_ylabel("Score")
    ax.set_title('Comparison between optimal and constant accuracy')
    ax.legend(title="montecraloPts \n parameter accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')



    #__________________________________________________________________________________________________
    # SUBGRÁFICA 3: representación gráfica del comportamiento del accuracy.
    if heuristic in [7,9]:
        ax=plt.subplot(131)
    if heuristic ==14:
        ax=plt.subplot(142)

    # Inicializar número de curvas.
    curve=1

    # Dibujar curvas.
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        draw_accuracy_behaviour(df,'elapsed_time',curve)
        curve+=1
    ax.set_xlabel("Train time")
    ax.set_ylabel("Accuracy value")
    ax.set_title('Behavior of optimal accuracy')


    #______________________________________________________________________________________________
    # SUBGRÁFICA 4:                                                                                                                                                                                                                                                                                                              
    if heuristic ==14:
        ax=plt.subplot(141)

        # Inicializar número de curvas.
        curve=1

        # Dibujar curvas.
        for param in list_params:
            df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
            draw_heuristic13_14_threshold_shape(df,'elapsed_time',curve)
            curve+=1
        ax.set_xlabel("Train time")
        ax.set_ylabel("Threshold value")
        ax.set_title('Behavior of bisection method threshold')


    plt.savefig('results/figures/WindFLO/OptimalAccuracyAnalysis_h'+str(heuristic)+'.png')
    plt.show()
    plt.close()

# Llamar a la función.
list_heuristics=[7,9,14]
for heuristic in list_heuristics:
    draw_and_save_figures_per_heuristic(heuristic)