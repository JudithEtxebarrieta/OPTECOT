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
# Parámetros:
#   >df_train: base de datos con datos de entrenamiento.
#   >type_eval: tipo de evaluación que se quiere representar en la gráfica ('n_eval': todas las evaluaciones; 
#   'n_eval_proc': solo las evaluaciones gastadas para el calculo de los scores por generación sin el extra de evaluaciones usado para 
#    el ajuste del accuracy).
# Devolver: 
#   >all_mean: medias de los scores por límite de número de evaluaciones de entrenamiento fijados en list_train_n_eval.
#   >all_q05,all_q95: percentiles de los scores por límite de evaluaciones de entrenamiento.
#   >list_train_n_eval: lista con número de evaluaciones que se representarán en la gráfica.

def train_data_to_figure_data(df_train,type_eval):

    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Definir número de evaluaciones que se desean dibujar.
    list_train_n_eval=range(50000,1000000,10000)

    # Rellenar listas.
    for train_n_eval in list_train_n_eval:

        # Indices de filas con un número de evaluaciones de entrenamiento menor que train_n_eval.
        ind_train=df_train[type_eval] <= train_n_eval
  
        # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
        # que menor valor de score tiene asociado.
        interest_rows=df_train[ind_train].groupby("train_seed")["score"].idxmin()

        # Calcular la media y el intervalo de confianza del score.
        interest=list(df_train['score'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95,list_train_n_eval

# FUNCIÓN 3 (construcción de las curvas que muestran el comportamiento del accuracy durante el entrenamiento)
# Parámetros:
#   >df_train: base de datos con datos de entrenamiento.
#   >type_n_eval: tipo de evaluación que se quiere representar en la gráfica ('n_eval': todas las evaluaciones; 
#   'n_eval_proc': solo las evaluaciones gastadas para el calculo de los scores por generación sin el extra de evaluaciones usado para 
#    el ajuste del accuracy).
#   >curve: número que indica que curva se esta dibujando (la n-ésima curva).
# Devolver: nada directamente dibuja la gráfica con la forma del accuracy.

def draw_accuracy_behaviour(df_train,type_n_eval,curve):
    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Definir número de evaluaciones que se desean dibujar.
    list_train_n_eval=range(50000,1000000,10000)

    # Rellenar listas.
    for train_n_eval in list_train_n_eval:

        # Indices de filas con un el número de evaluaciones de entrenamiento más cercano a train_n_eval.
        ind_down=df_train[type_n_eval] <= train_n_eval
        ind_per_seed=df_train[ind_down].groupby('train_seed')[type_n_eval].idxmax()

        # Agrupar las filas anteriores por semillas y quedarnos con los valores máximos de accuracy.
        list_acc=list(df_train[ind_down].loc[ind_per_seed]['acc'])

        # Calcular la media y el intervalo de confianza del accuracy.
        mean,q05,q95=bootstrap_mean_and_confiance_interval(list_acc)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Dibujar gráfica
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,color=colors[curve])

# FUNCIÓN 4 (representación gráfica de las funciones que definen el salto de accuracy en el heurístico 1)
# Parámetros: list_params, conjunto de parámetros que dejan definida la función.
# Devolver: nada directamente dibuja la gráfica con la forma del accuracy.

def draw_heristic1_acc_split_functions(list_params):

    def acc_split(corr,acc_rest,param):
        if param=='logistic':
            split=(1/(1+np.exp(12*(corr-0.5))))*acc_rest
        else:
            if corr<=param[0]:
                split=acc_rest
            else:
                split=-acc_rest*(((corr-param[0])/(1-param[0]))**(1/param[1]))+acc_rest
        return split

    acc_rest=1
    all_x=np.arange(-1,1.1,0.1)
    curve=1
    for params in list_params:
        if params!='logistic':
            params=params.translate({ord('('):'',ord(')'):''})
            params=tuple(float(i) for i in params.split(','))        
        all_y=[]
        for x in all_x:
            all_y.append(acc_split(x,acc_rest,params))

        ax=plt.subplot2grid((3, 6), (2,curve-1), colspan=1)
        plt.plot(all_x, all_y, linewidth=2,color=colors[curve])
        ax.set_xlabel("Spearman correlation")
        ax.set_ylabel("Accuracy ascent")
        ax.set_title('Heuristic 1 f'+str(curve))
        ax.set_yticks([0,1])
        ax.set_yticklabels([0,"1-accuracy"],rotation=90)

        curve+=1

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
colors=px.colors.qualitative.D3

# Función para dibujar y guardar las gráficas según el heurístico seleccionada.
def draw_and_save_figures_per_heuristic(heuristic):

    global ax

    # Inicializar gráfica.
    if heuristic==1:
        plt.figure(figsize=[20,9])
        plt.subplots_adjust(left=0.08,bottom=0.11,right=0.82,top=0.88,wspace=0.98,hspace=0.76)
    else:
        plt.figure(figsize=[20,5])
        plt.subplots_adjust(left=0.08,bottom=0.11,right=0.76,top=0.88,wspace=0.4,hspace=0.76)

    # Superficie.
    eval_expr=str(np.load('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/expr_surf.npy'))

    #__________________________________________________________________________________________________
    # SUBGRÁFICA 1: scores durante el entrenamiento usando el número de evaluaciones total.

    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,2), colspan=2,rowspan=2)
    else:
        ax=plt.subplot(132)

    # Lectura de bases de datos que se emplearán.
    if heuristic ==11:
        df_max_acc=pd.read_csv("results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_ConstantAccuracy1_RandomTrain.csv", index_col=0) # Accuracy constante 1 (situación por defecto).
    else:
        df_max_acc=pd.read_csv("results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_ConstantAccuracy1.csv", index_col=0) # Accuracy constante 1 (situación por defecto).
    df_optimal_acc=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0) # Accuracy ascendente.

    # Inicializar número de curvas.
    curve=0

    # Uso constante de la precisión.
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precisión.
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df,'n_eval')
        ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        if heuristic==1:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' f'+str(list_params.index(param)+1),color=colors[curve])
        else:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train total evaluations")
    ax.set_ylabel("Mean score (MAE)")
    if heuristic >6:
        ax.set_title('Comparison between optimal and constant accuracy \n (real surface '+str(eval_expr)+')')
    else:
        ax.set_title('Comparison between ascendant and constant accuracy \n (real surface '+str(eval_expr)+')')

    ax.set_yscale('log')
    ax.set_xscale('log')

    #__________________________________________________________________________________________________
    # SUBGRÁFICA 2: scores durante el entrenamiento sin considerar el número de evaluaciones extra
    # necesarios para reajustar el accuracy.
    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,4), colspan=2,rowspan=2)
    else:
        ax=plt.subplot(133)

    # Inicializar número de curvas.
    curve=0

    # Uso constante de la precisión.
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval_proc')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precisión.
    list_params=list(set(df_optimal_acc['heuristic_param']))
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df,'n_eval_proc')
        ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        if heuristic==1:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' f'+str(list_params.index(param)+1),color=colors[curve])
        else:
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1

    ax.set_xlabel("Train evaluations (without extra)")
    ax.set_ylabel("Mean score (MAE)")
    if heuristic >6:
        ax.set_title('Comparison between optimal and constant accuracy \n (real surface '+str(eval_expr)+')')
    else:
        ax.set_title('Comparison between ascendant and constant accuracy \n (real surface '+str(eval_expr)+')')

    ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')
    ax.set_yscale('log')
    ax.set_xscale('log')


    #__________________________________________________________________________________________________
    # SUBGRÁFICA 3: representación gráfica del comportamiento del accuracy.
    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,0), colspan=2,rowspan=2)
    else:
        ax=plt.subplot(131)

    # Inicializar número de curvas.
    curve=1

    # Dibujar curvas.
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        draw_accuracy_behaviour(df,'n_eval',curve)
        curve+=1
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    if heuristic >6:
        ax.set_title('Behavior of optimal accuracy')
    else:
        ax.set_title('Ascendant behavior of accuracy')
                                                                                                                                                                                                                                                                                                                        
    if heuristic==1:
        #__________________________________________________________________________________________________
        # SUBGRÁFICA 3: 

        # Dibujar funciones consideradas para definir el ascenso del accuracy.
        draw_heristic1_acc_split_functions(list_params)

    plt.savefig('results/figures/SymbolicRegressor/OptimalAccuracy_h'+str(heuristic)+'.png')
    plt.show()
    plt.close()

# Llamar a la función.
list_heuristics=[1,2,3,4,5,6,7,8,9,10,11,12,13]
for heuristic in list_heuristics:
    draw_and_save_figures_per_heuristic(heuristic)