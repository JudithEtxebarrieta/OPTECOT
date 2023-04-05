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
# Parametros:
#   >df_train: base de datos con datos de entrenamiento.
#   >type_eval: tipo de evaluacion que se quiere representar en la grafica ('n_eval': todas las evaluaciones; 
#   'n_eval_proc': solo las evaluaciones gastadas para el calculo de los scores por generacion sin el extra de evaluaciones usado para 
#    el ajuste del accuracy).
# Devolver: 
#   >all_mean: medias de los scores por limite de numero de evaluaciones de entrenamiento fijados en list_train_n_eval.
#   >all_q05,all_q95: percentiles de los scores por limite de evaluaciones de entrenamiento.
#   >list_train_n_eval: lista con numero de evaluaciones que se representaran en la grafica.

def train_data_to_figure_data(df_train,type_eval):

    # Inicializar listas para la grafica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_n_eval in list_train_n_eval:

        # Indices de filas con un numero de evaluaciones de entrenamiento menor que train_n_eval.
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

# FUNCION 3 (construccion de las curvas que muestran el comportamiento del accuracy durante el entrenamiento)
# Parametros:
#   >df_train: base de datos con datos de entrenamiento.
#   >type_n_eval: tipo de evaluacion que se quiere representar en la grafica ('n_eval': todas las evaluaciones; 
#   'n_eval_proc': solo las evaluaciones gastadas para el calculo de los scores por generacion sin el extra de evaluaciones usado para 
#    el ajuste del accuracy).
#   >curve: numero que indica que curva se esta dibujando (la n-esima curva).
# Devolver: nada directamente dibuja la grafica con la forma del accuracy.

def draw_accuracy_behaviour(df_train,type_n_eval,curve):
    # Inicializar listas para la grafica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_n_eval in list_train_n_eval:

        # Indices de filas con el numero de evaluaciones de entrenamiento mas cercano a train_n_eval.
        ind_down=df_train[type_n_eval] <= train_n_eval
        ind_per_seed=df_train[ind_down].groupby('train_seed')[type_n_eval].idxmax()

        # Agrupar las filas anteriores por semillas y quedarnos con los valores de accuracy.
        list_acc=list(df_train[ind_down].loc[ind_per_seed]['acc'])

        # Calcular la media y el intervalo de confianza del accuracy.
        mean,q05,q95=bootstrap_mean_and_confiance_interval(list_acc)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Dibujar grafica
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,color=colors[curve])

# FUNCION 4 (representacion grafica de las funciones que definen el salto de accuracy en el heuristico 1)
# Parametros: list_params, conjunto de parametros que dejan definida la funcion.
# Devolver: nada directamente dibuja la grafica con la forma del accuracy.

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

# FUNCION 5 (Dibujar comportamiento del umbral del metodo de biseccion para los heuristicos 13 y 14)
def draw_heuristic13_14_threshold_shape(df_train,type_n_eval,curve):
    # Inicializar listas para la grafica.
    all_mean=[]
    all_q05=[]
    all_q95=[]


    # Rellenar listas.
    for train_n_eval in list_train_n_eval:

        # Indices de filas con el numero de evaluaciones de entrenamiento mas cercano a train_n_eval.
        ind_down=df_train[type_n_eval] <= train_n_eval
        ind_per_seed=df_train[ind_down].groupby('train_seed')[type_n_eval].idxmax()

        # Agrupar las filas anteriores por semillas y quedarnos con los valores del umbral.
        list_threshold=list(df_train[ind_down].loc[ind_per_seed]['threshold'])

        # Calcular la media y el intervalo de confianza del umbral.
        mean,q05,q95=bootstrap_mean_and_confiance_interval(list_threshold)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Dibujar grafica
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,color=colors[curve])

# FUNCION 6 (Funcion para dibujar y guardar las graficas segun el heuristico seleccionada)
def draw_and_save_figures_per_heuristic(heuristic):

    global ax

    # Inicializar grafica.
    if heuristic==1:
        plt.figure(figsize=[20,9])
        plt.subplots_adjust(left=0.08,bottom=0.11,right=0.82,top=0.88,wspace=0.98,hspace=0.76)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12,'I','II']:
        plt.figure(figsize=[20,5])
        plt.subplots_adjust(left=0.08,bottom=0.11,right=0.76,top=0.88,wspace=0.4,hspace=0.76)
    if heuristic in [13,14]:
        plt.figure(figsize=[22,5])
        plt.subplots_adjust(left=0.04,bottom=0.11,right=0.87,top=0.88,wspace=0.37,hspace=0.76)

    # Superficie.
    eval_expr=str(np.load('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/expr_surf.npy'))

    #__________________________________________________________________________________________________
    # SUBGRAFICA 1: scores durante el entrenamiento usando el numero de evaluaciones total.

    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,2), colspan=2,rowspan=2)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12,'I','II']:
        ax=plt.subplot(132)
    if heuristic in [13,14]:
        ax=plt.subplot(143)

    # Lectura de bases de datos que se emplearan.
    df_max_acc=pd.read_csv("results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_ConstantAccuracy1.csv", index_col=0) # Accuracy constante 1 (situacion por defecto).
    if heuristic in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        df_optimal_acc=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0) # Accuracy ascendente.
    if heuristic in ['I','II']:
        df_optimal_acc=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'_'+str(sample_size_freq)+'.csv', index_col=0) # Accuracy ascendente.

    # Inicializar numero de curvas.
    curve=0

    # Uso constante de la precision.
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precision.
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
    if heuristic in [7,8,9,10,11,12,13,14,'I','II']:
        ax.set_title('Comparison between optimal and constant accuracy \n (real surface '+str(eval_expr)+')')
    else:
        ax.set_title('Comparison between ascendant and constant accuracy \n (real surface '+str(eval_expr)+')')

    # ax.set_yscale('log')
    ax.set_xscale('log')

    #__________________________________________________________________________________________________
    # SUBGRAFICA 2: scores durante el entrenamiento sin considerar el numero de evaluaciones extra
    # necesarios para reajustar el accuracy.
    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,4), colspan=2,rowspan=2)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12,'I','II']:
        ax=plt.subplot(133)
    if heuristic in [13,14]:
        ax=plt.subplot(144)

    # Inicializar numero de curvas.
    curve=0

    # Uso constante de la precision.
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval_proc')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Uso ascendente de la precision.
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
    if heuristic in [7,8,9,10,11,12,13,14,'I','II']:
        ax.set_title('Comparison between optimal and constant accuracy \n (real surface '+str(eval_expr)+')')
    else:
        ax.set_title('Comparison between ascendant and constant accuracy \n (real surface '+str(eval_expr)+')')

    ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')
    # ax.set_yscale('log')
    ax.set_xscale('log')


    #__________________________________________________________________________________________________
    # SUBGRAFICA 3: representacion grafica del comportamiento del accuracy.
    if heuristic==1:
        ax=plt.subplot2grid((3, 6), (0,0), colspan=2,rowspan=2)
    if heuristic in [2,3,4,5,6,7,8,9,10,11,12,'I','II']:
        ax=plt.subplot(131)
    if heuristic in [13,14]:
        ax=plt.subplot(142)

    # Inicializar numero de curvas.
    curve=1

    # Dibujar curvas.
    for param in list_params:
        df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
        draw_accuracy_behaviour(df,'n_eval',curve)
        curve+=1
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    ax.set_xticks(range(200000,800000,200000))
    if heuristic in [7,8,9,10,11,12,13,14,'I','II']:
        ax.set_title('Behavior of optimal accuracy')
    else:
        ax.set_title('Ascendant behavior of accuracy')

    #______________________________________________________________________________________________
    # SUBGRAFICA 4:                                                                                                                                                                                                                                                                                                              
    if heuristic==1:
        # Dibujar funciones consideradas para definir el ascenso del accuracy.
        draw_heristic1_acc_split_functions(list_params)

    if heuristic in[13,14]:
        ax=plt.subplot(141)

        # Inicializar numero de curvas.
        curve=1

        # Dibujar curvas.
        for param in list_params:
            df=df_optimal_acc[df_optimal_acc['heuristic_param']==param]
            draw_heuristic13_14_threshold_shape(df,'n_eval',curve)
            curve+=1
        ax.set_xlabel("Train evaluations")
        ax.set_ylabel("Threshold value")
        ax.set_title('Behavior of bisection method threshold')

    if heuristic in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        plt.savefig('results/figures/SymbolicRegressor/OptimalAccuracy_h'+str(heuristic)+'.png')
    if heuristic in ['I','II']:
        plt.savefig('results/figures/SymbolicRegressor/OptimalAccuracy_h'+str(heuristic)+'_'+str(sample_size_freq)+'.png')
    plt.show()
    plt.close()

# FUNCION 7 (dibujar grafica comparativa)
def draw_comparative_figure(heuristic_param_list):
    global ax

    # Inicializar grafica.
    plt.figure(figsize=[15,5])
    plt.subplots_adjust(left=0.12,bottom=0.11,right=0.73,top=0.88,wspace=0.4,hspace=0.76)

    #__________________________________________________________________________________________________
    # SUBGRAFICA 1: scores durante el entrenamiento usando el numero de evaluaciones total.

    ax=plt.subplot(122)
    # Caso por defecto.
    df_max_acc=pd.read_csv("results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_ConstantAccuracy1.csv", index_col=0) # Accuracy constante 1 (situacion por defecto).
    curve=0
    all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df_max_acc,'n_eval')
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='1',color=colors[curve])
    curve+=1

    # Heuristicos seleccionados.
    for heuristic_param in heuristic_param_list:
        heuristic=heuristic_param[0]
        param=heuristic_param[1]
        df=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'_'+str(sample_size_freq)+'.csv', index_col=0)
        all_mean,all_q05,all_q95,list_train_n_eval=train_data_to_figure_data(df[df['heuristic_param']==param],'n_eval')
        ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
        plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Optimal h'+str(heuristic)+' ('+str(param)+')',color=colors[curve])
        curve+=1
    
    ax.set_xlabel("Train evaluations (without extra)")
    ax.set_ylabel("Mean score (MAE)")
    ax.set_title('Comparison of heuristics and default case')
    ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')
    ax.set_xscale('log')

    #__________________________________________________________________________________________________
    # SUBGRAFICA 2: representacion grafica del comportamiento del accuracy.

    ax=plt.subplot(121)

    # Inicializar numero de curvas.
    curve=1

    # Dibujar curvas.
    for heuristic_param in heuristic_param_list:
        heuristic=heuristic_param[0]
        param=heuristic_param[1]
        df=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'_'+str(sample_size_freq)+'.csv', index_col=0)
        draw_accuracy_behaviour(df[df['heuristic_param']==param],'n_eval',curve)
        curve+=1
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    ax.set_xticks(range(200000,800000,200000))
    ax.set_title('Behavior of optimal accuracy')

    plt.savefig('results/figures/SymbolicRegressor/OptimalAccuracyAnalysis_comparison_'+str(sample_size_freq)+'.png')
    plt.show()
    plt.close()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Lista de colores.
colors=px.colors.qualitative.D3

# Llamar a la funcion.
list_heuristics=['I','II']
heuristic_param_list=[['I',0.95],['II',10]]
for heuristic in list_heuristics:
    
    if heuristic in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        # Definir numero de evaluaciones que se desean dibujar.
        list_train_n_eval=range(50000,1000000,10000)
    else:
        # Definir forma en que se selecciona el tamaño de muestra y la frecuencia para el metodo de biseccion.
        sample_size_freq='BisectionAndPopulation'
        # sample_size_freq='BisectionOnly'

        # Definir numero de evaluaciones que se desean dibujar.
        df_max_acc=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_ConstantAccuracy1.csv', index_col=0)
        df_hI=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristicI_'+str(sample_size_freq)+'.csv', index_col=0)
        df_hII=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristicII_'+str(sample_size_freq)+'.csv', index_col=0)
        min_time_acc_max=max(df_max_acc.groupby('train_seed')['n_eval_proc'].min())
        min_time_hI=max(df_hI.groupby('train_seed')['n_eval_proc'].min())
        min_time_hII=max(df_hII.groupby('train_seed')['n_eval_proc'].min())
        max_time_acc_max=min(df_max_acc.groupby('train_seed')['n_eval_proc'].max())
        max_time_hI=min(df_hI.groupby('train_seed')['n_eval_proc'].max())
        max_time_hII=min(df_hII.groupby('train_seed')['n_eval_proc'].max())

        df_cost_per_acc=pd.read_csv('results/data/SymbolicRegressor/UnderstandingAccuracy/df_Bisection.csv',index_col=0)
        time_split=list(df_cost_per_acc['cost_per_eval'])[0]*1000

        list_train_n_eval=np.arange(max(min_time_acc_max,min_time_hI,min_time_hII),min(max_time_acc_max,max_time_hI,max_time_hII)+time_split,time_split)

    # Construir y guardar graficas.
    draw_and_save_figures_per_heuristic(heuristic)
# Construir grafica comparativa.
draw_comparative_figure(heuristic_param_list)