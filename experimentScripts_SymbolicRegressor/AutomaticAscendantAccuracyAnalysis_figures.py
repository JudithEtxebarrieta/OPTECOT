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


# FUNCIÓN 2
# Parámetros:
#   >df_train_acc: base de datos con información extraída del proceso de búsqueda de la superficie 
#    para un valor concreto de accuracy (un tamaño del conjunto de puntos inicial).
#   >list_train_n_eval: lista con el número de evaluaciones que se desean dibujar en la gráfica.
# Devolver: lista de medias e intervalos de confianza asociados a cada número de evaluaciones destacado
# en list_train_n_eval, y calculados a partir de los datos df_train_acc almacenados para cada semilla.

def train_data_to_figure_data(df_train_acc,list_train_n_eval):

    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_n_eval in list_train_n_eval:

        # Indices de filas con un número de evaluaciones de entrenamiento menor que train_n_eval.
        ind_train=df_train_acc["n_eval"] <= train_n_eval
        
        # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
        # que menor valor de score tiene asociado.
        interest_rows=df_train_acc[ind_train].groupby("train_seed")["score"].idxmin()

        # Calcular la media y el intervalo de confianza del score.
        interest=list(df_train_acc['score'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95

# FUNCIÓN 3
# Parámetros:
#   >df_train: base de datos con información extraída del proceso de búsqueda de la superficie 
#    al usar una heurística para determinar el comportamiento ascendente del accuracy.
#   >list_train_n_eval: lista con el número de evaluaciones que se desean dibujar en la gráfica.
#   >n_curve: número que indica que la curva que se desea dibujar es la n-ésima.
# Devolver: nada, se dibuja directamente la curva en la gráfica.
def draw_ascendant_accuracy(df_train,list_train_n_eval,n_curve):
    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_n_eval in list_train_n_eval:

        # Indices de filas con un número de evaluaciones de entrenamiento menor que train_n_eval.
        ind_train=df_train["n_eval"] <= train_n_eval

        # Agrupar las filas anteriores por semillas y quedarnos con los valores máximos de accuracy.
        list_acc=list(df_train[ind_train].groupby('train_seed')['acc'].max())

        # Calcular la media y el intervalo de confianza del accuracy.
        mean,q05,q95=bootstrap_mean_and_confiance_interval(list_acc)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)
    
    # Dibujar gráfica
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[n_curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,color=colors[n_curve])
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    ax.set_title('Ascendant behavior of accuracy')

# FUNCIÓN 4
# Parámetros:
#   >corr: correlación de Spearman.
#   >acc_rest: máximo incremento que se puede hacer en el accuracy para alcanzar el máximo.
#   >param: parámetro que define la función con la cual se determinará el incremento del 
#    accuracy a partir del valor corr.
# Devolver: incremento del accuracy a considerar para la siguiente generación.
def acc_split(corr,acc_rest,param):
    if param=='logistic':
        split=(1/(1+np.exp(12*(corr-0.5))))*acc_rest
    else:
        if corr<=param[0]:
            split=acc_rest
        else:
            split=-acc_rest*(((corr-param[0])/(1-param[0]))**(1/param[1]))+acc_rest
    return split

# FUNCIÓN 5 
# Parámetros:list_params, lista con la combinación de parámetros que se consideran.
# Devolver: nada, se dibuja directamente la gráfica.
def draw_heristic2_acc_split_functions(list_params):
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
colors=px.colors.qualitative.D3+['#FECB52']
#--------------------------------------------------------------------------------------------------
# GRÁFICA 1: heurística 1.
#--------------------------------------------------------------------------------------------------
print('GRAFICA 1')

# Inicializar gráfica.
plt.figure(figsize=[13,6])
plt.subplots_adjust(left=0.08,bottom=0.11,right=0.95,top=0.88,wspace=0.98,hspace=0.76)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(100000,10100000,100000)

# Superficie.
eval_expr=str(np.load('results/data/SymbolicRegressor/expr_surf.npy'))
#__________________________________________________________________________________________________
# SUBGRÁFICA 1: 

ax=plt.subplot2grid((3, 6), (0,3), colspan=2,rowspan=2)

# Uso constante de la precisión.
accuracy=1.0
df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_train_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[0])
plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy),color=colors[0])

# Uso ascendente de la precisión.
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_heuristic1.csv', index_col=0)
list_params=dict.fromkeys(df_ascendant_acc['heuristic_param']).keys()
curve=1
for param in list_params:
    df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
    all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc_param,list_train_n_eval)
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant h1 f'+str(curve),color=colors[curve])
    curve+=1

ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Comparison between ascending and constant accuracy \n (real surface '+str(eval_expr)+')')
ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.4, 0, 0, 1), loc='center')
ax.set_yscale('log')
ax.set_xscale('log')

#__________________________________________________________________________________________________
# SUBGRÁFICA 2: 

ax=plt.subplot2grid((3, 6), (0,1), colspan=2,rowspan=2)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(3000,10103000,100000)

# Dibujar una curva para cada combinación de parámetros considerada.
n_curve=1
for param in list_params:
    df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
    draw_ascendant_accuracy(df_ascendant_acc_param,list_train_n_eval,n_curve)

    n_curve+=1
#__________________________________________________________________________________________________
# SUBGRÁFICA 3: 

# Dibujar funciones consideradas para definir el ascenso del accuracy.
draw_heristic2_acc_split_functions(list_params)

# Guardar gráfica.
plt.savefig('results/figures/SymbolicRegressor/heuristic1.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# GRÁFICA 2: heurística 2.
#--------------------------------------------------------------------------------------------------
print('GRAFICA 2')

# Inicializar gráfica.
plt.figure(figsize=[15,5])
plt.subplots_adjust(left=0.08,bottom=0.11,right=0.78,top=0.88,wspace=0.35,hspace=0.76)

#__________________________________________________________________________________________________
# SUBGRÁFICA 1: 

ax=plt.subplot(122)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(100000,10100000,100000)

# Uso constante de la precisión.
accuracy=1.0
df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_train_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[0])
plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy),color=colors[0])

# Uso ascendente de la precisión.
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_heuristic2.csv', index_col=0)
list_params=dict.fromkeys(df_ascendant_acc['heuristic_param']).keys()
curve=1
for param in list_params:
    df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
    all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc_param,list_train_n_eval)
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant h2 '+str(param),color=colors[curve])
    curve+=1

ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Comparison between ascending and constant accuracy \n (real surface '+str(eval_expr)+')')
ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.3, 0, 0, 1), loc='center')
ax.set_yscale('log')
ax.set_xscale('log')

#__________________________________________________________________________________________________
# SUBGRÁFICA 2: 

ax=plt.subplot(121)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(3000,10103000,100000)

# Dibujar una curva para cada combinación de parámetros considerada.
n_curve=1
for param in list_params:
    df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
    draw_ascendant_accuracy(df_ascendant_acc_param,list_train_n_eval,n_curve)

    n_curve+=1

# Guardar gráfica.
plt.savefig('results/figures/SymbolicRegressor/heuristic2.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# GRÁFICA 3: heurística 3.
#--------------------------------------------------------------------------------------------------
print('GRAFICA 3')

# Inicializar gráfica.
plt.figure(figsize=[15,5])
plt.subplots_adjust(left=0.08,bottom=0.11,right=0.78,top=0.88,wspace=0.35,hspace=0.76)

#__________________________________________________________________________________________________
# SUBGRÁFICA 1: 

ax=plt.subplot(122)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(100000,10100000,100000)

# Uso constante de la precisión.
accuracy=1.0
df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_train_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[0])
plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy),color=colors[0])

# Uso ascendente de la precisión.
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_heuristic3.csv', index_col=0)
list_params=dict.fromkeys(df_ascendant_acc['heuristic_param']).keys()
curve=1
for param in list_params:
    df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
    all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc_param,list_train_n_eval)
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant h3 '+str(param),color=colors[curve])
    curve+=1

ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Comparison between ascending and constant accuracy \n (real surface '+str(eval_expr)+')')
ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.3, 0, 0, 1), loc='center')
ax.set_yscale('log')
ax.set_xscale('log')

#__________________________________________________________________________________________________
# SUBGRÁFICA 2: 

ax=plt.subplot(121)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(3000,10103000,100000)

# Dibujar una curva para cada combinación de parámetros considerada.
n_curve=1
for param in list_params:
    df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
    draw_ascendant_accuracy(df_ascendant_acc_param,list_train_n_eval,n_curve)

    n_curve+=1

# Guardar gráfica.
plt.savefig('results/figures/SymbolicRegressor/heuristic3.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# GRÁFICA 4: todas las heurísticas a la vez.
#--------------------------------------------------------------------------------------------------
print('GRAFICA 4')

# Inicializar gráfica.
plt.figure(figsize=[15,5])
plt.subplots_adjust(left=0.08,bottom=0.11,right=0.78,top=0.88,wspace=0.35,hspace=0.76)

# Lista de heurísticas.
list_heuristics=np.load('results/data/SymbolicRegressor/list_heuristics.npy')

#__________________________________________________________________________________________________
# SUBGRÁFICA 1: 

ax=plt.subplot(122)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(100000,10100000,100000)

# Uso constante de la precisión.
accuracy=1.0
df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_train_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[0])
plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy),color=colors[0])

# Uso ascendente de la precisión.
curve=1
for heuristic in list_heuristics:
    df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0)
    list_params=dict.fromkeys(df_ascendant_acc['heuristic_param']).keys()
    for param in list_params:
        df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
        all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc_param,list_train_n_eval)
        if heuristic==1:
            ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant h1 f'+str(curve),color=colors[curve])
        else:
            ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=colors[curve])
            plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant h'+str(heuristic)+' '+str(param),color=colors[curve])
        curve+=1
ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Comparison between ascending and constant accuracy \n (real surface '+str(eval_expr)+')')
ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.3, 0, 0, 1), loc='center')
ax.set_yscale('log')
ax.set_xscale('log')


#__________________________________________________________________________________________________
# SUBGRÁFICA 2: 

ax=plt.subplot(121)

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(3000,10103000,100000)

# Dibujar una curva para cada combinación de parámetros considerada.
n_curve=1
for heuristic in list_heuristics:
    df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_heuristic'+str(heuristic)+'.csv', index_col=0)
    list_params=dict.fromkeys(df_ascendant_acc['heuristic_param']).keys()
    for param in list_params:
        df_ascendant_acc_param=df_ascendant_acc[df_ascendant_acc['heuristic_param']==param]
        draw_ascendant_accuracy(df_ascendant_acc_param,list_train_n_eval,n_curve)

        n_curve+=1

# Guardar gráfica.
plt.savefig('results/figures/SymbolicRegressor/all_heuristics.png')
plt.show()
plt.close()

