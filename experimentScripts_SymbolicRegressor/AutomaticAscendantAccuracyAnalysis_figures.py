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
#   >df_train_acc: base de datos con información extraído del proceso de búsqueda de la superficie 
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

def draw_ascendant_accuracy(df_train,list_train_n_eval,position):
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
    ax=plt.subplot(position)
    ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0,color=list(mcolors.TABLEAU_COLORS.keys())[1])
    plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy),color=list(mcolors.TABLEAU_COLORS.keys())[1])
    ax.set_xlabel("Train evaluations")
    ax.set_ylabel("Accuracy value")
    ax.set_title('Ascendant behavior of accuracy')


#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Inicializar gráfica.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.81,top=0.88,wspace=0.35,hspace=0.2)

#--------------------------------------------------------------------------------------------------
# GRÁFICA 1: comparación de scores entre usar un accuracy constante o ascendente durante el proceso
# de búsqueda de la superficie. 
#--------------------------------------------------------------------------------------------------
# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(100000,10100000,100000)

# Superficie.
eval_expr=str(np.load('results/data/SymbolicRegressor/expr_surf.npy'))

# Uso constante de la precisión.
ax=plt.subplot(122)
accuracy=1.0
df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_train_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy))

# Uso ascendente de la precisión.
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy.csv', index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant')

# Definir detalles de la gráfica y guardar.
ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Comparison between ascending and constant accuracy \n (real surface '+str(eval_expr)+')')
ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.3, 0, 0, 1), loc='center')
ax.set_yscale('log')
ax.set_xscale('log')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 2: definición del accuracy durante el procedimiento de búsqueda de la superficie.
#--------------------------------------------------------------------------------------------------
# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(1000,1001000,1000)

# Dibujar gráfica.
draw_ascendant_accuracy(df_ascendant_acc,list_train_n_eval,121)
plt.savefig('results/figures/SymbolicRegressor/AutomaticAscendantAccuracy.png')
plt.show()
plt.close()