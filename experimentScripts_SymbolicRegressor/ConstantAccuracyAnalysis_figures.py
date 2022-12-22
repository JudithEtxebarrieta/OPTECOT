# Mediante este script se representan gráficamente los resultados numéricos calculados por 
# "ConstantAccuracyAnalysis_data.py".

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
# Parámetros:
#   >position: código numérico con la posición donde se desea dibujar la gráfica de los errores
#    absolutos medios.
#   >eval_expr: expresión de la superficie que se desea dibujar.
# Devolver: nada, se dibuja directamente la gráfica.

def draw_surface(position,eval_expr):

    # Calcular coordenadas de los puntos que se desean dibujar.
	x = np.arange(-1, 1, 1/10.)
	y = np.arange(-1, 1, 1/10.)
	x, y= np.meshgrid(x, y)
	z = eval(eval_expr)

    # Dibujar todos los puntos.
	ax = plt.subplot(position,projection='3d')
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.plot_surface(x, y, z, rstride=1, cstride=1,color='green', alpha=0.5)
	ax.set_title('Real surface: '+eval_expr)  
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

#--------------------------------------------------------------------------------------------------
# GRÁFICA 1 (Resultados generales scores)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 1')
plt.figure(figsize=[18,6])
plt.subplots_adjust(left=0.02,bottom=0.11,right=0.89,top=0.88,wspace=0.36,hspace=0.2)

# Leer lista con valores de accuracy considerados.
list_acc=np.load('results/data/SymbolicRegressor/list_acc.npy')

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(100000,10100000,100000)

# Ir dibujando una curva por cada valor de accuracy.
ax=plt.subplot(133)
for accuracy in list_acc:

    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_n_eval)

    # Dibujar curva.
    ax.fill_between(list_train_n_eval,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_n_eval, all_mean_scores, linewidth=2,label=str(accuracy))


ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Model evaluation (train 30 seeds, test 50 pts)')
ax.legend(title="Train set point \n size accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.yscale('log')

#Zoom
axzoom = plt.subplot(132)
for accuracy in list_acc:

    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_n_eval)

    # Dibujar curva.
    axzoom.fill_between(list_train_n_eval,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_n_eval, all_mean_scores, linewidth=2,label=str(accuracy))

axzoom.set_xlim(0,600000)
axzoom.set_ylim(0.05,0.37)
axzoom.set_title('ZOOM')

# Superficie.
eval_expr=str(np.load('results/data/SymbolicRegressor/expr_surf.npy'))
draw_surface(131,eval_expr)
plt.savefig('results/figures/SymbolicRegressor/general_results.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# GRÁFICA 2 (Mejores resultados por valor de accuracy)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 2')
plt.figure(figsize=[8,6])
plt.subplots_adjust(left=0.15,bottom=0.1,right=0.74,top=0.9,wspace=0.36,hspace=0.4)

# Límite de score prefijado.
score_limit=0.1

# Conseguir datos para la gráfica.
train_times=[]
max_scores=[]
for accuracy in list_acc:

    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_n_eval)

    # Encontrar cuando se da el límite de score prefijado por primera vez.
    limit_scores=list(np.array(all_mean_scores)<=score_limit)
    if True in limit_scores:
        ind_min=limit_scores.index(True)
    else:
        ind_min=len(all_mean_scores)-1
    train_times.append(list_train_n_eval[ind_min])
    max_scores.append(all_mean_scores[ind_min])

# Dibujar la gráfica.
ind_sort=np.argsort(train_times)
train_times_sort=[str(i) for i in sorted(train_times)]
max_scores_sort=[max_scores[i] for i in ind_sort]
acc_sort=[list_acc[i] for i in ind_sort]
acc_sort_str=[str(list_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax=plt.subplot(111)
ax.bar(train_times_sort,max_scores_sort,acc_sort,label=acc_sort_str,color=colors)
ax.set_xlabel("Train evaluations")
ax.set_ylabel("Score (MAE)")
ax.set_title('Best results for each model')
ax.legend(title="Train set point \n size accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='red', linestyle='--')
plt.savefig('results/figures/SymbolicRegressor/best_results.png')
plt.show()
plt.close()


