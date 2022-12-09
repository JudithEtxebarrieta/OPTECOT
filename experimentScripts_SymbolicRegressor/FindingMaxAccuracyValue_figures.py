# Mediante este script se representan gráficamente los resultados numéricos calculados por 
# "FindingMaxAccuracyValue.py".

#==================================================================================================
# LIBRERÍAS
#==================================================================================================
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pandas as pd
import tqdm

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
#   >df: base de datos con información de los scores asociados a diferentes conjuntos de puntos
#    de diferentes tamaños.
#   >position_mae: código numérico con la posición donde se desea dibujar la gráfica de los errores
#    absolutos medios.
# Devolver: nada, se dibuja directamente la gráfica.

def from_data_to_figure(df,position_mae):

    # Inicializaciones. 
    all_mean_mae=[]
    all_q05_mae=[]
    all_q95_mae=[]

    # Lista de seeds. 
    list_train_seeds=list(set(df['seed']))
    list_train_n_pts=list(set(df['n_pts']))

    # Rellenar las listas inicializadas.
    for n_pts in list_train_n_pts:

        # Seleccionar los datos de todas las semillas asociados al número de puntos fijado.
        scores_mae=df[df['n_pts']==n_pts]['score_mae']

        # Calcular intervalo de confianza y media.
        mean_mae,q05_mae,q95_mae=bootstrap_mean_and_confiance_interval(scores_mae)

        # Acumular datos.
        all_mean_mae.append(mean_mae)
        all_q05_mae.append(q05_mae)
        all_q95_mae.append(q95_mae)

    #Dibujar gráfica
    ax1=plt.subplot(position_mae)
    ax1.fill_between(list_train_n_pts,all_q05_mae,all_q95_mae, alpha=.5, linewidth=0)
    plt.plot(list_train_n_pts,all_mean_mae)
    ax1.set_xlabel("Size of train point set")
    ax1.set_ylabel("Mean MAE("+str(len(list_train_seeds))+' seeds)')
    ax1.set_title('Behavior of the MAE depending on the \n size of the train point set')

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
# EJEMPLO 1 (Paraboloide hiperbólica)
# Solución: se define en 30 puntos la máxima precisión.
#--------------------------------------------------------------------------------------------------
# Lectura de datos.
df=pd.read_csv('results/data/FindingMaxAccuracyValue1.csv',index_col=0)
eval_expr=str(np.load('results/data/eval_expr1.npy'))

# Construir gráficas.
plt.figure(figsize=[10,5])
plt.subplots_adjust(left=0.07,bottom=0.11,right=0.94,top=0.88,wspace=0.2,hspace=0.2)

from_data_to_figure(df,121)
draw_surface(122,eval_expr)

plt.savefig('results/figures/FindingAccuracyConvergenceSURF1.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# EJEMPLO 2 (Plano)
# Solución: se define en 6 puntos la máxima precisión.
#--------------------------------------------------------------------------------------------------

# Lectura de datos.
df=pd.read_csv('results/data/FindingMaxAccuracyValue2.csv',index_col=0)
eval_expr=str(np.load('results/data/eval_expr2.npy'))

# Construir gráficas.
plt.figure(figsize=[10,5])
plt.subplots_adjust(left=0.07,bottom=0.11,right=0.94,top=0.88,wspace=0.2,hspace=0.2)

from_data_to_figure(df,121)
draw_surface(122,eval_expr)

plt.savefig('results/figures/FindingAccuracyConvergenceSURF2.png')
plt.show()
plt.close()
