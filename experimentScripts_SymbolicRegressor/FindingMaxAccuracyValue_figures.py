#==================================================================================================
# LIBRERÍAS
#==================================================================================================
from gplearn.genetic import SymbolicRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
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

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True)
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def from_data_to_figure(df,position_r2,position_mae):

    #Inicializaciones
    all_median_r2=[]
    all_q05_r2=[]
    all_q95_r2=[]
    all_median_mae=[]
    all_q05_mae=[]
    all_q95_mae=[]

    #Lista de seeds
    list_train_seeds=list(set(df['seed']))
    list_train_n_pts=list(set(df['n_pts']))

    #Rellenar las listas inicializadas
    for n_pts in list_train_n_pts:
        #Seleccionar los datos de todas las semillas asociados al número de puntos fijado
        scores_r2=df[df['n_pts']==n_pts]['score_r2']
        scores_mae=df[df['n_pts']==n_pts]['score_mae']

        #Calcular intervalo de confianza y mediana
        median_r2,q05_r2,q95_r2=bootstrap_median_and_confiance_interval(scores_r2)
        median_mae,q05_mae,q95_mae=bootstrap_median_and_confiance_interval(scores_mae)

        #Acumular datos
        all_median_r2.append(median_r2)
        all_q05_r2.append(q05_r2)
        all_q95_r2.append(q95_r2)
        all_median_mae.append(median_mae)
        all_q05_mae.append(q05_mae)
        all_q95_mae.append(q95_mae)

    #Dibujar gráfica
    ax1=plt.subplot(position_r2)
    ax1.fill_between(list_train_n_pts,all_q05_r2,all_q95_r2, alpha=.5, linewidth=0)
    plt.plot(list_train_n_pts,all_median_r2)
    ax1.set_xlabel("Size of train point set")
    ax1.set_ylabel("Median R²("+str(len(list_train_seeds))+' seeds)')
    ax1.set_title('Behavior of the R² depending on the \n size of the train point set')

    ax2=plt.subplot(position_mae)
    ax2.fill_between(list_train_n_pts,all_q05_mae,all_q95_mae, alpha=.5, linewidth=0)
    plt.plot(list_train_n_pts,all_median_mae)
    ax2.set_xlabel("Size of train point set")
    ax2.set_ylabel("Median MAE("+str(len(list_train_seeds))+' seeds)')
    ax2.set_title('Behavior of the MAE depending on the \n size of the train point set')

def draw_surface(position,eval_expr):
	x = np.arange(-1, 1, 1/10.)
	y = np.arange(-1, 1, 1/10.)
	x, y= np.meshgrid(x, y)
	z = eval(eval_expr)

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
#Lectura de datos
df=pd.read_csv('results/data/SymbolicRegressor/FindingMaxAccuracyValue1.csv',index_col=0)
eval_expr=str(np.load('results/data/SymbolicRegressor/eval_expr1.npy'))

#Construir gráficas
plt.figure(figsize=[15,5])
plt.subplots_adjust(left=0.07,bottom=0.11,right=0.94,top=0.88,wspace=0.2,hspace=0.2)

from_data_to_figure(df,131,132)
draw_surface(133,eval_expr)

plt.savefig('results/figures/SymbolicRegressor/FindingAccuracyConvergenceSURF1.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# EJEMPLO 2 (Plano)
# Solución: se define en 6 puntos la máxima precisión.
#--------------------------------------------------------------------------------------------------

#Lectura de datos
df=pd.read_csv('results/data/SymbolicRegressor/FindingMaxAccuracyValue2.csv',index_col=0)
eval_expr=str(np.load('results/data/SymbolicRegressor/eval_expr2.npy'))

#Construir gráficas
plt.figure(figsize=[15,5])
plt.subplots_adjust(left=0.07,bottom=0.11,right=0.94,top=0.88,wspace=0.2,hspace=0.2)

from_data_to_figure(df,131,132)
draw_surface(133,eval_expr)

plt.savefig('results/figures/SymbolicRegressor/FindingAccuracyConvergenceSURF2.png')
plt.show()
plt.close()
