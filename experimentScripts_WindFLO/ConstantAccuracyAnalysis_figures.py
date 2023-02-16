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
import math
#==================================================================================================
# FUNCIONES
#==================================================================================================


def bootstrap_mean_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def train_data_to_figure_data(df_train_acc,list_train_times):

    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_time in list_train_times:

        # Indices de filas con un tiempo de entrenamiento menor que train_time.
        ind_train=df_train_acc["elapsed_time"] <= train_time
        
        # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
        # que mayor valor de score tiene asociado.
        interest_rows=df_train_acc[ind_train].groupby("seed")["score"].idxmax()

        # Calcular la media y el intervalo de confianza del score.
        interest=list(df_train_acc['score'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95
 
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Inicializar gráfica.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.84,top=0.88,wspace=0.17,hspace=0.2)

#--------------------------------------------------------------------------------------------------
# GRÁFICA 1 (Mejores resultados por valor de accuracy)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 1')

# Leer lista con valores de accuracy considerados.
list_acc=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/list_acc.npy')

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
max_time=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/max_time.npy')
list_train_times=range(5,int(max_time),5)

# Conseguir datos para la gráfica.
train_times=[]
max_scores=[]
for accuracy in list_acc:

    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_times)

    # Fijar límite de evaluación de alcance de score.
    if accuracy==1:
        score_limit=all_mean_scores[-1]

    # Encontrar cuando se da el límite de score prefijado por primera vez.
    limit_scores=list(np.array(all_mean_scores)>=score_limit)
    if True in limit_scores:
        ind_min=limit_scores.index(True)
    else:
        ind_min=len(all_mean_scores)-1
    train_times.append(list_train_times[ind_min])
    max_scores.append(all_mean_scores[ind_min])


# Dibujar la gráfica.
ind_sort=np.argsort(train_times)
train_times_sort=[str(i) for i in sorted(train_times)]
max_scores_sort=[max_scores[i] for i in ind_sort]
acc_sort=[np.arange(len(list_acc)/10,0,-0.1)[i] for i in ind_sort]
acc_sort_str=[str(list_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax=plt.subplot(121)
ax.bar(train_times_sort,max_scores_sort,acc_sort,label=acc_sort_str,color=colors)
ax.set_xlabel("Train time")
ax.set_ylabel("Score (generated power)")
ax.set_title('Best results for each model')
# ax.legend(title="Train set point \n size accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='black', linestyle='--')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 2 (Resultados generales scores)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 2')

# Ir dibujando una curva por cada valor de accuracy.
ax=plt.subplot(122)
for accuracy in list_acc:

    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_times)

    # Dibujar curva.
    ax.fill_between(list_train_times,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_times, all_mean_scores, linewidth=2,label=str(accuracy))


ax.set_xlabel("Train time")
ax.set_ylabel("Score (generated power)")
ax.set_title('Model evaluation \n (train 100 seeds, test monteCarloPts=1000)')
ax.legend(title="monteCarloPts \n parameter accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='black', linestyle='--')
# plt.xscale('log')
# plt.yscale('log')


plt.savefig('results/figures/WindFLO/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()

