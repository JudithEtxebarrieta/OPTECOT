
# Mediante este script se representan graficamente los resultados numericos calculados por 
# "ConstantAccuracyAnalysis_data.py".

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
import math

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

# FUNCION 2 (para la construccion de la grafica de scores)
# Parametros:
#   >df_train_acc: base de datos con datos de entrenamiento.
#   >list_train_steps: lista con limites de numero de steps que se desean dibujar en la grafica de rewards.
# Devolver: 
#   >all_mean: medias de los rewards por limite de steps de entrenamiento fijados en list_train_steps.
#   >all_q05,all_q95: percentiles de los rewards por limite de steps de entrenamiento.

def train_data_to_figure_data(df_train_acc,list_train_steps):

    # Inicializar listas para la grafica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_steps in list_train_steps:

        # Indices de filas con un numero de steps de entrenamiento menor que train_steps.
        ind_train=df_train_acc["n_steps"] <= train_steps
        
        # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
        # que mayor valor de reward tiene asociado.
        interest_rows=df_train_acc[ind_train].groupby("train_seed")["reward"].idxmax()

        # Calcular la media y el intervalo de confianza del reward.
        interest=list(df_train_acc['reward'][interest_rows])
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95
 
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Inicializar grafica.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.15,right=0.84,top=0.9,wspace=0.17,hspace=0.2)

# Leer lista con valores de accuracy considerados.
list_acc=np.load('results/data/MuJoCo/ConstantAccuracyAnalysis/list_acc.npy')

# Definir lista con limites de tiempos de entrenamiento que se desean dibujar.
df_train_acc_min=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(min(list_acc))+".csv", index_col=0)
df_train_acc_max=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(max(list_acc))+".csv", index_col=0)
max_steps=np.load('results/data/MuJoCo/ConstantAccuracyAnalysis/max_steps.npy')
min_steps=max(df_train_acc_max.groupby("train_seed")["n_steps"].min())
split_steps=max(df_train_acc_min.groupby("train_seed")["n_steps"].min())
list_train_steps=np.arange(min_steps,max_steps,split_steps)

#--------------------------------------------------------------------------------------------------
# GRAFICA 1 (Mejores resultados por valor de accuracy)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 1')

# Conseguir datos para la grafica.
train_stepss=[]
max_scores=[]
for accuracy in list_acc:

    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la informacion relevante.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Fijar limite de evaluacion de alcance de reward.
    if accuracy==1:
        score_limit=all_mean_scores[-1]

    # Encontrar cuando se da el limite de score prefijado por primera vez.
    limit_scores=list(np.array(all_mean_scores)>=score_limit)
    if True in limit_scores:
        ind_min=limit_scores.index(True)
    else:
        ind_min=len(all_mean_scores)-1
    train_stepss.append(int(list_train_steps[ind_min]))
    max_scores.append(all_mean_scores[ind_min])


# Dibujar la grafica.
ind_sort=np.argsort(train_stepss)
train_stepss_sort=[str(i) for i in sorted(train_stepss)]
max_scores_sort=[max_scores[i] for i in ind_sort]
acc_sort=[np.arange(len(list_acc)/10,0,-0.1)[i] for i in ind_sort]
acc_sort_str=[str(list_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax=plt.subplot(121)
ax.bar(train_stepss_sort,max_scores_sort,acc_sort,label=acc_sort_str,color=colors)
ax.tick_params(axis='x', labelrotation = 45)
ax.set_xlabel("Train steps")
ax.set_ylabel("Reward")
ax.set_title('Best results for each model')
# ax.legend(title="Train time-step accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='black', linestyle='--')


#--------------------------------------------------------------------------------------------------
# GRAFICA 2 (Resultados generales scores)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 2')

# Ir dibujando una curva por cada valor de accuracy.
ax=plt.subplot(122)
for accuracy in list_acc:

    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la informacion relevante.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_steps)

    # Dibujar curva.
    ax.fill_between(list_train_steps,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_steps, all_mean_scores, linewidth=2,label=str(accuracy))


ax.set_xlabel("Train steps")
ax.set_ylabel("Reward")
ax.set_title('Model evaluation \n (train 100 seeds, test 10 episodes)')
ax.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.axhline(y=score_limit,color='black', linestyle='--')
# plt.xscale('log')
# plt.yscale('log')


plt.savefig('results/figures/MuJoCo/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()

