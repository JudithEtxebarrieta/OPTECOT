# Mediante este script se representan de forma gráfica los resultados numéricos obtenidos en 
# "constant_tau_save_train_data.py".

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

#==================================================================================================
# FUNCIONES
#==================================================================================================

# FUNCIÓN 1
# Parámetros:
#   >data: datos sobre los cuales se calculará el rango entre percentiles.
#   >bootstrap_iterations: número de submuestras que se considerarán de data para poder calcular el 
#    rango entre percentiles de sus medias.
# Devolver: la mediana de los datos originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre data.

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.median(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

# FUNCIÓN 2
# Parámetros:
#   >df_train_acc: base de datos con la información de entrenamiento asociada a un accuracy de time-step.
#   >list_train_steps: lista con número de steps de entrenamiento que se desean dibujar.
# Devolver: la mediana de los rewards originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre dichos datos.

def train_data_to_figure_data(df_train_acc,list_train_steps):

    # Inicializar listas para la gráfica.
    all_median_rewards=[]
    all_q05_reawrds=[]
    all_q95_rewards=[]

    # Rellenar listas.
    for train_steps in list_train_steps:
        # Indices de filas con un número de steps de entrenamiento menor que train_steps.
        ind_train=df_train_acc["info_steps"] <= train_steps
        
        # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
        # que mayor valor de reward tiene asociado.
        interest_rows=df_train_acc[ind_train].groupby("seed")["mean_reward"].idxmax()

        # Calcular la mediana y el intervalo de confianza del reward.
        interest_rewards=list(df_train_acc['mean_reward'][interest_rows])
        median_reward,q05_reward,q95_reward=bootstrap_median_and_confiance_interval(interest_rewards)

        # Guardar datos.
        all_median_rewards.append(median_reward)
        all_q05_reawrds.append(q05_reward)
        all_q95_rewards.append(q95_reward)

    return all_median_rewards,all_q05_reawrds,all_q95_rewards

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

#--------------------------------------------------------------------------------------------------
# GRÁFICA 1 (resultados generales)
#--------------------------------------------------------------------------------------------------
# Leer lista con valores de accuracy considerados.
grid_acc=np.load('results/data/grid_acc.npy')

# Lista con límites de steps de entrenamiento que se desean dibujar.
list_train_steps=range(500,10500,500)

# Ir dibujando una curva por cada valor de accuracy.
plt.figure(figsize=[8,6])
plt.subplots_adjust(left=0.12,bottom=0.11,right=0.74,top=0.88,wspace=0.2,hspace=0.2)
ax=plt.subplot(111)
for accuracy in grid_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_median_rewards,all_q05_rewards,all_q95_rewards=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Dibujar curva.
    ax.fill_between(list_train_steps,all_q05_rewards,all_q95_rewards, alpha=.5, linewidth=0)
    plt.plot(list_train_steps, all_median_rewards, linewidth=2,label=str(accuracy))

ax.set_xlabel("Train steps")
ax.set_ylabel("Median reward")
ax.set_title('Models evaluations (train 10 seeds, test 100 episodes)')
ax.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.savefig('results/figures/general_results.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# GRÁFICA 2 (mejores resultados por valor de accuracy)
#--------------------------------------------------------------------------------------------------
plt.figure(figsize=[8,6])
plt.subplots_adjust(left=0.15,bottom=0.1,right=0.74,top=0.9,wspace=0.36,hspace=0.4)

# Conseguir datos para la gráfica
train_steps=[]
max_rewards=[]
for accuracy in grid_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_median_rewards,all_q05_rewards,all_q95_rewards=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Encontrar cuando se da el máximo reward por primera vez
    ind_max=all_median_rewards.index(max(all_median_rewards))
    train_steps.append(list_train_steps[ind_max])
    max_rewards.append(all_median_rewards[ind_max])

# Dibujar la gráfica
ind_sort=np.argsort(train_steps)
train_steps_sort=[str(i) for i in sorted(train_steps)]
max_rewards_sort=[max_rewards[i] for i in ind_sort]
acc_sort=[grid_acc[i] for i in ind_sort]
acc_sort_str=[str(grid_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax=plt.subplot(111)
ax.bar(train_steps_sort,max_rewards_sort,acc_sort,label=acc_sort_str,color=colors)
ax.set_xlabel("Train steps")
ax.set_ylabel("Maximum reward")
ax.set_title('Best results for each model')
ax.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
plt.savefig('results/figures/best_results.png')
plt.show()
plt.close()