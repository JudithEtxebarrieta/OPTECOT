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
# Devolver: la media de los datos originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre data.

def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

# FUNCIÓN 2
# Parámetros:
#   >df_train_acc: base de datos con la información de entrenamiento asociada a un accuracy de time-step.
#   >list_train_steps: lista con número de steps de entrenamiento que se desean dibujar.
# Devolver: la media de los rewards originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre dichos datos.

def train_data_to_figure_data(df_train_acc,list_train_steps):

    # Inicializar listas para la gráfica.
    all_mean_rewards=[]
    all_q05_reawrds=[]
    all_q95_rewards=[]

    # Rellenar listas.
    for train_steps in list_train_steps:
        # Indices de filas con un número de steps de entrenamiento menor que train_steps.
        ind_train=df_train_acc["info_steps"] <= train_steps
        
        # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
        # que mayor valor de reward tiene asociado.
        interest_rows=df_train_acc[ind_train].groupby("seed")["mean_reward"].idxmax()

        # Calcular la media y el intervalo de confianza del reward.
        interest_rewards=list(df_train_acc['mean_reward'][interest_rows])
        mean_reward,q05_reward,q95_reward=bootstrap_mean_and_confidence_interval(interest_rewards)

        # Guardar datos.
        all_mean_rewards.append(mean_reward)
        all_q05_reawrds.append(q05_reward)
        all_q95_rewards.append(q95_reward)

    return all_mean_rewards,all_q05_reawrds,all_q95_rewards

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Inicializar gráfica.
plt.figure(figsize=[15,6])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.84,top=0.88,wspace=0.17,hspace=0.2)

# Leer lista con valores de accuracy considerados.
grid_acc=np.load('results/data/CartPole/grid_acc.npy')

# Lista con límites de steps de entrenamiento que se desean dibujar.
list_train_steps=range(500,10500,500)

#--------------------------------------------------------------------------------------------------
# GRÁFICA 1 (mejores resultados por valor de accuracy)
#--------------------------------------------------------------------------------------------------
ax=plt.subplot(121)

# Conseguir datos para la gráfica
train_steps=[]
max_rewards=[]
for accuracy in grid_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/CartPole/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_rewards,all_q05_rewards,all_q95_rewards=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Fijar límite de evaluación de alcance de score.
    if accuracy==1:
        score_limit=all_mean_rewards[-1]

    # Encontrar cuando se da el máximo reward por primera vez.
    limit_rewards=list(np.array(all_mean_rewards)>=score_limit)
    if True in limit_rewards:
        ind_max=limit_rewards.index(True)
    else:
        ind_max=len(all_mean_rewards)-1
    train_steps.append(list_train_steps[ind_max])
    max_rewards.append(all_mean_rewards[ind_max])

# Dibujar la gráfica
ind_sort=np.argsort(train_steps)
train_steps_sort=[str(i) for i in sorted(train_steps)]
max_rewards_sort=[max_rewards[i] for i in ind_sort]
acc_sort=[np.arange(len(grid_acc)/10,0,-0.1)[i] for i in ind_sort]
acc_sort_str=[str(grid_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]


ax.bar(train_steps_sort,max_rewards_sort,acc_sort,label=acc_sort_str,color=colors)
plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xlabel("Train steps")
ax.set_ylabel("Maximum reward")
ax.set_title('Best results for each model')
# ax.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

#--------------------------------------------------------------------------------------------------
# GRÁFICA 2 (resultados generales)
#--------------------------------------------------------------------------------------------------

# Ir dibujando una curva por cada valor de accuracy.
ax=plt.subplot(122)
for accuracy in grid_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/CartPole/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_rewards,all_q05_rewards,all_q95_rewards=train_data_to_figure_data(df_train_acc,list_train_steps)

    # Dibujar curva.
    ax.fill_between(list_train_steps,all_q05_rewards,all_q95_rewards, alpha=.5, linewidth=0)
    plt.plot(list_train_steps, all_mean_rewards, linewidth=2,label=str(accuracy))

plt.axhline(y=score_limit,color='black', linestyle='--')
ax.set_xlabel("Train steps")
ax.set_ylabel("Mean reward")
ax.set_title('Models evaluations (train 10 seeds, test 100 episodes)')
ax.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')



plt.savefig('results/figures/CartPole/ConstantAccuracyAnalysis.png')
plt.show()
plt.close()