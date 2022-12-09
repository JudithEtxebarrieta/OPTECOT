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

def bootstrap_mean_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)


def train_data_to_figure_data(df_train_acc,list_train_times,type):

    # Inicializar listas para la gráfica.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Rellenar listas.
    for train_time in list_train_times:
        # Indices de filas con un tiempo de entrenamiento menor que train_time.
        ind_train=df_train_acc["elapsed_time"] <= train_time
        
        if type=='score':
            # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
            # que menor valor de score tiene asociado.
            interest_rows=df_train_acc[ind_train].groupby("train_seed")["score"].idxmin()
        if type=='gen':
            # Agrupar las filas anteriores por la semilla y quedarnos con la fila por grupo 
            # que mayor valor de n_gen tiene asociado.
            interest_rows=df_train_acc[ind_train].groupby("train_seed")["n_gen"].idxmax()


        # Calcular la media y el intervalo de confianza del score.
        if type=='score' :
            interest=list(df_train_acc['score'][interest_rows])
        if type=='gen':
            interest=list(df_train_acc['n_gen'][interest_rows])
        if type=='time_gen':
            interest=df_train_acc[ind_train].groupby("train_seed")["time_gen"].mean()
   
        mean,q05,q95=bootstrap_mean_and_confiance_interval(interest)

        # Guardar datos.
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
plt.figure(figsize=[12,6])
plt.subplots_adjust(left=0.05,bottom=0.15,right=0.88,top=0.80,wspace=0.32,hspace=0.2)

#--------------------------------------------------------------------------------------------------
# GRÁFICA 1 (Resultados generales scores)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 1')
# Leer lista con valores de accuracy considerados.
list_acc=np.load('results/data/SymbolicRegressor/list_acc.npy')

# Lista con límites de tiempos de entrenamiento que se desean dibujar.
list_train_times=range(1,113,1)

# Ir dibujando una curva por cada valor de accuracy.
ax=plt.subplot(131)
for accuracy in list_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_scores,all_q05_scores,all_q95_scores =train_data_to_figure_data(df_train_acc,list_train_times,type='score')

    # Dibujar curva.
    ax.fill_between(list_train_times,all_q05_scores,all_q95_scores, alpha=.5, linewidth=0)
    plt.plot(list_train_times, all_mean_scores, linewidth=2,label=str(accuracy))


ax.set_xlabel("Train times")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Model evaluation (train 30 seeds, test 50 pts)')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 2 (Mejores resultados por valor de accuracy)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 2')
# Conseguir datos para la gráfica.
train_times=[]
max_scores=[]
for accuracy in list_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_scores,all_q05_scores,all_q95_scores=train_data_to_figure_data(df_train_acc,list_train_times,type='score')

    # Encontrar cuando se da el máximo score por primera vez.
    ind_max=all_mean_scores.index(min(all_mean_scores))
    train_times.append(list_train_times[ind_max])
    max_scores.append(all_mean_scores[ind_max])

# Dibujar la gráfica.
ind_sort=np.argsort(train_times)
train_times_sort=[str(i) for i in sorted(train_times)]
max_scores_sort=[max_scores[i] for i in ind_sort]
acc_sort=[list_acc[i] for i in ind_sort]
acc_sort_str=[str(list_acc[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax=plt.subplot(132)
ax.bar(train_times_sort,max_scores_sort,acc_sort,label=acc_sort_str,color=colors)
ax.set_xlabel("Train times")
ax.set_ylabel("Maximum score (MAE)")
ax.set_title('Best results for each model')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 3 (Influencia de la precisión en el número de generaciones evaluadas)
#--------------------------------------------------------------------------------------------------
print('GRAFICA 3')
# Ir dibujando una curva por cada valor de accuracy.
ax=plt.subplot(133)
for accuracy in list_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_gens,all_q05_gens,all_q95_gens =train_data_to_figure_data(df_train_acc,list_train_times,type='gen')

    # Dibujar curva.
    ax.fill_between(list_train_times,all_q05_gens,all_q95_gens, alpha=.5, linewidth=0)
    plt.plot(list_train_times, all_mean_gens, linewidth=2,label=str(accuracy))


ax.set_xlabel("Train times")
ax.set_ylabel("Mean gens (30 seeds)")
ax.set_title('Precision influence on \n evaluated generation number')
ax.legend(title="Train set point \n size accuracy",bbox_to_anchor=(1.25, 0, 0, 1), loc='center')

#Zoom
axins = zoomed_inset_axes(ax,2,loc='lower right')
for accuracy in list_acc:
    # Leer base de datos.
    df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

    # Extraer de la base de datos la información relevante.
    all_mean_gens,all_q05_gens,all_q95_gens =train_data_to_figure_data(df_train_acc,list_train_times,type='gen')

    # Dibujar curva.
    axins.fill_between(list_train_times,all_q05_gens,all_q95_gens, alpha=.5, linewidth=0)
    plt.plot(list_train_times, all_mean_gens, linewidth=2,label=str(accuracy))
axins.set_xlim(70,100)
axins.set_ylim(100,140)
#plt.xticks(visible=False)
#plt.yticks(visible=False)
mark_inset(ax,axins,loc1=1,loc2=3)


#--------------------------------------------------------------------------------------------------
# GRÁFICA 4 (Influencia de la precisión en el tiempo necesario para evaluar una generación)
#--------------------------------------------------------------------------------------------------
# print('GRAFICA 4')
# # Ir dibujando una curva por cada valor de accuracy.
# ax=plt.subplot(224)
# for accuracy in list_acc:
#     # Leer base de datos.
#     df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)

#     # Extraer de la base de datos la información relevante.
#     all_mean_gentimes,all_q05_gentimes,all_q95_gentimes =train_data_to_figure_data(df_train_acc,list_train_times,type='time_gen')

#     # Dibujar curva.
#     ax.fill_between(list_train_times,all_q05_gentimes,all_q95_gentimes, alpha=.5, linewidth=0)
#     plt.plot(list_train_times, all_mean_gentimes, linewidth=2,label=str(accuracy))


# ax.set_xlabel("Train times")
# ax.set_ylabel("Mean gens (30 seeds)")
# ax.set_title('Precision influence on \n generation evaluation time')
# ax.legend(title="Train set point \n size accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')


plt.savefig('results/figures/SymbolicRegressor/ConstantAnalysis.png')
plt.show()
plt.close()