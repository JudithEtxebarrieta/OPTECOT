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

# FUNCIÓN 4

def draw_changes_of_accuracy(df,acc_label_height,list_train_n_eval,color):

    # Valores de accuracy considerados.
    list_acc=list(set(df['acc']))

    # Buscar y dibujar número de evaluaciones en donde se produce un cambio de accuracy.
    for acc in list_acc:
        change_n_eval=int(df[df['acc']==acc]['n_eval'].min())
        if change_n_eval<list_train_n_eval[0]:
            change_n_eval=list_train_n_eval[0]
        plt.axvline(x=change_n_eval,ymax=0.8,color=color, linestyle='--')
        plt.text(change_n_eval,ax.get_ylim()[1]+acc_label_height*0.05+0.05,str(round(acc,1)),horizontalalignment='center',verticalalignment='top',color=color)
 

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Lista con límites de número de evaluaciones de entrenamiento que se desean dibujar.
list_train_n_eval=range(100000,10100000,100000)

# Superficie.
eval_expr=str(np.load('results/data/SymbolicRegressor/expr_surf.npy'))

# Inicializar gráfica.
plt.figure(figsize=[12,7])
plt.subplots_adjust(left=0.12,bottom=0.11,right=0.75,top=0.88,wspace=0.36,hspace=0.2)

ax=plt.subplot(111)

#--------------------------------------------------------------------------------------------------
# CURVA 1:  uso constante de la máxima precisión.
#--------------------------------------------------------------------------------------------------
print('CURVA 1')
accuracy=1.0
df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_train_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy))

#--------------------------------------------------------------------------------------------------
# CURVA 2: uso constante de la precisión inferior de la que se parte para el 
# comportamiento ascendente.
#--------------------------------------------------------------------------------------------------
print('CURVA 2')
accuracy=np.load('results/data/SymbolicRegressor/init_acc.npy')
df_train_acc=pd.read_csv("results/data/SymbolicRegressor/df_train_acc"+str(accuracy)+".csv", index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_train_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label=str(accuracy))

#--------------------------------------------------------------------------------------------------
# CURVA 3: uso de una precisión ascendente con frecuencia de 500000 evaluaciones para 
# cada ascenso, y un incremento de 0.1 de accuracy.
#--------------------------------------------------------------------------------------------------
print('CURVA 3')
freq_change=500000
split_acc=0.1
acc_label_height=1
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_freq'+str(freq_change)+'_split'+str(split_acc)+'.csv', index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant freq. '+str(freq_change)+'\n split '+str(split_acc))
draw_changes_of_accuracy(df_ascendant_acc,acc_label_height,list_train_n_eval,list(mcolors.TABLEAU_COLORS.keys())[2])

#--------------------------------------------------------------------------------------------------
# CURVA 4: uso de una precisión ascendente con frecuencia de 250000 evaluaciones para 
# cada ascenso, y un incremento de 0.1 de accuracy.
#--------------------------------------------------------------------------------------------------
print('CURVA 4')
freq_change=250000
split_acc=0.1
acc_label_height=2
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_freq'+str(freq_change)+'_split'+str(split_acc)+'.csv', index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant freq. '+str(freq_change)+'\n split '+str(split_acc))
draw_changes_of_accuracy(df_ascendant_acc,acc_label_height,list_train_n_eval,list(mcolors.TABLEAU_COLORS.keys())[3])

#--------------------------------------------------------------------------------------------------
# CURVA 5: uso de una precisión ascendente con frecuencia de 250000 evaluaciones para 
# cada ascenso, y un incremento de 0.2 de accuracy.
#--------------------------------------------------------------------------------------------------
print('CURVA 5')
freq_change=250000
split_acc=0.2
acc_label_height=3
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_freq'+str(freq_change)+'_split'+str(split_acc)+'.csv', index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant freq. '+str(freq_change)+'\n split '+str(split_acc))
draw_changes_of_accuracy(df_ascendant_acc,acc_label_height,list_train_n_eval,list(mcolors.TABLEAU_COLORS.keys())[4])

#--------------------------------------------------------------------------------------------------
# CURVA 6: uso de una precisión ascendente con frecuencia de 250000 evaluaciones para 
# cada ascenso, y un incremento de 0.5 de accuracy.
#--------------------------------------------------------------------------------------------------
print('CURVA 6')
freq_change=250000
split_acc=0.5
acc_label_height=4
df_ascendant_acc=pd.read_csv('results/data/SymbolicRegressor/df_train_AscendantAccuracy_freq'+str(freq_change)+'_split'+str(split_acc)+'.csv', index_col=0)
all_mean,all_q05,all_q95=train_data_to_figure_data(df_ascendant_acc,list_train_n_eval)
ax.fill_between(list_train_n_eval,all_q05,all_q95, alpha=.5, linewidth=0)
plt.plot(list_train_n_eval, all_mean, linewidth=2,label='Ascendant freq. '+str(freq_change)+'\n split '+str(split_acc))
draw_changes_of_accuracy(df_ascendant_acc,acc_label_height,list_train_n_eval,list(mcolors.TABLEAU_COLORS.keys())[5])

# Definir detalles de la gráfica y guardar.
ax.set_xlabel("Train evaluations")
ax.set_ylabel("Mean score (MAE)")
ax.set_title('Comparison between ascending and constant accuracy \n (real surface '+str(eval_expr)+')')
ax.legend(title="Train set point size accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')
ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]+0.3)
ax.set_yscale('log')
ax.set_xscale('log')
plt.savefig('results/figures/SymbolicRegressor/AscendantAccuracy.png')
plt.show()
plt.close()