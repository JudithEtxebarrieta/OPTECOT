# En este script se representan gráficamente los resultados numéricos obtenidos durante el 
# entrenamiento y la evaluación de los modelos construidos en "constant_tau_analysis.py".

#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.colors as mcolors

#==================================================================================================
# LECTURA DE DATOS
#==================================================================================================
matrix_median_reward=np.load("results/data/matrix_median_reward.npy")
matrix_quantile05_reward=np.load("results/data/matrix_quantile05_reward.npy")
matrix_quantile95_reward=np.load("results/data/matrix_quantile95_reward.npy")
matrix_train_steps_per_eval=np.load("results/data/matrix_train_steps_per_eval.npy")
matrix_train_n_eval=np.load("results/data/matrix_train_n_eval.npy")
grid_acc_tau=np.load("results/data/grid_acc_tau.npy")
list_train_steps=np.load("results/data/list_train_steps.npy")
n_eval_episodes=np.load("results/data/n_eval_episodes.npy")

#==================================================================================================
# CONSTRUCCIÓN DE GRÁFICAS
#==================================================================================================

plt.figure(figsize=[12,7])
plt.subplots_adjust(left=0.09,bottom=0.1,right=0.93,top=0.9,wspace=0.36,hspace=0.4)

#Construir gráfica de recompensas (medianas con rango entre percentiles 5 y 95)
ax1=plt.subplot(231)
for fila in range(len(matrix_median_reward)):
    ax1.fill_between(list_train_steps,matrix_quantile05_reward[fila],matrix_quantile95_reward[fila], alpha=.5, linewidth=0)
    plt.plot(list_train_steps, matrix_median_reward[fila], linewidth=2,label=str(grid_acc_tau[fila]))

ax1.set_xlabel("Train steps")
ax1.set_ylabel("Eval. Median reward ("+str(n_eval_episodes)+" episodes)")
ax1.set_title('Model evaluation')

ax1z=plt.subplot(234)
for fila in range(len(matrix_median_reward)):
    ax1z.fill_between(list_train_steps,matrix_quantile05_reward[fila],matrix_quantile95_reward[fila], alpha=.5, linewidth=0)
    plt.plot(list_train_steps, matrix_median_reward[fila], linewidth=2,label=str(grid_acc_tau[fila]))
ax1z.set_xlim(2000, 5000) 
ax1z.set_ylim(350,510) 
ax1z.set_title('Model evaluation ZOOM')

#Construir gráfica de evaluaciones (durante el entrenamiento)
ax2=plt.subplot(232)
ind_acc=0
for fila in matrix_train_n_eval:
    plt.plot(list_train_steps, fila, linewidth=2,label=str(grid_acc_tau[ind_acc]))
    ind_acc+=1

ax2.set_xlabel("Train steps")
ax2.set_ylabel("Train evaluations")
ax2.set_title('Model training I')

ax2z=plt.subplot(235)
ind_acc=0
for fila in matrix_train_n_eval:
    plt.plot(list_train_steps, fila, linewidth=2,label=str(grid_acc_tau[ind_acc]))
    ind_acc+=1
ax2z.set_ylim(0,2000) 
ax2z.legend(title="Train time-step accuracy",bbox_to_anchor=(1.7, 0, 0, 1), loc='center')
ax2z.set_title('Model training I ZOOM')

#Construir gráfica de steps por evaluación (durante el entrenamiento)
ax3=plt.subplot(233)
ind_acc=0
for fila in matrix_train_steps_per_eval:
    plt.plot(list_train_steps, fila, linewidth=2,label=str(grid_acc_tau[ind_acc]))
    ind_acc+=1

ax3.set_xlabel("Train steps")
ax3.set_ylabel("Max. train steps per evaluation")
ax3.set_title('Model training II')

#Guardar gráficas construidas hasta el momento
plt.savefig("results/figures/reward_train_eval_analysis.pdf")
plt.savefig("results/figures/reward_train_eval_analysis.png")

plt.show()
plt.close()

#Construir gráfica que muestra la eficacia de cada modelo entrenado
plt.figure()
plt.subplots_adjust(left=0.15,bottom=0.1,right=0.74,top=0.9,wspace=0.36,hspace=0.4)
ax=plt.subplot(111)

train_steps=[]
max_rewards=[]
for i in range(len(matrix_median_reward)):
    fila=list(matrix_median_reward[i])
    ind_max=fila.index(max(fila))
    train_steps.append(list_train_steps[ind_max])
    max_rewards.append(fila[ind_max])


ind_sort=np.argsort(train_steps)
train_steps_sort=sorted(train_steps)
train_steps_sort=[str(i) for i in train_steps_sort]
max_rewards_sort=[max_rewards[i] for i in ind_sort]
acc_sort=[grid_acc_tau[i] for i in ind_sort]
acc_sort_str=[str(grid_acc_tau[i]) for i in ind_sort]
colors=[list(mcolors.TABLEAU_COLORS.keys())[i] for i in ind_sort]

ax.bar(train_steps_sort,max_rewards_sort,acc_sort,label=acc_sort_str,color=colors)
ax.set_xlabel("Train steps")
ax.set_ylabel("Maximum reward")
ax.set_title('Best results for each model')
ax.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

#Guardar última gráfica construida
plt.savefig("results/figures/reward_best_results_analysis.pdf")
plt.savefig("results/figures/reward_best_results_analysis.png")

plt.show()
plt.close()

