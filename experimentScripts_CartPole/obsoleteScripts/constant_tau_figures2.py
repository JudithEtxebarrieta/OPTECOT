# En este script se representan gráficamente los resultados numéricos obtenidos durante el 
# entrenamiento y la evaluación de los modelos construidos en "constant_tau_analysis2.py".

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

matrix_new_median_reward=np.load("results/data/matrix_new_median_reward.npy")
matrix_new_q05_reward=np.load("results/data/matrix_new_q05_reward.npy")
matrix_new_q95_reward=np.load("results/data/matrix_new_q95_reward.npy")

matrix_sig_median_reward=np.load("results/data/matrix_sig_median_reward.npy")
matrix_sig_q05_reward=np.load("results/data/matrix_sig_q05_reward.npy")
matrix_sig_q95_reward=np.load("results/data/matrix_sig_q95_reward.npy")

matrix_sample_median_reward=np.load("results/data/matrix_sample_median_reward.npy")
matrix_sample_q05_reward=np.load("results/data/matrix_sample_q05_reward.npy")
matrix_sample_q95_reward=np.load("results/data/matrix_sample_q95_reward.npy")

grid_acc_tau=np.load("results/data/grid_acc_tau2.npy")
list_train_steps=np.load("results/data/list_train_steps2.npy")
n_eval_episodes=np.load("results/data/n_eval_episodes2.npy")


#==================================================================================================
# CONSTRUCCIÓN DE GRÁFICAS
#==================================================================================================

plt.figure(figsize=[15,5])
plt.subplots_adjust(left=0.06,bottom=0.17,right=0.89,top=0.82,wspace=0.22,hspace=0.4)

#Construir gráficas de recompensas (medianas con rango entre percentiles 5 y 95)
ax1=plt.subplot(131)
for fila in range(len(matrix_new_median_reward)):
    ax1.fill_between(list_train_steps,matrix_new_q05_reward[fila],matrix_new_q95_reward[fila], alpha=.5, linewidth=0)
    plt.plot(list_train_steps, matrix_new_median_reward[fila], linewidth=2,label=str(grid_acc_tau[fila]))

ax1.set_xlabel("Train steps")
ax1.set_ylabel("Median reward ("+str(n_eval_episodes)+" episodes)")
ax1.set_title('Model evaluation \n whit new episodes')

ax2=plt.subplot(132)
for fila in range(len(matrix_sig_median_reward)):
    ax2.fill_between(list_train_steps,matrix_sig_q05_reward[fila],matrix_sig_q95_reward[fila], alpha=.5, linewidth=0)
    plt.plot(list_train_steps, matrix_sig_median_reward[fila], linewidth=2,label=str(grid_acc_tau[fila]))

ax2.set_xlabel("Train steps")
ax2.set_ylabel("Median reward ("+str(n_eval_episodes)+" episodes)")
ax2.set_title('Model evaluation \n whit customized sample of train episodes')

ax3=plt.subplot(133)
for fila in range(len(matrix_sample_median_reward)):
    ax3.fill_between(list_train_steps,matrix_sample_q05_reward[fila],matrix_sample_q95_reward[fila], alpha=.5, linewidth=0)
    plt.plot(list_train_steps, matrix_sample_median_reward[fila], linewidth=2,label=str(grid_acc_tau[fila]))

ax3.set_xlabel("Train steps")
ax3.set_ylabel("Median reward ("+str(n_eval_episodes)+" episodes)")
ax3.set_title('Model evaluation \n whit sample of train episodes')

ax3.legend(title="Train time-step \n accuracy",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

plt.savefig('results/figures/different_evaluations.png')
plt.show()

#Construir gráfica que muestra la eficacia de cada modelo entrenado (asociada a la primera gráfica anterior).
plt.figure()
plt.subplots_adjust(left=0.15,bottom=0.1,right=0.74,top=0.9,wspace=0.36,hspace=0.4)
ax=plt.subplot(111)

train_steps=[]
max_rewards=[]
for i in range(len(matrix_new_median_reward)):
    fila=list(matrix_new_median_reward[i])
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

plt.savefig("results/figures/reward_best_results_analysis2.png")

plt.show()
plt.close()