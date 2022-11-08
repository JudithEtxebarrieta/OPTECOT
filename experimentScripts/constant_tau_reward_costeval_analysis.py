# Este script permite comparar el coste computacional (número de evaluaciones) del entrenamiento 
# (durante un mismo tiempo) de diferentes modelos con diferentes precisiones/resoluciones
# de time-step. Al mismo tiempo, cada uno de los modelos se evalúa (en términos de la recompensa)
# en un mismo entorno de evaluación independiente al entorno de entrenamiento, pudiendo comparar así
# el tiempo de entrenamiento necesario para cada modelo para alcanzar la máxima calidad (reward)
# posible (número máximo de steps definido por episodio). 

#De los resultados obtenidos se concluye que:
#1) Para un mismo tiempo de entrenamiento, los modelos con un time-step más preciso (más pequeño) 
# tienen asociada una evaluación mejor (mayores valores de reward), a cambio de un coste 
# computacional mayor (debido a que en el tiempo de entrenamiento se realizan más evaluaciones).

#2) El descenso en el reward asociado a un modelo entrenado con un time-step constante 
# (este será el mismo durante todo el proceso de entrenamiento) menos preciso (más grande),
# es menor que el descenso provocado en el coste computacional. Es decir, el beneficio de considerar
# un time-step menos preciso es mayor que la perdida.

#Futuros enfoques:
# En lugar de considerar un time-step constante durante el entrenamiento del modelo, considerar
# un time-step descendiente a medida que va pasando el tiempo de entrenamiento. 

# Se podría reservar una cierta prima parte del tiempo de entrenamiento, para comenzar entrenando 
# el modelo con un time-step preciso. El límite de tiempo de esa primera parte se podría definir 
# con intención de asegurarse el obtener un mínimo de reward en su evaluación. A partir de ese 
# momento seguir entrenando el modelo con un time-step menos preciso, de forma que en esta segunda
# parte el número de evaluaciones (el coste) descendería. Aunque el incremento en el reward en esta
# segunda parte no sería tan grande como si se considerase la máxima precisión, con la primera parte
# ya se a garantizado la construcción de un modelo que garantiza un mínimo reward.

# Este nuevo enfoque frente a un time-step constante de menor precisión, provocaría un menor descenso 
# en el coste computacional, pero a lo mejor una menor perdida de reward.

#==================================================================================================
#LIBRERÍAS
#==================================================================================================
import stable_baselines3 # Librería que sirve para crear un modelo RL, entrenarlo y evaluarlo.
import gym # Stable-Baselines funciona en entornos que siguen la interfaz gym.
from stable_baselines3 import PPO # Importar es el modelo RL.
from stable_baselines3.ppo import MlpPolicy # Importar es la clase de política que se usará para crear las redes.
                                            # Elegimos MlpPolicy porque la entrada de CartPole es un vector de características, no imágenes.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import time
from tqdm import tqdm as tqdm

#==================================================================================================
#FUNCIONES
#==================================================================================================
# FUNCIÓN 1
# Parámetros:
#   model: modelo que se desea evaluar.
#   eval_env: entorno de evaluación.
#   init_obs: estado (episodio) inicial del entorno de evaluación.
#   seed: semilla del entorno de evaluación.
#   n_eval_episodes: número de episodios en los que se evaluará el modelo.
# Devuelve: media y desviación estándar de la recompensa obtenida en los n_eval_episodes episodios.

def evaluate(model,eval_env,init_obs,seed,n_eval_episodes):
    all_episode_reward=[]
    obs=init_obs
    eval_env.seed(seed)

    for i in range(n_eval_episodes):
            episode_rewards = 0
            done = False # Parámetro que nos indica después de cada acción si la barra sigue en equilibrio (False) o no (True)
            while not done:
                action, _states = model.predict(obs, deterministic=True) # Se predice la acción que se debe tomar con el modelo           
                obs, reward, done, info = eval_env.step(action) # Se aplica la acción en el entorno 
                episode_rewards+=reward # Se guarda la recompensa

            # Al salir del ciclo se guarda la recompensa total.
            all_episode_reward.append(episode_rewards)

            # Para cada episodio se devuelve al estado original el entorno
            obs = eval_env.reset() 

    return np.mean(all_episode_reward),np.std(all_episode_reward)

# FUNCIÓN 2
# Parámetros:
#   train_eval: entorno de entrenamiento.
#   eval_env: entorno de evaluación.
#   init_obs: estado (episodio) inicial del entorno de evaluación.
#   seed: semilla del entorno de evaluación.
#   list_train_seconds: lista con los segundos que definen el tiempo máximo de entrenamiento.
#   n_eval_episodes: número de episodios en los que se evaluará el modelo.
# Devuelve: 
#   all_mean_reward: la lista con las medias de recompensas calculadas a partir de las recompensas 
#   obtenidas en los n_eval_episodes episodios evaluados con cada modelo entrenado durante un tiempo
#   determinado (tiempos de list_train_seconds).

#   all_sd_reward: la misma lista anterior pero con las desviaciones estándares.

#   all_costeval: lista con el número de evaluaciones realizado en cada entrenamiento.

def train_and_evaluate(train_env,eval_env,init_obs,seed,list_train_seconds,n_eval_episodes):
    all_mean_reward=[]
    all_sd_reward=[]
    all_cotseval=[]

    model = PPO(MlpPolicy, train_env, verbose=1)
    tau=train_env.env.tau

    previous_train_steps=0
    for train_seconds in list_train_seconds:
        #Cuantos time-steps hay que dar en train_seconds segundos
        new_train_steps=int(train_seconds/tau)
        all_cotseval.append(new_train_steps)

        #Cuantos time-steps más hay que dar para llegar a new_train_steps
        increment_train_timesteps=new_train_steps-previous_train_steps

        #Entrenar el modelo
        model.learn(total_timesteps=increment_train_timesteps,reset_num_timesteps=False)

        #Actualizar el número de time-steps dados
        previous_train_steps+=increment_train_timesteps

        #Evaluar el modelo actualizado
        mean_reward,sd_reward=evaluate(model,eval_env,init_obs,seed,n_eval_episodes)
        all_mean_reward.append(mean_reward)
        all_sd_reward.append(sd_reward)
    
    return all_mean_reward,all_sd_reward,all_cotseval

#==================================================================================================
#PROGRAMA PRINCIPAL
#==================================================================================================

#Entorno de entrenamiento y de evaluación
train_env=gym.make('CartPole-v1')
eval_env=gym.make('CartPole-v1')

#Variables relacionadas con el entorno de evaluación
init_obs=eval_env.reset()
seed=1234
n_eval_episodes=100

#Parámetros por defecto
default_tau=0.02
default_max_episode_step=500

#Mallado de precisiones/resoluciones consideradas para el time-step(tau)
grid_acc_tau=[1.0,0.9,0.8,0.7,0.6,0.5,0.4]

#Mallado de limites de tiempo de entrenamiento considerados
list_train_seconds=range(200,420,20)

#Construir matrices que se usarán para dibujar las gráficas
matrix_mean_reward=[]
matrix_sd_reward=[]
matrix_costeval=[]

print('CALCULANDO LAS MATRICES PARA LAS GRÁFICAS')
for accuracy in tqdm(grid_acc_tau):
    #Modificar valor de time-step en el entorno de entrenamiento
    train_env.env.tau=default_tau/accuracy
    all_mean_reward,all_sd_reward,all_costeval=train_and_evaluate(train_env,eval_env,init_obs,seed,list_train_seconds,n_eval_episodes)

    #Actualizar matrices 
    matrix_mean_reward.append(all_mean_reward)
    matrix_sd_reward.append(all_sd_reward)
    matrix_costeval.append(all_costeval)

#Construir las gráfica de recompensas y guardarla
fig, ax = plt.subplots()

ind_acc=0
print('DIBUJANDO Y GUARDANDO LAS GRÁFICAS DE LAS RECOMPENSAS (con intervalo de confianza)')
for fila in tqdm(matrix_mean_reward):
    up_limit=np.array(fila)+np.array(matrix_sd_reward[ind_acc])
    low_limit=np.array(fila)-np.array(matrix_sd_reward[ind_acc])
    ax.fill_between(list_train_seconds,low_limit,up_limit, alpha=.5, linewidth=0)
    ax.plot(list_train_seconds, fila, linewidth=2,label=str(grid_acc_tau[ind_acc]))
    ind_acc+=1

plt.xlabel("Train seconds")
plt.ylabel("Eval. Median reward ("+str(n_eval_episodes)+" episodes)")
plt.legend(title="Train accuracy")

plt.savefig("results/figures/reward_analysis1.pdf")
plt.savefig("results/figures/reward_analysis1.png")
plt.close()

fig, ax = plt.subplots()

ind_acc=0
print('DIBUJANDO Y GUARDANDO LAS GRÁFICAS DE LAS RECOMPENSAS (sin intervalo de confianza)')
for fila in tqdm(matrix_mean_reward):
    ax.plot(list_train_seconds, fila, linewidth=2,label=str(grid_acc_tau[ind_acc]))
    ind_acc+=1

plt.xlabel("Train seconds")
plt.ylabel("Eval. Median reward ("+str(n_eval_episodes)+" episodes)")
plt.legend(title="Train accuracy")

plt.savefig("results/figures/reward_analysis2.pdf")
plt.savefig("results/figures/reward_analysis2.png")
plt.close()

#Construir las gráfica de tiempos de ejecución y guardarla
fig, ax = plt.subplots()

ind_acc=0
print('DIBUJANDO Y GUARDANDO LAS GRÁFICAS DE LOS TIEMPOS')
for fila in tqdm(matrix_costeval):
    ax.plot(list_train_seconds, fila, linewidth=2,label=str(grid_acc_tau[ind_acc]))
    ind_acc+=1

plt.xlabel("Train seconds")
plt.ylabel("Number of train evaluations")
plt.legend(title="Train accuracy")

plt.savefig("results/figures/costeval_analysis.pdf")
plt.savefig("results/figures/costeval_analysis.png")
plt.close()




