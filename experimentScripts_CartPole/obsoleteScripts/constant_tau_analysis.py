# En este script se entrenan diferentes modelos todo ellos sobre el mismo entorno pero cada uno de
# ellos con un time-step de diferente precisión, para diferentes tiempos de entrenamiento (máximo
# número de steps durante el entrenamiento) predefinidos. Después del entrenamiento, cada modelo se 
# evalúa 100 veces (en 100 episodios) en un nuevo entorno de evaluación. 

# Los resultados asociados al número de evaluaciones o steps por evaluación durante el entrenamiento,
# y recompensas obtenidas con cada uno de los modelos en el proceso de evaluación, se guardan para
# después poder acceder a ellos sin tener que volver a ejecutar el código.

# Nota.- Para ejecutar este código se han modificado las líneas 472 y 473 del script "base_class.py"
# del paquete "stable_baselines3", sustituyendo "maxlen=100" por "maxlen=20000". Este cambio se 
# realiza para que el límite de entrenamiento venga fijado por el número de steps definido y no por
# el límite de evaluaciones predefinido en ese script. El valor escogido para la modificación es 
# uno lo suficientemente grande para garantizar lo anterior.

#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import stable_baselines3 # Librería que sirve para crear un modelo RL, entrenarlo y evaluarlo.
import gym # Stable-Baselines funciona en entornos que siguen la interfaz gym.
from stable_baselines3 import PPO # Importar el modelo RL.
from stable_baselines3.ppo import MlpPolicy # Importar la clase de política que se usará para crear las redes.
                                            # Elegimos MlpPolicy porque la entrada de CartPole es un vector de características, no imágenes.
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import time
from tqdm import tqdm as tqdm

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCIÓN 1
# Parámetros:
#   >data: datos sobre los cuales se calculará el rango entre percentiles.
#   >bootstrap_iterations: número de submuestras que se considerarán de data para poder calcular el 
#    rango entre percentiles de sus medias.
# Devolver: percentiles de las medias obtenidas del submuestreo realizado sobre data.

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=10):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))

    return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95),np.median(mean_list)

# FUNCIÓN 2
# Parámetros:
#   >model: modelo que se desea evaluar.
#   >eval_env: entorno de evaluación.
#   >init_obs: estado (episodio/evaluación) inicial del entorno de evaluación.
#   >seed: semilla del entorno de evaluación.
#   >n_eval_episodes: número de episodios (evaluaciones) en los que se evaluará el modelo.
# Devuelve: mediana y rango entre percentiles de las recompensa obtenida en los n_eval_episodes episodios.

def evaluate(model,eval_env,init_obs,seed,n_eval_episodes):
    all_episode_reward=[]
    obs=init_obs
    eval_env.seed(seed)

    for i in range(n_eval_episodes):
            episode_rewards = 0
            done = False # Parámetro que nos indica después de cada acción si la evaluación sigue (False) o se ha acabado (True)
            while not done:
                action, _states = model.predict(obs, deterministic=True) # Se predice la acción que se debe tomar con el modelo           
                obs, reward, done, info = eval_env.step(action) # Se aplica la acción en el entorno 
                episode_rewards+=reward # Se guarda la recompensa

            # Al salir del ciclo se guarda la recompensa total.
            all_episode_reward.append(episode_rewards)

            # Para cada episodio se devuelve al estado original el entorno
            obs = eval_env.reset() 

    quantile05_reward,quantile95_reward,median_reward=bootstrap_median_and_confiance_interval(all_episode_reward)

    return median_reward, quantile05_reward,quantile95_reward

# FUNCIÓN 3
# Parámetros:
#   >train_eval: entorno de entrenamiento.
#   >eval_env: entorno de evaluación.
#   >init_obs: estado (episodio) inicial del entorno de evaluación.
#   >seed: semilla del entorno de evaluación.
#   >list_train_steps: lista con los números de steps que definen el tiempo máximo de entrenamiento.
#   >n_eval_episodes: número de episodios en los que se evaluará el modelo.
# Devuelve: 
#   >all_train_n_eval: lista con el número de evaluaciones realizado en cada entrenamiento.
#   >all_train_steps_per_eval: lista con la media de steps dados por evaluación en cada entrenamiento.
#   >all_median_reward: la lista con las medianas de recompensas calculadas a partir de las recompensas 
#    obtenidas en los n_eval_episodes episodios evaluados con cada modelo entrenado durante un tiempo
#    determinado (steps de list_train_steps).
#   >all_quantile05_reward, all_quantile95_reward: la misma lista anterior pero con los percentiles 5 y
#    95 respectivamente.

def train_and_evaluate(train_env,eval_env,init_obs,seed,list_train_steps,n_eval_episodes):
    all_train_n_eval=[]
    all_train_steps_per_eval=[]
    all_median_reward=[]
    all_quantile05_reward=[]
    all_quantile95_reward=[]
    
    model = PPO(MlpPolicy,train_env,seed=seed,n_steps=train_env._max_episode_steps)
    for train_steps in list_train_steps:

        #Entrenar el modelo
        model.learn(total_timesteps=train_steps,reset_num_timesteps=False)

        #Extraer información de entrenamiento
        train_info=model.ep_info_buffer
        train_n_eval=0
        list_train_steps_per_eval=[]
        for info in train_info:
            list_train_steps_per_eval.append(int(info['l'])) #Longitud (en steps) de cada evaluación
            train_n_eval+=1 #Contar número de evaluaciones realizadas en el proceso de entrenamiento

        all_train_steps_per_eval.append(max(list_train_steps_per_eval))
        all_train_n_eval.append(train_n_eval)
        
        #Evaluar el modelo actualizado
        median_reward,quantile05_reward,quantile95_reward=evaluate(model,eval_env,init_obs,seed,n_eval_episodes)
        all_median_reward.append(median_reward)
        all_quantile05_reward.append(quantile05_reward)
        all_quantile95_reward.append(quantile95_reward)
    
    return all_train_n_eval,all_train_steps_per_eval,all_median_reward,all_quantile05_reward,all_quantile95_reward

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
#Entorno de entrenamiento y de evaluación
train_env=gym.make('CartPole-v1')
eval_env=gym.make('CartPole-v1')

#Variables relacionadas con el entorno de evaluación
init_obs=eval_env.reset()
seed=12345
n_eval_episodes=100

#Parámetros por defecto
default_tau=0.02
default_max_episode_step=500

#Duración de un episodio durante el entrenamiento
episode_duration=10

#Mallado de precisiones/resoluciones consideradas para el time-step(tau)
grid_acc_tau=[1,0.8,0.6,0.4,0.3,0.2,0.175,0.15,0.125,0.1]

#Mallado de limite máximo de steps de entrenamiento considerados
list_train_steps=range(2000,6250,250)

#Construir matrices que se usarán para dibujar las gráficas
matrix_median_reward=[]
matrix_quantile05_reward=[]
matrix_quantile95_reward=[]
matrix_train_steps_per_eval=[]
matrix_train_n_eval=[]

for accuracy in tqdm(grid_acc_tau):
    #Modificar valores de parámetros en el entorno de entrenamiento
    train_env.env.tau=default_tau/accuracy
    train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
    train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps

    #Entrenamiento y evaluación
    all_train_n_eval,all_train_steps_per_eval,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,init_obs,seed,list_train_steps,n_eval_episodes)

    #Actualizar matrices 
    matrix_median_reward.append(all_median_reward)
    matrix_quantile05_reward.append(all_quantile05_reward)
    matrix_quantile95_reward.append(all_quantile95_reward)
    matrix_train_steps_per_eval.append(all_train_steps_per_eval)
    matrix_train_n_eval.append(all_train_n_eval)


#Guardar los resultados obtenidos
np.save("results/data/matrix_median_reward", matrix_median_reward)
np.save("results/data/matrix_quantile05_reward", matrix_quantile05_reward)
np.save("results/data/matrix_quantile95_reward", matrix_quantile95_reward)
np.save("results/data/matrix_train_steps_per_eval", matrix_train_steps_per_eval)
np.save("results/data/matrix_train_n_eval", matrix_train_n_eval)

np.save("results/data/grid_acc_tau",grid_acc_tau)
np.save("results/data/list_train_steps",list_train_steps)
np.save("results/data/n_eval_episodes",n_eval_episodes)