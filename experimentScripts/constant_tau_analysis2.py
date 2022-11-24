# Este script esta basado en el script "constant_tau_analysis.py", pero ahora se evalúan 
# los modelos entrenados sobre tres conjuntos de episodios diferentes:

# 1) Uno extraído de forma aleatoria de un entorno diferente al de entrenamiento.
# 2) Otro formado por una muestra con la misma distribución que el conjunto de estados 
# iniciales de los episodios utilizados para el entrenamiento.
# 3) El último formado por un subconjunto de los episodios de entrenamiento.

# NOTA.- Para ejecutar este script se han hecho las siguientes modificaciones:
# 1) "on_policy_algorithm.py": 
#    -Acceso: PP0>>OnPolicyAlgorithm (Ctrl+Click)
#    -Modificación1: línea 190 agregar como argumento new_obs

# 2) "base_class.py":
#    -Acceso: ep_info_buffer (Ctrl+Click)
#    -Modificación1: línea 523 agregar al diccionario de información new_obs
#    -Modificación2: línea 472-473 modificar valor maxlen

# Modificación 1: para poder saber cuales son los episodios que se utilizan para entrenar los modelos. 

# Modificación 2: para que el límite al entrenamiento sea el número de steps de entrenamiento definido 
# y no el número máximo de evaluaciones.

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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

#==================================================================================================
# FUNCIONES
#==================================================================================================
def significant_state_sample(original_sample,n_sample):

    #Ajustar el modelo de densidad kernel a los datos disponibles.
    kde=KernelDensity(kernel='gaussian',bandwidth=0.1).fit(original_sample)

    #Generar muestra aleatoria de modelo.
    state_sample=kde.sample(n_samples=n_sample)

    #Formato adecuado.
    matrix_state_sample=[]
    for state in state_sample:
        matrix_state_sample.append(list(state))

    return matrix_state_sample

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95),np.median(data)

def evaluate_customized_episodes(model,eval_env,eval_seed,eval_ep):
    all_episode_reward=[]
    max_episode_steps=eval_env._max_episode_steps

    eval_env.seed(eval_seed)
    eval_env=eval_env.unwrapped

    for init_state in eval_ep:

        #Definir el estado inicial del nuevo episodio.
        eval_env.state=init_state
        obs=init_state

        #Usar el modelo para evaluar en nuevo episodio.
        episode_rewards = 0
        done = False # Parámetro que nos indica después de cada acción si la evaluación sigue (False) o se ha acabado (True).
        steps=0#Contador de nsteps por episodio
        while not done and steps<max_episode_steps+1:
            action, _states = model.predict(obs, deterministic=True) # Se predice la acción que se debe tomar con el modelo.         
            obs, reward, done, info = eval_env.step(action) # Se aplica la acción en el entorno.
            episode_rewards+=reward # Se guarda la recompensa.
            steps+=1
        
        #Guardar reward de episodio.
        all_episode_reward.append(episode_rewards)

    # Calcular mediana del reward e intervalo de confianza.
    quantile05_reward,quantile95_reward,median_reward=bootstrap_median_and_confiance_interval(all_episode_reward)

    return median_reward,quantile05_reward,quantile95_reward

def evaluate(model,eval_env,eval_seed,n_eval_episodes):
    #Para guardar el reward por episodio.
    all_episode_reward=[]

    #Para garantizar que en cada llamada a la función se usarán los mismos episodios.
    eval_env.seed(eval_seed)
    obs=eval_env.reset()
    
    for i in range(n_eval_episodes):

        episode_rewards = 0
        done = False # Parámetro que nos indica después de cada acción si la evaluación sigue (False) o se ha acabado (True).
        while not done:
            action, _states = model.predict(obs, deterministic=True) # Se predice la acción que se debe tomar con el modelo.         
            obs, reward, done, info = eval_env.step(action) # Se aplica la acción en el entorno.
            episode_rewards+=reward # Se guarda la recompensa.

        # Guardar reward total de episodio.
        all_episode_reward.append(episode_rewards)

        # Para cada episodio se devuelve al estado original el entorno.
        obs = eval_env.reset() 

    # Calcular mediana del reward e intervalo de confianza.
    quantile05_reward,quantile95_reward,median_reward=bootstrap_median_and_confiance_interval(all_episode_reward)

    
    return median_reward,quantile05_reward,quantile95_reward

def train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes):
    
    #Guardar el total de episodios que se usarán para entrenar.
    model = PPO(MlpPolicy,train_env,seed=train_seed,n_steps=train_env._max_episode_steps)
    model.set_random_seed(train_seed)
    model.learn(total_timesteps=list_train_steps[-1])
    info=model.ep_info_buffer
    all_train_states=[]
    for dict in info:
        all_train_states.append(list(dict['new_obs'][0])) 
    
    #Obtener episodios de validación para evaluar los modelos sobre una muestra de episodios significativa pero diferente a los episodios usados para entrenar
    eval_ep=significant_state_sample(original_sample=all_train_states,n_sample=n_eval_episodes)
    #Obtener episodios de validación para evaluar los modelos  sobre una muestra de episodios de los episodios usados para entrenar
    train_ep=random.sample(all_train_states,n_eval_episodes)

    #Para guardar los resultados de validación.
    new_all_median_reward=[]
    new_all_q05_reward=[]
    new_all_q95_reward=[]

    sig_all_median_reward=[]
    sig_all_q05_reward=[]
    sig_all_q95_reward=[]

    sample_all_median_reward=[]
    sample_all_q05_reward=[]
    sample_all_q95_reward=[]
    
    model = PPO(MlpPolicy,train_env,seed=train_seed,n_steps=train_env._max_episode_steps)
    previous_steps=0
    for train_steps in list_train_steps:

        #Entrenar el modelo.
        model.set_random_seed(train_seed)
        model.learn(total_timesteps=train_steps-previous_steps,reset_num_timesteps=False)

        #Evaluar el modelo usando nuevos episodios.
        new_median_reward,new_q05_reward,new_q95_reward=evaluate(model,eval_env,eval_seed,n_eval_episodes)
        #Evaluar el modelo usando una submuestra significativa y diferente de los episodios de entrenamiento.
        sig_median_reward,sig_q05_reward,sig_q95_reward=evaluate_customized_episodes(model,eval_env,eval_seed,eval_ep)
        #Evaluar el modelo usando una submuestra de los episodios de entrenamiento.
        sample_median_reward,sample_q05_reward,sample_q95_reward=evaluate_customized_episodes(model,eval_env,eval_seed,train_ep)

        #Actualizaciones.
        new_all_median_reward.append(new_median_reward)
        new_all_q05_reward.append(new_q05_reward)
        new_all_q95_reward.append(new_q95_reward)

        sig_all_median_reward.append(sig_median_reward)
        sig_all_q05_reward.append(sig_q05_reward)
        sig_all_q95_reward.append(sig_q95_reward)

        sample_all_median_reward.append(sample_median_reward)
        sample_all_q05_reward.append(sample_q05_reward)
        sample_all_q95_reward.append(sample_q95_reward)

        previous_steps=train_steps
   
    
    return new_all_median_reward,new_all_q05_reward,new_all_q95_reward,sig_all_median_reward,sig_all_q05_reward,sig_all_q95_reward,sample_all_median_reward,sample_all_q05_reward,sample_all_q95_reward

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

#Entorno de entrenamiento y de evaluación.
train_env=gym.make('CartPole-v1')
eval_env=gym.make('CartPole-v1')

#Variables relacionadas con el entorno de entrenamiento.
train_seed=12345
episode_duration=10 # Duración de un episodio en segundos.

#Variables relacionadas con el entorno de evaluación.
eval_seed=54321
n_eval_episodes=100

#Parámetros por defecto.
default_tau=0.02
default_max_episode_step=500

#Mallado de precisiones/resoluciones consideradas para el time-step(tau).
grid_acc_tau=[1,0.8,0.6,0.4,0.3,0.2,0.175]

#Mallado de limite máximo de steps de entrenamiento considerados.
list_train_steps=range(2000,22000,2000)
    
#Construir matrices que se usarán para dibujar las gráficas.
matrix_new_median_reward=[]
matrix_new_q05_reward=[]
matrix_new_q95_reward=[]

matrix_sig_median_reward=[]
matrix_sig_q05_reward=[]
matrix_sig_q95_reward=[]

matrix_sample_median_reward=[]
matrix_sample_q05_reward=[]
matrix_sample_q95_reward=[]

for accuracy in tqdm(grid_acc_tau):
    #Modificar valores de parámetros en el entorno de entrenamiento y validación.
    train_env.env.tau=default_tau/accuracy
    train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
    train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps

    eval_env.env.spec.max_episode_steps = default_max_episode_step
    eval_env._max_episode_steps = eval_env.unwrapped.spec.max_episode_steps

    #Entrenamiento y evaluación.
    new_all_median_reward,new_all_q05_reward,new_all_q95_reward,sig_all_median_reward,sig_all_q05_reward,sig_all_q95_reward,sample_all_median_reward,sample_all_q05_reward,sample_all_q95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes)

    #Actualizar matrices.
    matrix_new_median_reward.append(new_all_median_reward)
    matrix_new_q05_reward.append(new_all_q05_reward)
    matrix_new_q95_reward.append(new_all_q95_reward)

    matrix_sig_median_reward.append(sig_all_median_reward)
    matrix_sig_q05_reward.append(sig_all_q05_reward)
    matrix_sig_q95_reward.append(sig_all_q95_reward)

    matrix_sample_median_reward.append(sample_all_median_reward)
    matrix_sample_q05_reward.append(sample_all_q05_reward)
    matrix_sample_q95_reward.append(sample_all_q95_reward)

#Guardar datos.
np.save("results/data/matrix_new_median_reward", matrix_new_median_reward)
np.save("results/data/matrix_new_q05_reward", matrix_new_q05_reward)
np.save("results/data/matrix_new_q95_reward", matrix_new_q95_reward)

np.save("results/data/matrix_sig_median_reward", matrix_sig_median_reward)
np.save("results/data/matrix_sig_q05_reward", matrix_sig_q05_reward)
np.save("results/data/matrix_sig_q95_reward", matrix_sig_q95_reward)

np.save("results/data/matrix_sample_median_reward", matrix_sample_median_reward)
np.save("results/data/matrix_sample_q05_reward", matrix_sample_q05_reward)
np.save("results/data/matrix_sample_q95_reward", matrix_sample_q95_reward)

np.save("results/data/grid_acc_tau2",grid_acc_tau)
np.save("results/data/list_train_steps2",list_train_steps)
np.save("results/data/n_eval_episodes2",n_eval_episodes)