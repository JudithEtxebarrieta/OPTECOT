# Mediante este script se guarda la información relevante extraída del proceso de entrenamiento 
# de diferentes modelos. Cada modelo esta entrenado 10 veces (10 semillas diferentes) con un 
# time-step diferente (en total se consideran 10 valores de diferentes precisiones), y las 
# validaciones se hacen sobre un entorno independiente al de entrenamiento (100 episodios) con
# una máxima precisión del time-step.

# NOTA.- Para ejecutar este script se han hecho las siguientes modificaciones:
# 1) "on_policy_algorithm.py": 
#    -Acceso: PP0>>OnPolicyAlgorithm (Ctrl+Click).
#    -Modificación1: línea 190 agregar como argumento new_obs.
# 2) "base_class.py":
#    -Acceso: ep_info_buffer (Ctrl+Click)
#    -Modificación1: línea 523 agregar al diccionario de información new_obs.
#    -Modificación2: línea 472-473 modificar valor maxlen.

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
import pandas as pd

#==================================================================================================
# CLASES
#==================================================================================================
# CLASE 1
# Se definen los métodos necesarios para medir el tiempo de ejecución durante el entrenamiento.
class stopwatch:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t

#==================================================================================================
# FUNCIONES
#==================================================================================================

# FUNCIÓN 1
# Parámetros:
#   >model: modelo que se desea evaluar.
#   >eval_env: entorno de evaluación.
#   >init_obs: estado inicial del primer episodio/evaluación del entorno de evaluación.
#   >seed: semilla del entorno de evaluación.
#   >n_eval_episodes: número de episodios (evaluaciones) en los que se evaluará el modelo.
# Devuelve: media de las recompensa obtenida en los n_eval_episodes episodios.

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
    
    return np.mean(all_episode_reward)

# FUNCIÓN 2
# Parámetros:
#   >num_timesteps: número de steps "teóricos" que se han dado ya durante el entrenamiento.
#   >list_eval_train_steps: lista con los números de steps en los cuales se desea evaluar el modelo.
# Devuelve: "True" o "False" para indicar si el modelo  se debe o no evaluar en este punto 
# del entrenamiento,respectivamente.

def evaluate_true_or_false(num_timesteps,list_eval_train_steps):
    #Inicialización de la variable a devolver
    yes_no=False

    #Encontrar el valor mayor más cercano y mirar si la diferencia es mayor que el salto.
    list_dif=num_timesteps-np.array(list_eval_train_steps)
    neg_dif=np.abs(list_dif[list_dif<=0])
    split=model.n_steps
    if (len(neg_dif)!=0) and (min(neg_dif)< split):
        yes_no=True    

    return yes_no

# FUNCIÓN 3
# Esta función es la versión adaptada de la función "_update_current_progress_remaining" ya existente.
# Se define con intención de poder evaluar el modelo durante el proceso de entrenamiento y poder
# recolectar información relevante (steps dados, evaluaciones hechas, calidad del modelo medida en
# reward, tiempo computacional gastado, semilla utilizada,...) asociado a ese momento del entrenamiento. 

def callback_in_each_iteration(self, num_timesteps: int, total_timesteps: int) -> None:
    # Pausar el tiempo durante la validación.
    sw.pause() 

    # Mirar si en esta altura del entrenamiento deseamos evaluar en modelo o no.
    if evaluate_true_or_false(num_timesteps,list_eval_train_steps) :

        # Para saber en que punto de la ejecución estamos
        print('seed:'+str(seed))
        print('num_timesteps:'+str(num_timesteps))

        # Extraer la información relevante.
        mean_reward = evaluate(model,eval_env,eval_seed,n_eval_episodes)
        info=pd.DataFrame(model.ep_info_buffer)
        info_steps=sum(info['r'])
        info_time=sum(info['t'])
        n_eval=len(info)
        max_step_per_eval=max(info['r'])

        #Reanudar el tiempo.
        sw.resume()

        #Guardar la información extraída.
        df_train_acc.append([num_timesteps, info_steps,model.seed,n_eval,max_step_per_eval,sw.get_time(),info_time,mean_reward])
    
    #Reanudar el tiempo.
    sw.resume()

    # Esta línea la usa la función que sustituimos: no cambiar esta línea.
    self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps) 

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Para usar la función callback modificada.
import stable_baselines3.common.base_class
stable_baselines3.common.base_class.BaseAlgorithm._update_current_progress_remaining = callback_in_each_iteration
    
# Inicialización de entornos.
train_env = gym.make('CartPole-v1')# Entrenamiento.
eval_env = gym.make('CartPole-v1')# validación.

# Parámetros entrenamiento.
max_train_steps=10000
episode_duration=10

# Parámetros validación.
eval_seed=1234
n_eval_episodes=100
list_eval_train_steps=range(500,10500,500)

# Parámetros por defecto.
default_tau = 0.02
default_max_episode_steps = 500

# Mallados.
grid_acc=[1.0,0.8,0.6,0.4,0.3,0.2,0.175,0.15,0.125,0.1]
grid_seed=range(1,11,1)

# Guardar en una base de datos por valor de accuracy la información relevante durante el entrenamiento.
for accuracy in grid_acc:

    # Actualizar parámetros del entorno de entrenamiento.
    train_env.env.tau = default_tau / accuracy
    train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
    train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps
    
    # Guardar en una base de datos la información del proceso de entrenamiento para el accuracy seleccionado.
    df_train_acc=[]

    for seed in tqdm(grid_seed):
        # Empezar a contar el tiempo.
        sw = stopwatch()
        sw.reset()

        # Entrenamiento.
        model = PPO(MlpPolicy,train_env,seed=seed, verbose=0,n_steps=train_env._max_episode_steps)
        model.set_random_seed(seed)
        model.learn(total_timesteps=max_train_steps)

    df_train_acc=pd.DataFrame(df_train_acc,columns=['steps','info_steps','seed','n_eval','max_step_per_eval','time','info_time','mean_reward'])
    df_train_acc.to_csv('results/data/df_train_acc'+str(accuracy)+'.csv')

# Guardar los demás datos que se usarán para las gráficas.
np.save('results/data/grid_acc',grid_acc)





