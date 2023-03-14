# Mediante este script se guarda la información relevante extraída del proceso de entrenamiento 
# de diferentes políticas sobre el entorno CartPole. Cada políticas se entrena con un 
# time-step diferente (en total se consideran 10 valores diferentes de accuracy) considerando un 
# total de 30 semillas (30 formas diferentes de definir los episodios sobre los cuales se entrena la política).
# Las validaciones se hacen sobre un entorno independiente al de entrenamiento (100 episodios) con
# una máxima precisión del time-step.

# Basado en: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb

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
import multiprocessing as mp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback

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
# NUEVAS FUNCIONES
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


#==================================================================================================
# FUNCIONES EXISTENTES MODIFICADAS
#==================================================================================================
# FUNCIÓN 3
# Esta función es la versión adaptada de la función "_update_current_progress_remaining" ya existente.
# Se define con intención de poder evaluar el modelo durante el proceso de entrenamiento y poder
# recolectar información relevante (steps dados, evaluaciones hechas, calidad del modelo medida en
# reward, tiempo computacional gastado, semilla utilizada,...) asociado a ese momento del entrenamiento. 

def callback_in_each_iteration(self, num_timesteps: int, total_timesteps: int) -> None:
    # Pausar el tiempo durante la validación.
    sw.pause() 

    # Mirar si en esta altura del entrenamiento deseamos evaluar en modelo o no.
    # if evaluate_true_or_false(num_timesteps,list_eval_train_steps) :

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

# FUNCIÓN 4
# Esta función es la versión adaptada de la función "_setup_learn" ya existente.
# Se modifica para que el límite al entrenamiento sea el número de steps de entrenamiento definido 
# y no el número máximo de evaluaciones que viene fijado por defecto (número máximo de episodios,maxlen).
def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=max_train_steps)#MODIFICACIÓN(antes:maxlen=100)
            self.ep_success_buffer = deque(maxlen=max_train_steps)#MODIFICACIÓN (antes:maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback


#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Para usar la función callback modificada.
import stable_baselines3.common.base_class
stable_baselines3.common.base_class.BaseAlgorithm._update_current_progress_remaining = callback_in_each_iteration
# Para usar la función _setup_learn modificada.
from stable_baselines3.common.base_class import *
BaseAlgorithm._setup_learn=_setup_learn
    
# Variables y parámetros de entrenamiento.
train_env = gym.make('CartPole-v1')
max_train_steps=10000

# Variables y parámetros de validación.
eval_env = gym.make('CartPole-v1')
eval_seed=0
n_eval_episodes=100

# Parámetros por defecto.
default_tau = 0.02
default_max_episode_steps = 500

# Mallados.
grid_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
grid_seed=range(1,31,1)

# Función para ejecución en paralelo.
def parallel_processing(accuracy):
    # Actualizar parámetros del entorno de entrenamiento.
    train_env.env.tau = default_tau / accuracy
    train_env.env.spec.max_episode_steps = int(default_max_episode_steps*accuracy)
    train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps
    
    # Guardar en una base de datos la información del proceso de entrenamiento para el accuracy seleccionado.
    global df_train_acc,seed
    df_train_acc=[]

    for seed in tqdm(grid_seed):
        # Empezar a contar el tiempo.
        global sw
        sw = stopwatch()
        sw.reset()

        # Entrenamiento.
        global model
        model = PPO(MlpPolicy,train_env,seed=seed, verbose=0,n_steps=train_env._max_episode_steps)
        model.set_random_seed(seed)
        model.learn(total_timesteps=max_train_steps)

    df_train_acc=pd.DataFrame(df_train_acc,columns=['steps','info_steps','seed','n_eval','max_step_per_eval','time','info_time','mean_reward'])
    df_train_acc.to_csv('results/data/CartPole/ConstantAccuracyAnalysis/df_train_acc'+str(accuracy)+'_.csv')

# Procesamiento en paralelo.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,grid_acc)
pool.close()

# Guardar los demás datos que se usarán para las gráficas.
np.save('results/data/CartPole/ConstantAccuracyAnalysis/grid_acc',grid_acc)

# Guardar número máximo de steps de entrenamiento.
np.save('results/data/CartPole/ConstantAccuracyAnalysis/max_train_steps',max_train_steps)




