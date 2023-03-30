
#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'cleanrl'))

from argparse import ArgumentError
from statistics import median
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import subprocess
import time
import re
from os.path import exists
import sys
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse

from garage import wrap_experiment
import garage.sampler.default_worker
from garage.sampler.default_worker import *
from garage._functions import *
import garage.trainer
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.algos import CMAES
from garage.sampler import LocalSampler
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy
from garage.trainer import TFTrainer,Trainer

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

import numpy as np
import pandas as pd
from dowel.logger import Logger
import warnings
import multiprocessing as mp
import psutil as ps
import scipy as sc
import random
import cma
import copy

#==================================================================================================
# NUEVAS FUNCIONES
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Funciones para el proceso de aprendizaje o búsqueda de la óptima política.
#--------------------------------------------------------------------------------------------------
# FUNCIÓN  (aprender política óptima con CMA-ES)
@wrap_experiment(snapshot_mode="none", log_dir="/tmp/",archive_launch_repo=False)
def learn(ctxt=None, gymEnvName=None, action_space=None, max_episode_length=None,
                       policy_name=None,seed=None,heuristic=None,heuristic_param=None):

    global global_heuristic, global_heuristic_param,optimal_acc,train_env
    global_heuristic=heuristic
    global_heuristic_param=heuristic_param
    optimal_acc=1
    

    # Inicialización de contadores.
    global n_steps_proc,n_steps_acc,n_episodes,n_generations
    n_steps_proc=0 # Contador que índica el número de steps consumidos hasta el momento para el procedimiento.
    n_steps_acc=0 # Contador que índica el número de steps consumidos hasta el momento para el ajuste del accuracy.
    n_episodes = 0 # Contador que indica en que episodio estamos.
    n_generations=0 # Contador que indica en que generación estamos.

    # Definición de parámetros para el algoritmos CMA-ES.
    global popsize,batch_size_ep,total_generations
    popsize= 20 # Tamaño de las generaciones/poblaciones en CMA-ES.
    batch_size_ep=1 # Número de episodios que se van a considerar para evaluar cada política/individuos de una generación.

    # Fijar semilla.
    set_seed(seed)

    with TFTrainer(ctxt) as trainer:

        # Definir entorno de entrenamiento con el accuracy seleccionado.
        current_max_episode_length=int(max_episode_length*optimal_acc)
        train_env = GymEnv(gymEnvName, max_episode_length=current_max_episode_length)
        train_env._env.unwrapped.model.opt.timestep=default_frametime/optimal_acc
        train_env._env.seed(seed=train_env_seed)

        # Definir entorno de validación con el accuracy máximo.
        global test_env
        test_env = GymEnv(gymEnvName, max_episode_length=max_episode_length)
        test_env._env.unwrapped.model.opt.timestep=default_frametime
        
        # Definir política.
        is_action_space_discrete=bool(["continuous", "discrete"].index(action_space))
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name=policy_name, env_spec=train_env.spec)
        else:
            policy = ContinuousMLPPolicy(name=policy_name, env_spec=train_env.spec)
        sampler = LocalSampler(agents=policy, envs=train_env, max_episode_length=train_env.spec.max_episode_length, is_tf_worker=True)

        # Inicializar algoritmo CMA-ES.
        algo= CMAES(env_spec=train_env.spec, policy=policy, sampler=sampler, n_samples=popsize)
        trainer.setup(algo, train_env)

        # Entrenamiento.
        total_generations= int(max_steps/(popsize*batch_size_ep*current_max_episode_length)) # Número de generaciones a evaluar, dependiendo del número máximo de steps definido para entrenar. (CRITERIO DE PARADA)
        trainer.train(n_epochs=total_generations, batch_size=batch_size_ep*current_max_episode_length)

#--------------------------------------------------------------------------------------------------
# Funciones auxiliares para definir el accuracy apropiado en cada momento del proceso.
#--------------------------------------------------------------------------------------------------
# FUNCIÓN (Cálculo de la correlación de Spearman entre dos secuencias)
def spearman_corr(x,y):
    return sc.stats.spearmanr(x,y)[0]

# FUNCIÓN (Convertir vector de rewards en vector de ranking)
def from_rewards_to_ranking(list_rewards):
    list_pos_ranking=np.argsort(np.array(list_rewards))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

# FUNCIÓN (validar una política)
def evaluate_policy_test(agent,env,n_eval_episodes):
    global n_generations

    # Fijar semilla del entorno (para que los episodios a validar sena los mismos por cada 
    # llamada a la función) y definir el estado inicial (obs) del primer episodio.
    env._env.seed(seed=test_env_seed)
    obs,_=env.reset()

    # Guardar reward por episodios evaluado.
    all_ep_reward=[]
    for _ in range(n_eval_episodes):
        # Evaluar episodio con la política y guardar el reward asociado.
        episode_reward=0
        done=False
        while not done:
            action,_=agent.get_action(obs)
            env_step_info=env.step(action)
            episode_reward+=env_step_info.reward
            done=env_step_info.last
            obs=env_step_info.observation
        all_ep_reward.append(episode_reward)
        obs,_=env.reset()

    return np.mean(all_ep_reward)

# FUNCIÓN  (Generar lista con los rewards asociados a cada política que forma la muestra de la generación para el método de bisección)
def generation_reward_list(df_sample_policy_info,accuracy,count_time_acc=True):

    global batch_size_ep,n_steps_acc
    
    # Obtener rewards asociados a las politicas que forman la muestra de la población.
    list_rewards=list(df_sample_policy_info[df_sample_policy_info['accuracy']==accuracy]['reward'])

    # Actualizar contadores relacionados con el tiempo (steps) de entrenamiento.
    if count_time_acc:
        n_steps_acc+=len(list_rewards)*batch_size_ep*int(max_episode_length*accuracy)

    return list_rewards
#--------------------------------------------------------------------------------------------------
# Funciones asociadas a los heurísticos que se aplicarán para ajustar el accuracy.
#--------------------------------------------------------------------------------------------------
# FUNCIÓN (calcular los posibles valores intermedios que se pueden evaluar en el método de bisección)
def possible_accuracy_values_bisection():
    df_bisection=pd.read_csv('results/data/MuJoCo/UnderstandingAccuracy/df_Bisection.csv')
    interpolation_acc=list(df_bisection['accuracy'])
    interpolation_time=list(df_bisection['cost_per_eval'])
    lower=min(list(interpolation_time))
    upper=max(list(interpolation_time))

    list_bisection_acc=[]
    list_bisection_time=np.arange(lower,upper+(upper-lower)/(2**4),(upper-lower)/(2**4))[1:]# Se consideran 4 iteraciones del método de bisección.
    for time in list_bisection_time:
        list_bisection_acc.append(np.interp(time,interpolation_time,interpolation_acc))
    
    return list_bisection_acc

# FUNCIÓN  (Implementación adaptada del método de bisección)
def bisection_method(lower_time,upper_time,df_sample_policy_info,interpolation_pts,threshold=0.95):

    # Inicializar límite inferior y superior.
    time0=lower_time
    time1=upper_time 

    # Punto intermedio.
    prev_m=lower_time
    m=(time0+time1)/2

    # Función para calcular la correlación entre los rankings de sample_size políticas
    # aleatoria usando el accuracy actual y el máximo.
    def similarity_between_current_best_acc(acc,df_sample_policy_info,first_iteration):

        # Guardar los rewards asociados a cada solución seleccionada.
        best_rewards=generation_reward_list(df_sample_policy_info,1,count_time_acc=first_iteration)# Con el máximo accuracy. 
        new_rewards=generation_reward_list(df_sample_policy_info,acc)# Accuracy nuevo. 

        # Obtener vectores de rankings asociados.
        new_ranking=from_rewards_to_ranking(new_rewards)# Accuracy nuevo. 
        best_ranking=from_rewards_to_ranking(best_rewards)# Máximo accuracy. 

        # Comparar ambos rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)
        # print('corr bisec: '+str(metric_value))
        return metric_value

    # Reajustar límites del intervalo hasta que este tenga un rango lo suficientemente pequeño.
    first_iteration=True
    stop_threshold=(time1-time0)*0.1# Equivalente a 4 iteraciones.
    while time1-time0>stop_threshold:
        metric_value=similarity_between_current_best_acc(np.interp(m,interpolation_pts[0],interpolation_pts[1]),df_sample_policy_info,first_iteration)
        if metric_value>=threshold:
            time1=m
        else:
            time0=m

        prev_m=m
        m=(time0+time1)/2
        
        first_iteration=False

    return np.interp(prev_m,interpolation_pts[0],interpolation_pts[1])

# FUNCIÓN (Ejecutar heurísticos durante el proceso de entrenamiento)
def execute_heuristic(gen,acc,df_sample_policy_info,list_variances,heuristic,param):
    global last_time_heuristic_accepted,unused_bisection_executions
    global n_steps_proc
    global n_steps_acc
    global max_steps
    global batch_size_ep
    global heuristic_accepted

    heuristic_accepted=False
    
    # Para la interpolación en el método de bisección..
    df_interpolation=pd.read_csv('results/data/MuJoCo/UnderstandingAccuracy/df_Bisection.csv')
    interpolation_acc=list(df_interpolation['accuracy'])
    interpolation_time=list(df_interpolation['cost_per_eval'])
    lower_time=min(interpolation_time)
    upper_time=max(interpolation_time)

    # HEURÍSTICO I de Symbolic Regressor: Bisección con frecuencia constante (el umbral es el parámetro).
    if heuristic=='I': 
        if gen==0:
            acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc],threshold=param)
            heuristic_accepted=True
        else:
            if (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq:
                acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc],threshold=param)
                heuristic_accepted=True

    # HEURÍSTICO II de Symbolic Regressor: Bisección con definición automática para frecuencia 
    # de actualización de accuracy (depende de parámetro) y umbral del método de bisección fijado en 0.95.
    if heuristic=='II': 
        if gen==0: 
            acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc])
            unused_bisection_executions=0
            heuristic_accepted=True
            
        else:
            if len(list_variances)>=param+1:
                # Función para calcular el intervalo de confianza.
                def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                    mean_list=[]
                    for i in range(bootstrap_iterations):
                        sample = np.random.choice(data, len(data), replace=True) 
                        mean_list.append(np.mean(sample))
                    return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

                variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-param):-2])
                last_variance=list_variances[-1]

                # Calcular el mínimo accuracy con el que se obtiene la máxima calidad.
                if last_variance<variance_q05 or last_variance>variance_q95:

                    if (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq:   
                        unused_bisection_executions+=int((n_steps_proc+n_steps_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                        acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc])
                        heuristic_accepted=True
                    else:
                        if unused_bisection_executions>0:
                            acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc])
                            unused_bisection_executions-=1
                            heuristic_accepted=True

            else:
                if (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq:
                    acc=bisection_method(lower_time,upper_time,df_sample_policy_info,[interpolation_time,interpolation_acc])
                    heuristic_accepted=True


    return acc

#==================================================================================================
# FUNCIONES DISEÑADAS PARA SUSTITUIR ALGUNAS YA EXISTENTES
#==================================================================================================
# FUNCIÓN (Esta función esta diseñada para sustituir la función "rollout" ya existente. Se modifica
# para poder evaluar la muestra de políticas de una generación en el método de bisección.)
def save_policy_train_bisection_info(self):
    # Nuevo código.
    global df_policy_train_bisection_info,list_idx_bisection,idx_sample,list_bisection_acc,n_sample
    if idx_sample in list_idx_bisection:
        n_sample+=1
        for accuracy in list_bisection_acc:
            self.start_episode()
            self._max_episode_length = int(max_episode_length*accuracy) # Fijar tamaño máximo del episodio.
            policy_reward=0
            episode_steps=0
            while not self.step_episode(): # Hasta que no se puedan dar más steps en el episodio.
                policy_reward+=self._env_steps[episode_steps].reward # Dar un nuevo step.
                episode_steps+= 1 # Sumar step.
            df_policy_train_bisection_info.append([n_sample,accuracy,policy_reward])

    # Código por defecto.
    self.start_episode()
    while not self.step_episode():
        pass
    return self.collect_episode()


# FUNCIÓN  (para poder almacenar los datos de interés durante el entrenamiento).
def rollout(self):
    global n_steps_proc,n_steps_acc,n_episodes,n_generations,df,test_n_eval_episodes,popsize,total_generations
    global seed,global_heuristic_param,optimal_acc
    global list_gen_policies,list_gen_policies_rewards,policy_reward_per_ep
    global sample_size,heuristic_accepted,last_time_heuristic_accepted

    # Listas para guardar los rewards de las políticas asociadas a una generación.
    n_policies=n_episodes/batch_size_ep #Políticas evaluadas hasta el momento.
    if n_policies%popsize==0:
        list_gen_policies=[]
        list_gen_policies_rewards=[]

        policy_reward_per_ep=[]

    # En caso de usar un entorno con criterio de parada dependiente del parámetro 
    # terminate_when_unhealthy, anular este tipo de parada.
    if DTU:
        self.env._env.env._terminate_when_unhealthy = False

    # Inicializar episodio.
    self.start_episode() 
    self._max_episode_length = int(max_episode_length*optimal_acc) # Fijar tamaño máximo del episodio.

    # Actualizar contadores tras evaluar el episodio.
    episode_steps = 0 # Inicializar contador para sumar el número de steps dados en este episodio.
    policy_reward=0
    while not self.step_episode(): # Hasta que no se puedan dar más steps en el episodio.
        policy_reward+=self._env_steps[episode_steps].reward # Dar un nuevo step.
        episode_steps+= 1 # Sumar step.

    n_steps_proc += episode_steps #  Steps totales dados hasta el momento.
    n_episodes += 1 # Episodios evaluados hasta el momento.
    policy_reward_per_ep.append(policy_reward)

    # Cuando se haya evaluado una política/individuo al completo (número de episodios evaluado=batch_size_ep),
    # se almacena su correspondiente reward (reward total en batch_size_ep).
    if n_episodes%batch_size_ep==0:
        list_gen_policies.append(self.agent)
        list_gen_policies_rewards.append(np.mean(policy_reward_per_ep))
        policy_reward_per_ep=[]
    
    # Cuando se han evaluado las suficientes políticas como para poder completar una generación nueva,
    # se guarda la información obtenida de la nueva generación.
    n_policies=n_episodes/batch_size_ep#Políticas evaluadas hasta el momento.
    if n_policies!=0 and n_policies%popsize == 0:

        # Restar número de steps sumados de forma duplicada (en la última iteración del método de bisección ya se han evaluado 
        # sample_size políticas que forman la generación, y los steps dados se han vuelto a contar ahora en n_steps_proc)
        if heuristic_accepted:
            n_steps_acc-=sample_size*batch_size_ep*int(max_episode_length*optimal_acc)
            last_time_heuristic_accepted=n_steps_proc+n_steps_acc

        # Actualizar base de datos.
        n_generations+=1
        best_policy=list_gen_policies[list_gen_policies_rewards.index(max(list_gen_policies_rewards))]
        reward=evaluate_policy_test(best_policy,test_env,test_n_eval_episodes)
        df.append([global_heuristic_param,seed,n_generations,reward,optimal_acc,np.var(list_gen_policies_rewards),n_steps_proc,n_steps_acc,n_steps_proc+n_steps_acc])
        
        
        # print('accuracy: '+str(optimal_acc)+'n_steps_proc: '+str(n_steps_proc)+'n_steps_acc: '+str(n_steps_acc)+'n_steps: '+str(n_steps_proc+n_steps_acc))

    return self.collect_episode()

# FUNCIÓN (para poder aplicar los heurísticos durante el proceso de entrenamiento)
def train(self, trainer):

    init_mean = self.policy.get_param_values()
    self._es = cma.CMAEvolutionStrategy(init_mean, self._sigma0,
                                        {'popsize': self._n_samples})
    self._all_params = self._sample_params()
    self._cur_params = self._all_params[0]
    self.policy.set_param_values(self._cur_params)
    self._all_returns = []

    # start actual training
    last_return = None

    global n_generations,df,global_heuristic,global_heuristic_param,optimal_acc,batch_size_ep
    
    global n_steps_proc,n_steps_acc,train_env,last_time_heuristic_accepted,unused_bisection_executions
    while n_steps_proc+n_steps_acc<max_steps:# MODIFICACIÓN: cambiar criterio de parada.
        # MODIFICACIÓN: obtener individuos de la generación.
        global df_policy_train_bisection_info
        def obtain_population():
            global seed,idx_sample,df_policy_train_bisection_info,list_idx_bisection,list_bisection_acc,n_sample

            garage.sampler.default_worker.DefaultWorker.rollout = save_policy_train_bisection_info

            df_policy_train_bisection_info=[]
            list_bisection_acc=possible_accuracy_values_bisection()
            n_sample=0

            random.seed(seed)
            list_idx_bisection=random.sample(range(self._n_samples),sample_size)


            trainer_pop=trainer
            trainer_pop.step_itr = trainer._stats.total_itr
            trainer_pop.step_episode = None
            for idx_sample in range(self._n_samples):

                trainer_pop.step_episode = trainer_pop.obtain_episodes(trainer_pop.step_itr)
                last_return = trainer_pop._algo._train_once(trainer_pop.step_itr,trainer_pop.step_episode)
                trainer_pop.step_itr += 1

        if n_generations==0:
            obtain_population()
        else:
            if global_heuristic=='I' and (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq: 
                obtain_population()
            elif global_heuristic=='II':
                df_seed=pd.DataFrame(df)
                df_seed=df_seed[df_seed[1]==seed]
                list_variances=list(df_seed[5])

                # Función para calcular el intervalo de confianza.
                def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                    mean_list=[]
                    for i in range(bootstrap_iterations):
                        sample = np.random.choice(data, len(data), replace=True) 
                        mean_list.append(np.mean(sample))
                    return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

                variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-global_heuristic_param):-2])
                last_variance=list_variances[-1]

                if (len(list_variances)>=global_heuristic_param+1) and (last_variance<variance_q05 or last_variance>variance_q95):
                    if (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq: 
                        obtain_population()
                    elif unused_bisection_executions>0:
                        obtain_population()
                elif len(list_variances)<global_heuristic_param+1 and (n_steps_proc+n_steps_acc)-last_time_heuristic_accepted>=heuristic_freq: 
                    obtain_population()

        # MODIFICACIÓN: aplicar el heurístico.
        df_policy_train_bisection_info=pd.DataFrame(df_policy_train_bisection_info,columns=['n_sample','accuracy','reward'])

        if n_generations==0:
            optimal_acc=execute_heuristic(n_generations,optimal_acc,df_policy_train_bisection_info,[],global_heuristic,global_heuristic_param)
        else:
            df_seed=pd.DataFrame(df)
            df_seed=df_seed[df_seed[1]==seed]
            optimal_acc=execute_heuristic(n_generations,optimal_acc,df_policy_train_bisection_info,list(df_seed[5]),global_heuristic,global_heuristic_param)
        
        train_env = GymEnv(gymEnvName, max_episode_length=int(max_episode_length*optimal_acc))
        train_env._env.unwrapped.model.opt.timestep=default_frametime/optimal_acc
        train_env._env.seed(seed=train_env_seed)
        trainer._env=train_env   
        trainer._train_args.batch_size=batch_size_ep*int(max_episode_length*optimal_acc)

        # MODIFICACIÓN: evaluar la generación.
        garage.sampler.default_worker.DefaultWorker.rollout = rollout
        trainer.step_itr = trainer._stats.total_itr
        trainer.step_episode = None
        for _ in range(self._n_samples):
            trainer.step_episode = trainer.obtain_episodes(trainer.step_itr)
            last_return = self._train_once(trainer.step_itr,trainer.step_episode)
            trainer.step_itr += 1

    return last_return

# FUNCIÓN (para poder evaluar cada política de cada generación durante el entrenamiento con el mismo
# conjunto de episodios, esta modificación hace que el proceso sea determinista y las comparaciones 
# de individuos por generación sean justas).
def start_episode(self):

    self._eps_length = 0

    # MODIFICACIÓN: para que siempre se usen los mismos episodios para validar las políticas/individuos y estas puedan ser comparables.
    global n_episodes,batch_size_ep,n_generations
    if n_episodes%batch_size_ep==0:
        self.env._env.seed(seed=n_generations)#seed=train_env_seed (si se quiere evaluar los mismos episodios en todas las generaciones)
    self._prev_obs, episode_info = self.env.reset()

    for k, v in episode_info.items():
        self._episode_infos[k].append(v)

    self.agent.reset()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Modificación de función existente ( para no perder tiempo).
garage.trainer.Trainer.save = lambda self, epoch: "skipp save."
# Modificación de función existente (para no imprimir información de episodio durante el entrenamiento).
Logger.log= lambda self, data: 'skipp info message.'
warnings.filterwarnings("ignore")
# Modificación de funciones existentes (para poder aplicar los heurísticos durante el entrenamiento)
CMAES.train=train
# Modificación de función existente (para evaluar todas las políticas/individuos de cada generación con los mismos episodios y así puedan ser comparables).
from garage.sampler.default_worker import DefaultWorker
DefaultWorker.start_episode=start_episode


# Características del entorno.
gymEnvName='Swimmer-v3'
action_space="continuous"
max_episode_length=1000
default_frametime=0.01 # Parámetro del que se modificará el accuracy.
policy_name='SwimmerPolicy'
DTU=False # Criterio de parada dependiente de terminate_when_unhealthy.

# Mallados y parámetros para el entrenamiento.
list_train_seeds = list(range(7,22,1)) # Lista de semillas de entrenamiento.
train_env_seed=0 # Semilla para el entorno de entrenamiento.
max_steps=1000000 # Límite de entrenamiento medido en steps (asumiendo que el coste de todos los steps es el mismo). (Referencia: https://huggingface.co/sb3/ppo-Swimmer-v3/tree/main)

# Parámetros de validación.
test_env_seed=1 # Semilla para el entorno de validación.
test_n_eval_episodes=10 # Número de episodios que se evaluarán en la validación.

# Lista de argumentos para el procesamiento en paralelo.
list_arg=[['I',0.8],['I',0.95],['II',5],['II',10]]
sample_size_freq='BisectionOnly'
df_sample_freq=pd.read_csv('results/data/general/sample_size_freq_'+str(sample_size_freq)+'.csv',index_col=0)
sample_size=int(df_sample_freq[df_sample_freq['env_name']=='MuJoCo']['sample_size'])
heuristic_freq=float(df_sample_freq[df_sample_freq['env_name']=='MuJoCo']['frequency_time'])

# Función para ejecución en paralelo.
def parallel_processing(arg):

    heuristic=arg[0]
    heuristic_param=arg[1]

    global df
    df=[]

    global seed

    for seed in tqdm(list_train_seeds):
        learn(gymEnvName=gymEnvName, action_space=action_space, max_episode_length=max_episode_length,policy_name=policy_name,seed=seed,heuristic=heuristic,heuristic_param=heuristic_param) 

    # Guardar base de datos.
    df=pd.DataFrame(df,columns=['heuristic_param','seed','n_gen','reward','accuracy','variance','n_steps_proc','n_steps_acc','n_steps'])
    df.to_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(heuristic_param)+'_2.csv')

# Procesamiento en paralelo.
phisical_cpu=ps.cpu_count(logical=False)
pool=mp.Pool(phisical_cpu)
pool.map(parallel_processing,list_arg)
pool.close()

# Unir bases de datos.
# df1=pd.read_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_hI_param0.8_1.csv', index_col=0)
# df2=pd.read_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_hI_param0.8_2.csv', index_col=0)
# df3=pd.read_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_hI_param0.95_1.csv', index_col=0)
# df4=pd.read_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_hI_param0.95_2.csv', index_col=0)
# df=pd.concat([df1,df2,df3,df4],ignore_index=True)
# df.to_csv('results/data/MuJoCo/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_hI_'+str(sample_size_freq)+'.csv')