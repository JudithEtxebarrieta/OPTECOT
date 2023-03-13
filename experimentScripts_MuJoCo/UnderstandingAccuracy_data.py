# Mediante este script se evalúan 100 políticas aleatorias (asociadas a una precisión máxima) 
# sobre 10 episodios de un entorno definido con 10 valores de accuracy diferentes para el parámetro
# timesteps (tiempo transcurrido entre frame y frame de un episodio). Los datos relevantes (medias 
# de rewards y número de steps por evaluación) se almacenan para después poder acceder a ellos.

#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import os
import sys
from shutil import rmtree
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import numpy as np
from garage.envs import GymEnv
import pandas as pd
from garage.tf.policies import CategoricalMLPPolicy,ContinuousMLPPolicy
from garage.tf.policies.policy import Policy
from garage import wrap_experiment
from garage.trainer import TFTrainer
from tqdm import tqdm as tqdm
from garage.experiment.deterministic import set_seed
from garage.np.algos import CMAES
from garage.sampler import LocalSampler
import garage.sampler.default_worker
from garage.sampler.default_worker import *
import dowel.logger
from dowel.logger import *

import tensorflow as tf
from tensorflow.python.ops.variable_scope import VariableScope
VariableScope.reuse=tf.compat.v1.AUTO_REUSE

import multiprocessing as mp
import psutil as ps


#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCIÓN 1 (Esta función es una modificación de otra ya existente, con la modificación se pretenden
# ir almacenando los datos de interés).
def rollout(self):

    self.start_episode()
    while not self.step_episode():
        pass

    # MODIFICACIÓN: cada vez que se entre en esta función estaremos trabajando con una política nueva,
    # por tanto es el momento en donde debemos evaluar la política y guardar la información de interés.
    global n_policy,global_n_sample
    print('Policy: '+str(n_policy)+str('/')+str(global_n_sample))
    for accuracy in tqdm(list_acc):
        reward,steps=evaluate_policy(self.agent,accuracy)
        df.append([accuracy,n_policy,reward,steps])
    n_policy+=1

    return self.collect_episode()

# FUNCIÓN 2 (evaluar una muestra aleatoria de políticas, para ello se considerarán los 
# individuos de una única generación del algoritmo CMA-ES)
@wrap_experiment
def evaluate_policy_sample(ctxt=None,n_sample=100):

    global global_n_sample
    global_n_sample=n_sample

    with TFTrainer(ctxt) as trainer:

        set_seed(0)

        # Inicializar entorno sobre el cual se definirán las políticas (con máximo accuracy).
        env = GymEnv(gymEnvName, max_episode_length=max_episode_length)
        env._env.seed(seed=0)

        # En caso de usar un entorno con criterio de parada dependiente del parámetro 
        # terminate_when_unhealthy, anular este tipo de parada.
        if DTU:
            env._env.env._terminate_when_unhealthy = False 

        # Evaluar muestra aleatoria de 100 políticas con los accuracys seleccionados.
        global n_policy
        n_policy=1
        is_action_space_discrete=bool(["continuous", "discrete"].index(action_space))
        if is_action_space_discrete:
            policy = CategoricalMLPPolicy(name=policy_name, env_spec=env.spec)
        else:
            policy = ContinuousMLPPolicy(name=policy_name, env_spec=env.spec)

        sampler = LocalSampler(agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, is_tf_worker=True)
        algo= CMAES(env_spec=env.spec, policy=policy, sampler=sampler, n_samples=n_sample)
        trainer.setup(algo, env)
        trainer.train(n_epochs=1, batch_size=env.spec.max_episode_length) # Esta función llamará internamente a "rollout".

# FUNCIÓN 3 (evaluar una política)
def evaluate_policy(policy,accuracy):

    # Inicializar entorno de validación fijando el valor de accuracy adecuado.
    eval_env = GymEnv(gymEnvName, max_episode_length=max_episode_length*accuracy)
    eval_env._env.unwrapped.model.opt.timestep=default_frametime/accuracy

    # Fijar semilla para que los episodios a evaluar sean los mismos por cada llamada a la función 
    # y definir el estado inicial (obs) del primer episodio.
    eval_env._env.seed(seed=0)
    obs,_=eval_env.reset()

    # Guardar reward por episodios evaluado.
    all_ep_reward=[]
    steps=0
    for _ in range(10):
        # Evaluar episodio con la política y guardar el reward asociado.
        episode_reward=0
        done=False
        while not done:
            action,_=policy.get_action(obs)
            env_step_info=eval_env.step(action)
            episode_reward+=env_step_info.reward
            done=env_step_info.last
            obs=env_step_info.observation
            steps+=1
        all_ep_reward.append(episode_reward)
        obs,_=eval_env.reset()

    reward=np.mean(all_ep_reward)

    return reward,steps
 
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Modificación de función existente (para guardar datos durante el entrenamiento).
garage.sampler.default_worker.DefaultWorker.rollout = rollout
# Modificación de función existente ( para no perder tiempo).
garage.trainer.Trainer.save = lambda self, epoch: "skipp save."
# Modificación de función existente (para no imprimir información de episodio durante el entrenamiento).
Logger.log= lambda self, data: 'skipp info message.'
warnings.filterwarnings("ignore")

# Características del entorno (para ejecutar el script con otro entorno MuJoCo estas son las únicas 
# variables que habrá que modificar).
gymEnvName='Swimmer-v3'
action_space="continuous"
max_episode_length=1000
default_frametime=0.01 # Parámetro del que se modificará el accuracy.
policy_name='SwimmerPolicy'
DTU=False # Criterio de parada dependiente de terminate_when_unhealthy.

#--------------------------------------------------------------------------------------------------
# Para el análisis de motivación.
#--------------------------------------------------------------------------------------------------

# Lista de accuracys con los que se evaluará la muestra anterior.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# Evaluar una muestra aleatoria de 100 políticas usando diferentes valores de accuracy y construir una 
# base de datos con los scores (rewards) y tiempos de ejecución (steps) por evaluación.
df=[]
evaluate_policy_sample(n_sample=100)

# Guardar base de datos.
df=pd.DataFrame(df,columns=['accuracy','n_policy','reward','steps'])
df.to_csv('results/data/MuJoCo/UnderstandingAccuracy/df_UnderstandingAccuracy.csv')

#--------------------------------------------------------------------------------------------------
# Para la fijación del tamaño de muestra del método de bisección.
#--------------------------------------------------------------------------------------------------
# Lista con los valores de accuracy que se considerarían por el método de bisección, teniendo en
# cuenta que el criterio de parada es alcanzar un rango del intervalo de 0.1 y suponiendo que
# en todas las iteraciones se acota el intervalo por arriba (caso más costoso).
def upper_middle_point(lower,upper=1.0):
    list=[] 
    while abs(lower-upper)>0.1:       
        middle=(lower+upper)/2
        list.append(middle)
        lower=middle
    return list

list_acc=upper_middle_point(1/max_episode_length)+[1.0]

# Evaluar una muestra aleatoria usando los valores anteriores de accuracy.
df=[]
evaluate_policy_sample(n_sample=100)

df_bisection=pd.DataFrame(df,columns=['accuracy','n_policy','reward','cost_per_eval'])
df_bisection=df_bisection[['accuracy','cost_per_eval']]
df_bisection=df_bisection.groupby('accuracy').mean()
df_bisection.to_csv('results/data/MuJoCo/UnderstandingAccuracy/df_BisectionSample.csv')

# Eliminar ficheros auxiliares.
sys.path.append('data')
rmtree('data')

