# Mediante este script se evalúan 100 políticas aleatorias (asociadas a una precisión máxima) 
# sobre 10 episodios de un entorno definido con 10 valores de accuracy diferentes para el parámetro
# timesteps (tiempo transcurrido entre frame y frame de un episodio). Los datos relevantes (medias 
# de rewards y número de steps por evaluación) se almacenan para después poder acceder a ellos.


#==================================================================================================
# LIBRERÍAS
#==================================================================================================
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import numpy as np
import pandas as pd
from tqdm import tqdm

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCIÓN 1 (construir muestra aleatoria de políticas)
def build_policy_sample(n_sample=100):
    # Inicializar entorno.
    env = gym.make('CartPole-v1')

    # Construir muestra de políticas.
    policy_sample=[]
    for i in range(0,n_sample):
        policy = PPO(MlpPolicy,env,seed=i)
        policy_sample.append(policy)

    return policy_sample

# FUNCIÓN 2 (evaluar política sobre 10 episodios de un entorno definido con un accuracy concreto)
def evaluate_policy(policy,accuracy):
    # Inicializar entorno de entrenamiento con el accuracy indicado.
    env = gym.make('CartPole-v1')
    env.env.tau = env.env.tau / accuracy
    env.env.spec.max_episode_steps = int(env.env.spec.max_episode_steps*accuracy)
    env._max_episode_steps = env.unwrapped.spec.max_episode_steps
    
    #Para garantizar que en cada llamada a la función se usarán los mismos episodios.
    env.seed(0)
    obs=env.reset()
    
    all_episode_reward=[]
    all_episode_steps=[]
    for i in range(10):
        episode_rewards = 0
        episode_steps=0
        done = False
        while not done:
            action, _states = policy.predict(obs, deterministic=True)         
            obs, reward, done, info = env.step(action) 
            episode_rewards+=reward 
            episode_steps+=1

        all_episode_reward.append(episode_rewards)
        all_episode_steps.append(episode_steps)
        obs = env.reset() 
    
    return np.mean(all_episode_reward),np.mean(all_episode_steps)

# FUNCIÓN 3 (evaluar conjunto de políticas sobre 10 episodios de un entorno definido con un accuracy concreto)
def evaluate_policy_sample(policy_sample,accuracy):
    for i in tqdm(range(len(policy_sample))):
        reward,steps=evaluate_policy(policy_sample[i],accuracy)
        df.append([accuracy,i,reward,steps])

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Construir muestra de políticas.
policy_sample=build_policy_sample(100)

# Lista de accuracys.
list_acc=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# Construir base de datos con información de rewards y costes obtenidos al evaluar las políticas de
# la muestra con los diferentes valores de accuracy.
df=[]
for accuracy in tqdm(list_acc):
    evaluate_policy_sample(policy_sample,accuracy)

# Guardar base de datos.
df=pd.DataFrame(df,columns=['accuracy','n_policy','reward','steps'])
df.to_csv('results/data/CartPole/UnderstandingAccuracy.csv')


