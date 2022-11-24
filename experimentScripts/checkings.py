# En este script se comprueban y analizan diferentes propiedades del proceso de entrenamiento 
# y validación de los modelos construidos en "constant_tau_analysis.py". Esas propiedades son:

# 1) Si todos los modelos se validan con los mismos episodios
# 2) Si todos los modelos se entrenan con los mismos episodios
# 3) Si al incrementar el número de iteraciones en el bootstrap el rango del intervalo de confianza aumenta.
# 4) Se analiza el motivo de los descensos en las curvas de evaluación.

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
    kde=KernelDensity(kernel='gaussian',bandwidth=0.01).fit(original_sample)

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

    return median_reward,quantile05_reward,quantile95_reward,eval_ep

def evaluate(model,eval_env,eval_seed,n_eval_episodes):
    #Para guardar el reward por episodio.
    all_episode_reward=[]

    #Para guardar los estados iniciales de cada episodio.
    all_episode_state=[]

    #Para garantizar que en cada llamada a la función se usarán los mismos episodios.
    eval_env.seed(eval_seed)
    obs=eval_env.reset()
    
    for i in range(n_eval_episodes):

        #Ir guardando los estados iniciales de los episodios empleados.
        all_episode_state.append(obs)

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

    
    return median_reward,quantile05_reward,quantile95_reward, all_episode_state

def train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type):
    
    #Guardar el total de episodios que se usarán para entrenar.
    model = PPO(MlpPolicy,train_env,seed=train_seed,n_steps=train_env._max_episode_steps)
    model.set_random_seed(train_seed)
    model.learn(total_timesteps=list_train_steps[-1])
    info=model.ep_info_buffer
    all_train_states=[]
    for dict in info:
        all_train_states.append(list(dict['new_obs'][0])) 
    
    #Obtener episodios de validación.
    if customized==True:
        #Si se desea evaluar los modelos sobre una muestra de episodios significativa pero diferente a los episodios usados para entrenar.
        eval_ep=significant_state_sample(original_sample=all_train_states,n_sample=n_eval_episodes)
        #Si se desea evaluar los modelos sobre una muestra de episodios de los episodios usados para entrenar.
        train_ep=random.sample(all_train_states,n_eval_episodes)

    #Para guardar los resultados de validación.
    all_median_reward=[]
    all_quantile05_reward=[]
    all_quantile95_reward=[]

    #Para guardar los estados iniciales de cada episodio en el entrenamiento y la validación.
    all_test_states=[]
    
    model = PPO(MlpPolicy,train_env,seed=train_seed,n_steps=train_env._max_episode_steps)
    previous_steps=0
    for train_steps in list_train_steps:

        #Entrenar el modelo.
        model.set_random_seed(train_seed)
        model.learn(total_timesteps=train_steps-previous_steps,reset_num_timesteps=False)

        #Evaluar el modelo actualizado.
        if customized==False:
            median_reward,quantile05_reward,quantile95_reward, test_states=evaluate(model,eval_env,eval_seed,n_eval_episodes)
        if customized==True and eval_type=='test':
            median_reward,quantile05_reward,quantile95_reward, test_states=evaluate_customized_episodes(model,eval_env,eval_seed,eval_ep)
        if customized==True and eval_type=='train':
            median_reward,quantile05_reward,quantile95_reward, test_states=evaluate_customized_episodes(model,eval_env,eval_seed,train_ep)

        #Actualizaciones.
        all_median_reward.append(median_reward)
        all_quantile05_reward.append(quantile05_reward)
        all_quantile95_reward.append(quantile95_reward)
        all_test_states.append(test_states)

        previous_steps=train_steps
   
    
    return all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
#Entorno de entrenamiento y de evaluación
train_env=gym.make('CartPole-v1')
eval_env=gym.make('CartPole-v1')

#Variables relacionadas con el entorno de entrenamiento
train_seed=12345
episode_duration=10 # Duración de un episodio en segundos

#Variables relacionadas con el entorno de evaluación
eval_seed=54321
n_eval_episodes=100

#Parámetros por defecto
default_tau=0.02
default_max_episode_step=500

#--------------------------------------------------------------------------------------------------
# ¿Todos los modelos se validan con los mismos episodios?
# Si, las dos matrices con las que está formada la lista de salida son idénticas.
#--------------------------------------------------------------------------------------------------
customized=False
eval_type='test'
accuracy=1
list_train_steps=range(100,300,100)# Para construir solo dos modelos

train_env.env.tau=default_tau/accuracy
train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps

all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type)

print('Matriz con estados iniciales de los episodios de validación:')
print(np.array(all_test_states))


#--------------------------------------------------------------------------------------------------
# ¿Todos los modelos se entrenan con los mismos episodios?
# Si, al entrenar dos modelos diferentes sobre el mismo entorno y durante el mismo tiempo pero
# con diferentes time-steps, los episodios empleados para el entrenamiento coinciden. Se observa que 
# el número de episodios para el modelo menos preciso es mayor, pero los primeros coinciden con los
# del modelo más preciso.
#--------------------------------------------------------------------------------------------------
customized=False
eval_type='test'
accuracy=1
list_train_steps=range(100,200,100)# Para construir solo un modelo por valor de accuracy

train_env.env.tau=default_tau/accuracy
train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps

all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type)
print('Matriz con estados iniciales de los episodios de entrenamiento para accuracy 1:')
print(np.array(all_train_states))

accuracy=0.8
train_env.env.tau=default_tau/accuracy
train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps

all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type)
print('Matriz con estados iniciales de los episodios de entrenamiento para accuracy 0.8:')
print(np.array(all_train_states))

#--------------------------------------------------------------------------------------------------
# ¿Al aumentar el número de iteraciones en el bootstrap el rango del intervalo de confianza aumenta? 
# Si, se ejecuta dos veces el código de abajo modificando el número de iteraciones manualmente
# en la función bootstrap_median_and_confiance_interval, y se dibuja la gráfica para 10 y 1000 
# iteraciones. Se observa que el rango del intervalo de confianza aumenta de la gráfica asociada a 10
# a la gráfica asociada a 1000.
#--------------------------------------------------------------------------------------------------
customized=False
eval_type='test'
accuracy=1
list_train_steps=range(2000,10000,1000)

train_env.env.tau=default_tau/accuracy
train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps

all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type)

#Gráfica
ax=plt.subplot(111)

ax.fill_between(list_train_steps,all_quantile05_reward,all_quantile95_reward, alpha=.5, linewidth=0)
plt.plot(list_train_steps, all_median_reward, linewidth=2,label=str(accuracy))

ax.set_xlabel("Train steps")
ax.set_ylabel("Eval. Median reward ("+str(n_eval_episodes)+" episodes)")
ax.set_title('Model evaluation (bootstrap 1000 it.)')
plt.savefig('results/figures/checkings_bootstrap1000it.png')

#--------------------------------------------------------------------------------------------------
# ¿Porque en la gráfica de validación las curvas en ciertos tramos descienden?
#--------------------------------------------------------------------------------------------------
#Para la grafíca 1
customized=False
eval_type='test'
accuracy=1
list_train_steps=[]
prev_steps=0
list_train_steps=range(2000,10000,1000)

train_env.env.tau=default_tau/accuracy
train_env.env.spec.max_episode_steps = int(episode_duration/train_env.env.tau)
train_env._max_episode_steps = train_env.unwrapped.spec.max_episode_steps

eval_env.env.spec.max_episode_steps = default_max_episode_step
eval_env._max_episode_steps = eval_env.unwrapped.spec.max_episode_steps

all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type)

#Gráfica1
plt.figure(figsize=[10,5])
plt.subplots_adjust(left=0.1,bottom=0.11,right=0.97,top=0.88,wspace=0.34,hspace=0.2)
ax1=plt.subplot(131)

ax1.fill_between(list_train_steps,all_quantile05_reward,all_quantile95_reward, alpha=.5, linewidth=0)
plt.plot(list_train_steps, all_median_reward, linewidth=2,label=str(accuracy))

ax1.set_xlabel("Train steps")
ax1.set_ylabel("Median reward ("+str(n_eval_episodes)+" episodes)")
ax1.set_title('Model evaluation \n whit new episodes')

#Para la gráfica 2
customized=True
eval_type='test'

all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type)

#Gráfica2
ax2=plt.subplot(132)

ax2.fill_between(list_train_steps,all_quantile05_reward,all_quantile95_reward, alpha=.5, linewidth=0)
plt.plot(list_train_steps, all_median_reward, linewidth=2,label=str(accuracy))

ax2.set_xlabel("Train steps")
ax2.set_ylabel("Median reward ("+str(n_eval_episodes)+"episodes)")
ax2.set_title('Model evaluation \n whit customized sample of train episodes')

#Para la gráfica 3
customized=True
eval_type='train'

all_train_states,all_test_states,all_median_reward,all_quantile05_reward,all_quantile95_reward=train_and_evaluate(train_env,eval_env,train_seed,eval_seed,list_train_steps,n_eval_episodes,customized,eval_type)

#Gráfica3
ax3=plt.subplot(133)

ax3.fill_between(list_train_steps,all_quantile05_reward,all_quantile95_reward, alpha=.5, linewidth=0)
plt.plot(list_train_steps, all_median_reward, linewidth=2,label=str(accuracy))

ax3.set_xlabel("Train steps")
ax3.set_ylabel("Median reward ("+str(n_eval_episodes)+" episodes)")
ax3.set_title('Model evaluation \n whit sample of train episodes')

plt.savefig('results/figures/checkings_falls.png')





















