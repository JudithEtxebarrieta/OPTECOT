# Mediante este scrip se calcula el tamaño de muestra de la población que se deberá tener en cuenta
# a la hora de aplicar el método de bisección en los heurísticos diseñados para Symbolic Regressor,
# WindFLO, MuJoCo y Turbines. Al mismo tiempo, se indica con que frecuencia se deberán considerar 
# las indicaciones del heurístico, para que el coste de aplicar el método de bisección no supere
# al coste predefinido.

#==================================================================================================
# LIBRERIAS
#==================================================================================================
import pandas as pd
import math
import numpy as np

#==================================================================================================
# FUNCIONES
#==================================================================================================
# Calcular el tamaño de muestra para el método de bisección y la frecuencia con la que se deberá 
# tener en cuenta las indicaciones del heurístico que nos indica cuando debemos reajustar el accuracy.
def sample_size_and_frequency(env_name,popsize):

	# Leer base de datos con la información de coste por evaluación con accuracys considerados
	# en el peor caso de una aplicación del método de bisección.
	df=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_BisectionSample.csv',index_col=0)

	# Coste de evaluar una población con el máximo accuracy.
	cost_max_acc=list(df['cost_per_eval'])[-1]*popsize

	# Coste de aplicar el método de bisección sobre una población.
	cost_bisection=min_sample_size*(sum(list(df['cost_per_eval']))-list(df['cost_per_eval'])[-2])+popsize*list(df['cost_per_eval'])[-2]

	# Porcentaje real.
	perc=cost_bisection/cost_max_acc

	# Si el aplicar el método de bisección sobrepasa el porcentaje de coste predefinido,
	# definir la frecuencia con la que se le deberá hacer caso al heurístico (cuando reajustar el accuracy).
	if perc>perc_total_cost:
		sample_size=min_sample_size
		freq=math.ceil(perc/perc_total_cost)
	# En otro caso.
	else:
		sample_size=int((perc_total_cost*cost_max_acc-popsize*list(df['cost_per_eval'])[-2])/(sum(list(df['cost_per_eval']))-list(df['cost_per_eval'])[-2]))
		freq=1

	return sample_size,freq

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Parámetros.
list_envs_popsize={'SymbolicRegressor':1000,'MuJoCo':20,'WindFLO':50}#,'Turbines':8}
perc_total_cost=0.8 # Porcentaje del coste por defecto que se desea considerar para el coste de evaluar una población al usar el método de bisección.
min_sample_size=10 # Tamaño mínimo

# Construir base de datos con la información de interés.
df=[]
for env_name in list(list_envs_popsize.keys()):
	# Tamaño de población.
	popsize=list_envs_popsize[env_name]

	# Número de generaciones evaluadas con el máximo accuracy en el límite de tiempo prefijado para cada entorno.
	if env_name=='SymbolicRegressor':
		df_max_acc=pd.read_csv('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc0.1.csv',index_col=0)
		n_gen=int(np.mean(list(df_max_acc.groupby('train_seed')['n_gen'].max())))
	if env_name=='MuJoCo':
		df_max_acc=pd.read_csv('results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc0.1.csv',index_col=0)
		n_gen=int(np.mean(list(df_max_acc.groupby('train_seed')['n_gen'].max())))
	if env_name=='WindFLO':
		df_max_acc=pd.read_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis0.1.csv',index_col=0)
		n_gen=int(np.mean(list(df_max_acc.groupby('seed')['n_gen'].max())))

	### FALTA POR INTRODUCIR EL ENTORNO "Turbines"

	# Tamaño de muestra para el método de bisección y frecuencia con la que se ejecutarán las ordenes del heurístico.
	sample_size,freq=sample_size_and_frequency(env_name,list_envs_popsize[env_name])

	# Añadir información a la base de datos.
	df.append([env_name,popsize,n_gen,sample_size,freq,int(popsize/freq)])

df=pd.DataFrame(df,columns=['env_name','popsize','n_gen_max_acc','sample_size','frequency','max_update_acc'])
df.to_csv('results/data/general/bisection_sample_size_heuristic_freq.csv')


	
	
	
