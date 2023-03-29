# Mediante este scrip se calcula el tamaño de muestra de la población que se deberá tener en cuenta
# a la hora de aplicar el método de bisección en los heurísticos diseñados para Symbolic Regressor,
# WindFLO, MuJoCo y Turbines. Al mismo tiempo, se indica con que frecuencia se deberán considerar 
# las indicaciones del heurístico, para que el coste de aplicar el método de bisección no supere
# al coste predefinido.

#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import pandas as pd
import math
import numpy as np

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCIÓN 1 (Obtener lista de puntos intermedios que se seleccionan en 4 iteraciones del método de 
# bisección, en el pero caso y en el caso medio)
def bisection_middle_points(lower,upper,type_cost='max'):
	list=[] 
	stop_threshold=(upper-lower)*0.1
	max_value=upper
	# Caso intermedio.
	if type_cost=='mean':
		first=True
		while abs(lower-upper)>stop_threshold:       
			middle=(lower+upper)/2
			list.append(middle)
			if first:
				lower=middle
				first=False
			else:
				upper=middle
	# En todas las iteraciones se acota el intervalo por abajo (caso más costoso).
	if type_cost=='max':
		while abs(lower-upper)>stop_threshold:       
			middle=(lower+upper)/2
			list.append(middle)
			lower=middle

	return list+[max_value]

# FUNCIÓN 2 (Calcular el tamaño de muestra para el método de bisección y la frecuencia con la que se deberá 
# tener en cuenta las indicaciones del heurístico que nos indica cuando debemos reajustar el accuracy)
def sample_size_and_frequency(env_name,popsize,perc_cost,bisection_cost_type='max',only_bisection=True):

	df_acc_eval_cost=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_Bisection.csv')
	time_list=bisection_middle_points(min(df_acc_eval_cost['cost_per_eval']),max(df_acc_eval_cost['cost_per_eval']),type_cost=bisection_cost_type)

	# Coste de evaluar una población con el máximo accuracy.
	cost_max_acc=time_list[-1]*popsize

	# Coste de aplicar el método de bisección sobre una población.
	if only_bisection:
		cost_bisection=min_sample_size*sum(time_list)
	else:
		cost_bisection=min_sample_size*(sum(time_list)-time_list[-2])+popsize*time_list[-2]

	# Porcentaje real.
	perc=cost_bisection/cost_max_acc

	# Si el aplicar el método de bisección sobrepasa el porcentaje de coste predefinido,
	# definir la frecuencia con la que se le deberá hacer caso al heurístico (cuando reajustar el accuracy).
	if perc>perc_cost:
		sample_size=min_sample_size
		if only_bisection:
			freq_gen=math.ceil(perc/perc_cost)
		else:
			freq_gen=math.ceil((cost_bisection-popsize*time_list[-2])/(perc_cost*cost_max_acc-popsize*time_list[-2]) )

	# En otro caso.
	else:
		if only_bisection:
			sample_size=int((perc_cost*cost_max_acc)/sum(time_list))
		else:
			sample_size=int((perc_cost*cost_max_acc-popsize*time_list[-2])/(sum(time_list)-time_list[-2]))
		freq_gen=1
	freq_time=freq_gen*cost_max_acc

	return sample_size,freq_gen,freq_time

# FUNCIÓN 3 (Construir base de datos con la información de interés)
def build_interest_info_df(list_envs_popsize,perc_cost,bisection_cost_type,only_bisection,csv_name):
	df=[]
	for env_name in list(list_envs_popsize.keys()):
		# Tamaño de población.
		popsize=list_envs_popsize[env_name]

		# Número de generaciones evaluadas con el máximo accuracy en el límite de tiempo prefijado para cada entorno.
		if env_name=='SymbolicRegressor':
			df_max_acc=pd.read_csv('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc1.0.csv',index_col=0)
			n_gen=int(np.mean(list(df_max_acc.groupby('train_seed')['n_gen'].max())))
		if env_name=='MuJoCo':
			df_max_acc=pd.read_csv('results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc1.0.csv',index_col=0)
			n_gen=int(np.mean(list(df_max_acc.groupby('train_seed')['n_gen'].max())))
		if env_name=='WindFLO':
			df_max_acc=pd.read_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv',index_col=0)
			n_gen=int(np.mean(list(df_max_acc.groupby('seed')['n_gen'].max())))

		### FALTA POR INTRODUCIR EL ENTORNO "Turbines"

		# Tamaño de muestra para el método de bisección y frecuencia con la que se ejecutarán las ordenes del heurístico.
		sample_size,freq_gen,freq_time=sample_size_and_frequency(env_name,list_envs_popsize[env_name],perc_cost,bisection_cost_type=bisection_cost_type,only_bisection=only_bisection)

		# Añadir información a la base de datos.
		df.append([env_name,popsize,n_gen,sample_size,freq_gen,freq_time,str(int(n_gen/freq_gen))])

	df=pd.DataFrame(df,columns=['env_name','popsize','n_gen_max_acc','sample_size','frequency_gen','frequency_time','max_update_acc'])
	df.to_csv('results/data/general/sample_size_freq_'+str(csv_name)+'.csv')

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Parámetros.
list_envs_popsize={'SymbolicRegressor':1000,'MuJoCo':20,'WindFLO':50}#,'Turbines':20}
perc_total_cost=0.95 # Porcentaje del coste por defecto que se desea considerar para el coste de evaluar una población al usar el método de bisección.
perc_bisec_cost=0.05 # Porcentaje del coste por defecto que se desea considerar únicamente para la aplicación del método de bisección.
min_sample_size=10 # Tamaño mínimo

# Construir tabla para el caso en que el coste de aplicar la bisección y evaluar después la población con
# el accuracy seleccionado sea un 95% del coste que supone evaluar una población entera por defecto.
build_interest_info_df(list_envs_popsize,perc_total_cost,'mean',False,'BisectionAndPopulation')

# Construir tabla para en el caso en que el coste de aplicar la bisección sea como mucho un 5% más 
# que el coste de evaluar una población por defecto.
build_interest_info_df(list_envs_popsize,perc_total_cost,'max',True,'BisectionOnly')


	
	
	
