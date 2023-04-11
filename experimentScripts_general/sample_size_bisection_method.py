# Mediante este scrip se calcula el tamaño de muestra de la poblacion que se debera tener en cuenta
# a la hora de aplicar el metodo de biseccion en los heuristicos diseñados para Symbolic Regressor,
# WindFLO, MuJoCo y Turbines. Al mismo tiempo, se indica con que frecuencia se deberan considerar 
# las indicaciones del heuristico, para que el coste de aplicar el metodo de biseccion no supere
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
# FUNCION 1 (Obtener lista de puntos intermedios que se seleccionan en 4 iteraciones del metodo de 
# biseccion, en el pero caso y en el caso medio)
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
	# En todas las iteraciones se acota el intervalo por abajo (caso mas costoso).
	if type_cost=='max':
		while abs(lower-upper)>stop_threshold:       
			middle=(lower+upper)/2
			list.append(middle)
			lower=middle

	return list+[max_value]

# FUNCION 2 (Calcular el tamaño de muestra para el metodo de biseccion y la frecuencia con la que se debera 
# tener en cuenta las indicaciones del heuristico que nos indica cuando debemos reajustar el accuracy)
def sample_size_and_frequency(env_name,popsize,perc_cost,bisection_cost_type='max',only_bisection=True):

	df_acc_eval_cost=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_Bisection.csv')
	time_list=bisection_middle_points(min(df_acc_eval_cost['cost_per_eval']),max(df_acc_eval_cost['cost_per_eval']),type_cost=bisection_cost_type)

	# Coste de evaluar una poblacion con el maximo accuracy.
	cost_max_acc=time_list[-1]*popsize

	# Coste de aplicar el metodo de biseccion sobre una poblacion.
	if only_bisection:
		cost_bisection=min_sample_size*sum(time_list)
	else:
		cost_bisection=min_sample_size*(sum(time_list)-time_list[-2])+popsize*time_list[-2]

	# Porcentaje real.
	perc=cost_bisection/cost_max_acc

	# Si el aplicar el metodo de biseccion sobrepasa el porcentaje de coste predefinido,
	# definir la frecuencia con la que se le debera hacer caso al heuristico (cuando reajustar el accuracy).
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

# FUNCION 3 (Construir base de datos con la informacion de interes)
def build_interest_info_df(list_envs_popsize,perc_cost,bisection_cost_type,only_bisection,csv_name):
	df=[]
	for env_name in list(list_envs_popsize.keys()):
		# Tamaño de poblacion.
		popsize=list_envs_popsize[env_name]

		# Numero de generaciones evaluadas con el maximo accuracy en el limite de tiempo prefijado para cada entorno.
		if env_name=='SymbolicRegressor':
			df_max_acc=pd.read_csv('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/df_train_acc1.0.csv',index_col=0)
			n_gen=int(np.mean(list(df_max_acc.groupby('train_seed')['n_gen'].max())))
		if env_name=='MuJoCo':
			df_max_acc=pd.read_csv('results/data/MuJoCo/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc1.0.csv',index_col=0)
			n_gen=int(np.mean(list(df_max_acc.groupby('train_seed')['n_gen'].max())))
		if env_name=='WindFLO':
			df_max_acc=pd.read_csv('results/data/WindFLO/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv',index_col=0)
			n_gen=int(np.mean(list(df_max_acc.groupby('seed')['n_gen'].max())))
		if env_name=='Turbines':
			df_max_acc=pd.read_csv('results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis1.0.csv',index_col=0)
			n_gen=int(np.mean(list(df_max_acc.groupby('seed')['n_gen'].max())))


		# Tamaño de muestra para el metodo de biseccion y frecuencia con la que se ejecutaran las ordenes del heuristico.
		sample_size,freq_gen,freq_time=sample_size_and_frequency(env_name,list_envs_popsize[env_name],perc_cost,bisection_cost_type=bisection_cost_type,only_bisection=only_bisection)

		# Añadir informacion a la base de datos.
		df.append([env_name,popsize,n_gen,sample_size,freq_gen,freq_time,str(int(n_gen/freq_gen))])

	df=pd.DataFrame(df,columns=['env_name','popsize','n_gen_max_acc','sample_size','frequency_gen','frequency_time','max_update_acc'])
	df.to_csv('results/data/general/sample_size_freq_'+str(csv_name)+'.csv')

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Parametros.
list_envs_popsize={'SymbolicRegressor':1000,'MuJoCo':20,'WindFLO':50}#,'Turbines':20}
perc_total_cost=0.95 # Porcentaje del coste por defecto que se desea considerar para el coste de evaluar una poblacion al usar el metodo de biseccion.
perc_bisec_cost=0.05 # Porcentaje del coste por defecto que se desea considerar unicamente para la aplicacion del metodo de biseccion.
min_sample_size=10 # Tamaño minimo

# Construir tabla para el caso en que el coste de aplicar la biseccion y evaluar despues la poblacion con
# el accuracy seleccionado sea un 95% del coste que supone evaluar una poblacion entera por defecto.
build_interest_info_df(list_envs_popsize,perc_total_cost,'mean',False,'BisectionAndPopulation')

# Construir tabla para en el caso en que el coste de aplicar la biseccion sea como mucho un 5% mas 
# que el coste de evaluar una poblacion por defecto.
build_interest_info_df(list_envs_popsize,perc_total_cost,'max',True,'BisectionOnly')


	
	
	
