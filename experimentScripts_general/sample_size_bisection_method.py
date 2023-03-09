import pandas as pd


# calcular el porcentaje de la población que se consederará para aplicar el método de bisección.
def population_percentage_bisection_method(env_name,popsize,perc_total_cost):

	# Leer base de datos con información de coste por evaluación.
	df=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_BisectionThresold.csv',index_col=0)
	
	# Calcular coste.
	all_pop_bisection=sum([i*popsize for i in list(df['cost_per_eval'])[:-1]])+popsize*df['cost_per_eval'][-2]
	all_pop_max_acc=popsize*df['cost_per_eval'][-1]
	
	# El porcentaje de la población que habrá que considerar para que la aplicación del método de bisección 
	# suponga un coste igual al perc_total_cost % del coste asociado al accuracy 1.
	perc_pop=(perc_total_cost*all_pop_max_acc)/all_pop_bisection
	
	return perc_pop


list_envs_popsize={'SymbolicRegressor':1000,'MuJoCo':20,'WindFLO':50,'Turbines':8}
perc_total_cost=0.5 # Reducir a la mitad el coste de evaluar una población al usar el método de bisección.

df=[]
for env_name in list(list_envs_popsize.keys()):

	perc_pop=population_percentage_bisection_method(env_name,list_envs_popsize[env_name],perc_total_cost)
	df.append([env_name,list_envs_popsize[env_name],perc_pop,int(list_envs_popsize[env_name]*perc_pop)])
	
df=pd.dataFrame(df,columns=['env','pop_size','sample_perc','sample_size'])
df.to_csv('results/data/general/sample_size.csv')
	
	
	
