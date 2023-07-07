'''
This scrip constructs a table for each selected environment. It gathers the following information: 
possible optimal accuracies provided by the bisection, cost of the bisection in each case and 
percentage of time saved when evaluating a population with each possible optimal accuracy.
'''
#==================================================================================================
# LIBRARIES
#==================================================================================================
import pandas as pd
import numpy as np

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def bisection_extra_cost_and_population_evaluation_cost_saving(env_name,popsize,sample_size):

	df=[]
	list_sequences=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

	# Read database with evaluation time of each accuracy.
	df_acc_eval_cost=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_Bisection.csv')
	list_acc=list(df_acc_eval_cost['accuracy'])
	list_time=list(df_acc_eval_cost['cost_per_eval'])

	for sequence in list_sequences:
		upper=max(list_time)
		lower=min(list_time)

		# Cost of the bisection method when it bounded interval like specified in sequence.
		middle=(lower+upper)/2
		j=1 # Fist midpoint
		cost=upper
		cost+=middle
		for i in sequence:       
			if i==0:
				lower=middle
			else:
				upper=middle
			middle=(lower+upper)/2
			j+=1 # Update midpoints counter.
			if j<=len(sequence):
				cost+=middle
		cost=cost*sample_size

		# Cost of evaluate by default a population.
		default_cost=popsize*max(list_time)

		# Cost of evaluate with optimal accuracy a population.
		opt_cost=middle*popsize

		# Table content.
		opt_acc=np.interp(middle,list_time,list_acc)
		bisec_cost_perc=cost/default_cost
		pop_eval_save_perc=1-opt_cost/default_cost

		# Update database.
		df.append([opt_acc,cost,pop_eval_save_perc])

	# Save database.
	df=pd.DataFrame(df,columns=['opt_acc','bisec_cost','pop_eval_save_perc'])
	df.to_csv('results/data/general/ExtraCost_SavePopEvalCost/ExtraCost_SavePopEvalCost_'+str(env_name)+'.csv')

	return df

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
list_envs_popsize={'SymbolicRegressor':1000,'MuJoCo':20,'WindFLO':50,'Turbines':20}
bisection_secuences=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

for env_name in list_envs_popsize:
	df=pd.read_csv('results/data/general/SampleSize_Frequency_bisection_method.csv')
	sample_size=int(df[df['env_name']==env_name]['sample_size'])
	bisection_extra_cost_and_population_evaluation_cost_saving(env_name,list_envs_popsize[env_name],sample_size)