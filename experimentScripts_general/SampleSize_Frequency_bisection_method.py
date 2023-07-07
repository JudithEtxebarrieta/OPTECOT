'''
This scrip calculates the sample size of the population to be considered when applying the bisection
method in the designed heuristics. At the same time, it is indicated how often the indications of the
heuristic should be considered (to readjust accuracy), so that the cost of applying the bisection 
method does not exceed a predefined percentage of total runtime.
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import pandas as pd
import math
import numpy as np

#==================================================================================================
# FUNCTIONS
#==================================================================================================
def bisection_middle_points(lower,upper,type_cost='max'):
	'''
	Obtain list of midpoints that are selected in 4 iterations of the bisection method, 
	in the worst case and in the mean case.
	'''
	list=[] 
	stop_threshold=(upper-lower)*0.1
	max_value=upper

	# Mean case.
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

	# Worst case: In all the iterations, the interval is bounded below.
	if type_cost=='max':
		while abs(lower-upper)>stop_threshold:       
			middle=(lower+upper)/2
			list.append(middle)
			lower=middle

	return list+[max_value]

def sample_size_and_frequency(env_name,popsize,perc_cost,bisection_cost_type='max'):
	'''
	Calculate the sample size for the bisection method and the frequency with which the indications 
	of the heuristic should be accepted.'''

	df_acc_eval_cost=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_Bisection.csv')
	time_list=bisection_middle_points(min(df_acc_eval_cost['cost_per_eval']),max(df_acc_eval_cost['cost_per_eval']),type_cost=bisection_cost_type)

	# Cost of evaluating a population with maximum accuracy.
	cost_max_acc=time_list[-1]*popsize

	# Cost of applying the bisection method on a population.
	cost_bisection=min_sample_size*(sum(time_list)-time_list[-2])

	# Real percentage.
	perc=cost_bisection/cost_max_acc

	# If applying the bisection method exceeds the predefined cost percentage, define how often 
	# the heuristic should be heeded (when to readjust the accuracy).
	if perc>perc_cost:
		sample_size=min_sample_size
		freq_gen=math.ceil(perc/perc_cost)

	# Otherwise.
	else:
		sample_size=int((perc_cost*cost_max_acc)/(sum(time_list)-time_list[-2]))
		freq_gen=1
	freq_time=freq_gen*cost_max_acc

	return sample_size,freq_gen,freq_time,sample_size*(sum(time_list)-time_list[-2]),cost_max_acc

def build_interest_info_df(list_envs_popsize,perc_cost,bisection_cost_type):
	'''Build database with the information of interest.'''

	df=[]
	for env_name in list(list_envs_popsize.keys()):
		# Population size.
		popsize=list_envs_popsize[env_name]

		# Sample size for the bisection method and frequency with which the heuristic indications will be accepted.
		sample_size,freq_gen,freq_time,cost_worst_bisec,cost_max_acc=sample_size_and_frequency(env_name,list_envs_popsize[env_name],perc_cost,bisection_cost_type=bisection_cost_type)


		# Number of generations evaluated with the maximum accuracy in the given time limit for each environment.
		if env_name=='SymbolicRegressor':
			max_time=np.load('results/data/SymbolicRegressor/ConstantAccuracyAnalysis/max_n_eval.npy')
		if env_name=='WindFLO':
			max_time=np.load('results/data/WindFLO/ConstantAccuracyAnalysis/max_time.npy')
		if env_name=='MuJoCo':
			max_time=np.load('results/data/MuJoCo/ConstantAccuracyAnalysis/max_steps.npy')
		if env_name=='Turbines':
			max_time=np.load('results/data/Turbines/ConstantAccuracyAnalysis/max_time.npy')

		n_gen=int(max_time/cost_max_acc)

		# Maximum number of updates that can be done not no excede predefined bisection cost.
		max_update=int(n_gen/freq_gen)

		# Verified extra bisection cost percentage.
		bisec_perc_cost=(max_update*cost_worst_bisec)/max_time
		print(env_name,': ',bisec_perc_cost)
		
		# Add information to the database.
		df.append([env_name,popsize,n_gen,sample_size,freq_gen,freq_time,max_update])

	df=pd.DataFrame(df,columns=['env_name','popsize','n_gen_max_acc','sample_size','frequency_gen','frequency_time','max_update'])
	df.to_csv('results/data/general/SampleSize_Frequency_bisection_method.csv')

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
# Parameters.
list_envs_popsize={'SymbolicRegressor':1000,'MuJoCo':20,'WindFLO':50,'Turbines':20}
perc_bisec_cost=0.25 # Percentage of the default cost to be considered only for the application of the bisection method.
min_sample_size=10 # Minimum sample size.

# Build table.
build_interest_info_df(list_envs_popsize,perc_bisec_cost,'max')


	
	
	
