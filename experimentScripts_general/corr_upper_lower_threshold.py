from itertools import permutations
import numpy as np
import scipy as sc
from operator import add
import pandas as pd
from tqdm import tqdm

def cero_one_bool_list(True_False_list):
    list=[0]*len(True_False_list)
    inverse_list=[1]*len(True_False_list)
    for i in list:
        if i==True:
            list.append(1)
            inverse_list.append(0)

    return list,inverse_list

def correlations_lower_upper_threshold(popsize,list_thresholds):

    # Calcular todas las permutaciones posibles.
    list_rankings=list(permutations(range(1,popsize+1)))

    # Inicializar contadores.
    upper_list=[0]*len(list_thresholds)
    lower_list=[0]*len(list_thresholds)

    for i in range(len(list_rankings)):
        for j in range(i+1,len(list_rankings)):
            ranking1=list_rankings[i]
            ranking2=list_rankings[j]
            corr=sc.stats.spearmanr(ranking1,ranking2)[0]
            upper,lower=cero_one_bool_list(corr>list_thresholds)
            upper_list=list(map(add,upper_list,upper))
            lower_list=list(map(add,lower_list,lower))

    return upper_list,lower_list
            


list_sizes=range(1,101,1)
list_thresholds=np.arange(0.8,0.96,0.01)
df=[]
for popsize in tqdm(list_sizes):
    upper_list,lower_list=correlations_lower_upper_threshold(popsize,list_thresholds)
    upper_list=[str(i) for i in upper_list]
    lower_list=[str(i)+'; ' for i in lower_list]
    df.append([popsize]+list(map(add,lower_list,upper_list)))

df=pd.DataFrame(df,columns=['popsize']+list(map(add,['threshold ']*len(list_thresholds),list_thresholds)))
df.to_csv('results/data/general/corr_upper_lower_threshold.csv')

