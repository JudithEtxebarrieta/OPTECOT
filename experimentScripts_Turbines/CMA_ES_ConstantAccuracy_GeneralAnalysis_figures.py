#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import numpy as np
import matplotlib as mpl
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import os
import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt

#==================================================================================================
# FUNCIONES
#==================================================================================================

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def build_constargs_dict(N):
	# Definir parámetros constantes.
	omega = 2100# Rotational speed.
	rcas = 0.4# Casing radius.
	airfoils = ["NACA0015", "NACA0018", "NACA0021"]# Set of possible airfoils.
	polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)# Polars.
	cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]# Target dumping coefficients.
	devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]# Input sea-state standard pressure deviations.
	weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]# Input sea-state weights.
	Nmin = 1000#Max threshold rotational speeds
	Nmax = 3200#Min threshold rotational speeds

	# Construir el diccionario que necesita la función fitness
	constargs = {"N": N,
		     "omega": omega,
		     "rcas": rcas,
		     "airfoils": airfoils,
		     "polars": polars,
		     "cpobjs": cpobjs,
		     "devobjs": devobjs,
		     "weights": weights,
		     "Nmin": Nmin,
		     "Nmax": Nmax,
		     "Mode": "mono"}

	return constargs

def fitness_function(turb_params,N=50):

    # Construir diccionario de parámetros constantes.
    constargs=build_constargs_dict(N)

    # Crear turbina instantantanea.
    os.chdir('OptimizationAlgorithms_KONFLOT')
    turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
    os.chdir('../')

    # Calcular evaluación.
    scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

    return -scores[1] 

def join_df(blade_number):
    # Lista de accuracys.
    list_acc=np.load('results/data/Turbines/CMA_ES_GeneralAnalysis/list_acc.npy')

    # Primera base de datos.
    df=pd.read_csv('results/data/Turbines/CMA_ES_GeneralAnalysis/df_blade_number'+str(blade_number)+'_acc'+str(list_acc[0])+'.csv',index_col=0)
    
    # Las demás.
    for accuracy in list_acc[1:]:
        new_df=pd.read_csv('results/data/Turbines/CMA_ES_GeneralAnalysis/df_blade_number'+str(blade_number)+'_acc'+str(accuracy)+'.csv',index_col=0)
        df=pd.concat([df,new_df])    
    return df

def from_argsort_to_ranking(list):
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

def ranking_matrix(df,seed):

    # Mínimo número de evaluaciones totales hecho con N=50.
    min_n_eval_max_acc=min(df[df['N']==50].groupby('seed')['n_eval'].max())

    # Reducir df a filas de interés.
    df=df[(df['seed']==seed) & (df['n_eval']<min_n_eval_max_acc)]

    # Guardar rankings de cada N en una matriz por filas.
    matrix=[]
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)
    for N in list_N:
        df_N=df[df['N']==N]
        list_scores=df_N.sort_values('n_eval')['score']
        ranking=from_argsort_to_ranking(np.argsort(-np.array(list_scores)))# Orden descendente.
        matrix.append(ranking)

    return np.matrix(matrix),min_n_eval_max_acc

def loss_quality(df):

    # Inicializar listas.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Mínimo número de evaluaciones totales hecho con N=50.
    min_n_eval_max_acc=min(df[df['N']==50].groupby('seed')['n_eval'].max())

    # Reducir df a filas de interés.
    df=df[df['n_eval']<min_n_eval_max_acc]

    # Lista con valores de N ordenada de forma descendente.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Score con máxima precisión para cada semilla.
    best_scores=[]
    list_seeds=list(set(df['seed']))
    list_seeds.sort()
    for seed in list_seeds:
        max_score=df[(df["seed"]==seed) & (df["N"]==50)]['score'].max()
        best_scores.append(max_score)


    # Guardar datos en las listas inicializa.
    for N in list_N:

        # Seleccionar registros de interés.
        df_N=df[df['N']==N]

        # Quedarnos los registros por semilla con mayor valor de score.
        rows_seed_max_score= df_N.groupby('seed')['score'].idxmax()
        df_N_seed_max_score=df_N.loc[rows_seed_max_score]

        # Calcular perdida de score por semilla
        list_loss_score=[]
        for seed in list_seeds:
            n_eval=int(df_N_seed_max_score[df_N_seed_max_score["seed"]==seed]['n_eval'])
            seed_score=float(df[(df["seed"]==seed) & (df["N"]==50) & (df["n_eval"]==n_eval)]['score'])
            list_loss_score.append((best_scores[seed]-seed_score)/best_scores[seed])

        # Bootstrap.
        mean,q05,q95=bootstrap_median_and_confiance_interval(list_loss_score)
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95

def saved_time(df):
    # Inicializar listas.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Mínimo número de evaluaciones totales hecho con N=50.
    min_n_eval_max_acc=min(df[df['N']==50].groupby('seed')['n_eval'].max())

    # Reducir df a filas de interés.
    df=df[df['n_eval']<min_n_eval_max_acc]

    # Lista con valores de N ordenada de forma descendente.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Guardar datos de interés para cada valor de N considerado.
    for N in list_N:
        # Extraer lista con tiempos de ejecución máximos por semilla.
        list_times=df[df['N']==N].groupby('seed')['time'].max()

        # Bootstrap.
        mean,q05,q95=bootstrap_median_and_confiance_interval(list_times)
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return  all_mean,all_q05,all_q95

def extra_evaluations(df):
    # Inicializar listas.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Lista con valores de N ordenada de forma descendente.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Lista de semillas.
    list_seeds=list(set(df['seed']))
    list_seeds.sort()

    # Seleccionar filas con el número máximo de evaluaciones por semilla con N=50.
    ind_max_acc=df[df['N']==50].groupby('seed')['n_eval'].idxmax()
    df_max_acc=df[df['N']==50].loc[ind_max_acc]

    # Guardar datos de interés para cada valor de N considerado.
    for N in list_N:
        # Indices por semilla de las filas con número máximo de evaluaciones.
        ind_seed_max_n_eval=df[df['N']==N].groupby('seed')['n_eval'].idxmax()

        # Seleccionar filas de interés.
        df_seed_max_n_eval=df[df['N']==N].loc[ind_seed_max_n_eval]

        # Calcular lista con las evaluaciones extra por semilla.
        list_extra_eval=[]
        for seed in list_seeds:
            n_eval_max_acc=int(df_max_acc[df_max_acc['seed']==seed]['n_eval'])
            n_eval_other_acc=int(df_seed_max_n_eval[df_seed_max_n_eval['seed']==seed]['n_eval'])
            list_extra_eval.append(n_eval_other_acc-n_eval_max_acc)

        # Bootstrap.
        mean,q05,q95=bootstrap_median_and_confiance_interval(list_extra_eval)
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)


    return all_mean,all_q05,all_q95

def total_score(df):
    # Inicializar listas.
    all_mean=[]
    all_q05=[]
    all_q95=[]

    # Lista con valores de N ordenada de forma descendente.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Lista de semillas.
    list_seeds=list(set(df['seed']))
    list_seeds.sort()

    # Guardar datos de interés para cada valor de N considerado.
    for N in list_N:
        # Indices por semilla de las filas con máximo score.
        ind_seed_max_score=df[df['N']==N].groupby('seed')['score'].idxmax()

        # Seleccionar filas de interés y quedarnos con el número de evaluaciones por semilla.
        df_seed_max_score=df[df['N']==N].loc[ind_seed_max_score]
        list_n_eval_max_score=list(df_seed_max_score.sort_values('seed')['n_eval'])

        # Calcular lista con los scores asociados a las evaluaciones destacadas por semilla
        # usando N=50.
        list_scores=[]
        for seed in list_seeds:
            # Cargar base de datos con las evaluaciones asociadas a la semilla y al valor de N.
            df_eval_seed_N=pd.read_csv('results/data/Turbines/CMA_ES_GeneralAnalysis/df_turb_param_blade_number'+str(blade_number)+'_seed'+str(seed)+'_N'+str(N)+'.csv',index_col=0)

            # Evaluar con la máxima precisión.
            n_eval=list_n_eval_max_score[seed]
            turb_params=df_eval_seed_N.iloc[n_eval-1]
            score=fitness_function(turb_params,N=50)

            # Actualizar lista.
            list_scores.append(-score)

        # Bootstrap.
        mean,q05,q95=bootstrap_median_and_confiance_interval(list_scores)
        all_mean.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    return all_mean,all_q05,all_q95
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# GRÁFICA 1: Rankings
#--------------------------------------------------------------------------------------------------
plt.figure(figsize=[15,10])
plt.subplots_adjust(left=0.05,bottom=0.11,right=1,top=0.93,wspace=0.01,hspace=0.97)

# Leer datos y juntar todos en una base de datos.
blade_number=3
df=join_df(blade_number)

# Lista de semillas
list_seeds=list(set(df['seed']))

# Tamaños de conjunto inicial considerados.
list_N=list(set(df['N']))
list_N.sort(reverse=True)

# Por cada semilla dibujar una gráfica de rankings.
for seed in list_seeds:

    # Matriz.
    matrix,min_n_eval_max_acc=ranking_matrix(df,seed)

    # Gráfica.
    ax=plt.subplot(5,2,seed+1)
    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,min_n_eval_max_acc))
    color[:1, :]=np.array([248/256, 67/256, 24/256, 1])# Rojo (código rgb)
    color = ListedColormap(color)
    ax = sns.heatmap(matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    if seed in [8,9]:
        ax.set_xlabel('Turbine design')
    if seed in [0,2,4,6,8]:
        ax.set_ylabel('N')

    colorbar=ax.collections[0].colorbar
    if seed==5:
        colorbar.set_label('Ranking position', rotation=270,labelpad=15)
        
    colorbar.set_ticks(range(0,matrix.shape[1],50))
    colorbar.set_ticklabels(range(1,matrix.shape[1]+1,50))
    ax.set_title('Seed: '+str(seed))
    ax.set_xticks(range(0,matrix.shape[1],5))
    ax.set_xticklabels(range(1,matrix.shape[1]+1,5), rotation=0)
    ax.set_yticks(range(0,matrix.shape[0]))
    ax.set_yticklabels(list_N, rotation=0)

plt.savefig('results/figures/Turbines/CMA_ES_GeneralAnalysis_rankings.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# GRÁFICA 2: perdida de calidad.
#--------------------------------------------------------------------------------------------------

# Calcular arrays que se usarán para dibujar la gráfica.
all_mean,all_q05,all_q95=loss_quality(df)

# Dibujar gráfica.
plt.figure(figsize=[12,8])
plt.subplots_adjust(left=0.06,bottom=0.11,right=0.94,top=0.88,wspace=0.2,hspace=0.58)

ax=plt.subplot(221)
ax.fill_between(list_N,all_q05,all_q95,alpha=.5,linewidth=0)
plt.plot(list_N,all_mean,linewidth=2)
ax.set_ylabel('Score loss (%)')
ax.set_xlabel('N')
ax.set_title('Comparing loss of score quality depending on N')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 3: tiempo ahorrado.
#--------------------------------------------------------------------------------------------------

# Calcular arrays que se usarán para dibujar la gráfica.
all_mean,all_q05,all_q95=saved_time(df)

# Dibujar gráfica.
ax=plt.subplot(222)
ax.fill_between(list_N,all_q05,all_q95,alpha=.5,linewidth=0)
plt.plot(list_N,all_mean,linewidth=2)
ax.set_ylabel('Time')
ax.set_xlabel('N')
ax.set_title('Times needed to evaluate the configurations \n evaluated by N=50')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 4: evaluaciones extra.
#--------------------------------------------------------------------------------------------------
# Calcular arrays que se usarán para dibujar la gráfica.
all_mean,all_q05,all_q95=extra_evaluations(df)

# Dibujar gráfica.
ax=plt.subplot(223)
ax.fill_between(list_N,all_q05,all_q95,alpha=.5,linewidth=0)
plt.plot(list_N,all_mean,linewidth=2)
ax.set_ylabel('Number of extra evaluations')
ax.set_xlabel('N')
ax.set_title('Extra evaluations in the same time required by maximum N')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 5: scores totales en el límite de tiempo definido.
#--------------------------------------------------------------------------------------------------
# Calcular arrays que se usarán para dibujar la gráfica.
all_mean,all_q05,all_q95=total_score(df)

# Dibujar gráfica.
ax=plt.subplot(224)
ax.fill_between(list_N,all_q05,all_q95,alpha=.5,linewidth=0)
plt.plot(list_N,all_mean,linewidth=2)
ax.set_ylabel('Total score')
ax.set_xlabel('N')
ax.set_title('Score of the best designs \n calculated by the maximum N')

plt.savefig('results/figures/Turbines/CMA_ES_GeneralAnalysis_others.png')
plt.show()
plt.close()