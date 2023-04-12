#==================================================================================================
# LIBRERIAS
#==================================================================================================
import numpy as np
import cma
import time
import os
from tqdm import tqdm
import pandas as pd
import itertools
from joblib import Parallel, delayed
import shutil

import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt

#==================================================================================================
# FUNCIONES
#==================================================================================================


def transform_turb_params(scaled_x):

    # Definir rangos de los parametros que definen el diseno de la turbina.
    blade_number = [3,5,7]# Blade-number gene.
    sigma_hub = [0.4, 0.7]# Hub solidity gene.
    sigma_tip = [0.4, 0.7]# Tip solidity gene.
    nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
    tip_clearance=[0,3]# Tip-clearance gene.	  
    airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

    # Array con los rangos.
    bounds=np.array([
    [sigma_hub[0]    , sigma_hub[1]],
    [sigma_tip[0]    , sigma_tip[1]],
    [nu[0]           , nu[1]],
    [tip_clearance[0], tip_clearance[1]],
    [0               , 26]
    ])

    # Transformar los valores escalados de los parametros a los valores reales.
    def blade_number_transform(posible_blade_numbers,scaled_blade_number):
        discretization=np.arange(0,1+1/len(posible_blade_numbers),1/len(posible_blade_numbers))
        detection_list=discretization>scaled_blade_number
        return posible_blade_numbers[list(detection_list).index(True)-1]

    real_x = scaled_x[1:] * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    real_bladenumber= blade_number_transform(blade_number,scaled_x[0])

    return [real_bladenumber]+list(real_x[:-1])+[round(real_x[-1])]


def fitness_function(turb_params,N=100):

    # Construir diccionario de parametros constantes.
    def build_constargs_dict(N):
        # Definir parametros constantes.
        omega = 2100# Rotational speed.
        rcas = 0.4# Casing radius.
        airfoils = ["NACA0015", "NACA0018", "NACA0021"]# Set of possible airfoils.
        polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)# Polars.
        cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]# Target dumping coefficients.
        devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]# Input sea-state standard pressure deviations.
        weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]# Input sea-state weights.
        Nmin = 1000#Max threshold rotational speeds
        Nmax = 3200#Min threshold rotational speeds

        # Construir el diccionario que necesita la funcion fitness
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

    constargs=build_constargs_dict(N)

    # Crear turbina instantantanea.
    os.chdir('OptimizationAlgorithms_KONFLOT')
    turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
    os.chdir('../')

    # Calcular evaluacion.
    scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

    return -scores[1]


def learn(accuracy,seed,popsize=20):

    # Inicializar CMA-ES.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(6), 0.33,inopts={'bounds': [0, 1],'seed':seed,'popsize':popsize})

    # Inicializar contadores de tiempo.
    eval_time = 0

    # Evaluar los disenos de las generaciones hasta agotar el tiempo maximo definido por el accuracy maximo.
    n_gen=0
    while eval_time<max_time:

        # Nueva generacion.
        solutions = es.ask()

        # Transformar los valores escalados de los parametros a los valores reales.
        list_turb_params=[transform_turb_params(x) for x in solutions]

        # Obtener scores y tiempos por evaluacion.
        os.makedirs(sys.path[0]+'/PopulationScores'+str(task))

        def parallel_f(turb_params,index):
            score=fitness_function(turb_params, N=int(default_N*accuracy))
            np.save(sys.path[0]+'/PopulationScores'+str(task)+'/'+str(index),score)

        t=time.time()
        Parallel(n_jobs=4)(delayed(parallel_f)(list_turb_params[i],i) for i in range(popsize))
        eval_time+=time.time()-t

        def obtain_score_list(popsize):
            list_scores=[]
            for i in range(popsize):
                score=float(np.load(sys.path[0]+'/PopulationScores'+str(task)+'/'+str(i)+'.npy'))
                list_scores.append(score) 
            shutil.rmtree(sys.path[0]+'/PopulationScores'+str(task),ignore_errors=True)
            return list_scores
        list_scores=obtain_score_list(popsize)

        # Para construir la siguiente generacion.
        es.tell(solutions, list_scores)

        # Acumular datos de interes.
        test_score= fitness_function(transform_turb_params(es.result.xbest))
        df.append([accuracy,seed,n_gen,-test_score,eval_time])

        n_gen+=1
        # print('eval_time: '+str(eval_time)+'/'+str(max_time))

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Mallados.
list_acc=[round(i,3) for i in np.arange(0.06,1+(1-0.06)/9,(1-0.06)/9)]# Lista de accuracys a considerar.                    
list_seeds=range(2,102,1)# Lista con semillas de entrenamiento.

# Combinación de valores (accuracy,seed) que definen cada task de la ejecución en el cluster.
list_tasks=list(itertools.product(list_acc,list_seeds))

# Parametros.
default_N=100
max_time=30*60 # 50h 100 semillas y 1 accuracy.

#--------------------------------------------------------------------------------------------------
# Para ejecutar en el Cluster (comentar lineas 179-188)
#--------------------------------------------------------------------------------------------------
# # Construir base de datos con datos relevantes de la ejecución del task.
# task=int(sys.argv[1])
# accuracy,seed=list_tasks[int(sys.argv[1])]
# df=[]
# learn(accuracy,seed)

# # Guardar base de datos.
# df=pd.DataFrame(df,columns=['accuracy','seed','n_gen','score','elapsed_time'])
# df.to_csv('results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc'+str(accuracy)+'_seed'+str(seed)+'.csv')

#--------------------------------------------------------------------------------------------------
# Para ejecutar en el PC (comentar lineas 165-173)
#--------------------------------------------------------------------------------------------------
# Construir base de datos con datos relevantes de la ejecución del task.
for accuracy in list_acc:

    for seed in tqdm(list_seeds):
        task=seed
        df=[]
        learn(accuracy,seed)

    # Guardar base de datos.
    df=pd.DataFrame(df,columns=['accuracy','seed','n_gen','score','elapsed_time'])
    df.to_csv('results/data/Turbines/ConstantAccuracyAnalysis/df_ConstantAccuracyAnalysis_acc'+str(accuracy)+'.csv')

# Guardar lista con valores de accuracy.
np.save('results/data/Turbines/ConstantAccuracyAnalysis/list_acc',list_acc)

# Guardar limite de entrenamiento.
np.save('results/data/Turbines/ConstantAccuracyAnalysis/max_time',max_time)

