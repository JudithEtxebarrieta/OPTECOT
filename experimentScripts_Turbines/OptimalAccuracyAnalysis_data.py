#==================================================================================================
# LIBRERIAS
#==================================================================================================
import numpy as np
import cma
import time
import os
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt

#==================================================================================================
# FUNCIONES
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Funciones auxiliares para definir el accuracy apropiado en cada momento del proceso.
#--------------------------------------------------------------------------------------------------
# FUNCION 1 (Calculo de la correlacion de Spearman entre dos secuencias)
def spearman_corr(x,y):
    return sc.stats.spearmanr(x,y)[0]

# FUNCION 2 (Convertir vector de scores en vector de ranking)
def from_scores_to_ranking(list_scores):
    list_pos_ranking=np.argsort(np.array(list_scores))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

# FUNCION 3 (Generar lista con los scores asociados a cada diseno que forma la generacion)
# Parametros:
#   >population: lista con las soluciones que forman la generacion.
#   >accuracy: accuracy fijado como optimo en la generacion anterior.
#   >count_time_acc: True o False si se quieren sumar o no respectivamente, el tiempo de evaluacion
#    como tiempo adicional para ajustar el accuracy.
#   >count_time_gen: True o False si se quiere sumar o no respectivamente, el tiempo de evaluacion
#    como tiempo natural para la evaluacion de la generacion.
# Devolver: lista con los scores asociados a cada solucion que forma la generacion.

def generation_score_list(population,accuracy,count_time_acc=True,count_time_gen=False):
    global time_acc,time_proc

    # Evaluar poblacion.
    list_scores=[]
    for sol in population:
        t=time.time()
        score=fitness_function(sol,N=int(default_N*accuracy))
        elapsed_time=time.time()-t
        if count_time_acc and not count_time_gen:
            time_acc+=elapsed_time
        if count_time_gen:
            time_proc+=elapsed_time
        list_scores.append(score)

    return list_scores
#--------------------------------------------------------------------------------------------------
# Funciones asociadas a los heuristicos que se aplicaran para ajustar el accuracy..
#--------------------------------------------------------------------------------------------------
# FUNCION 4 (Implementacion adaptada del metodo de biseccion)
# Parametros:
#   >init_acc: accuracy inicial (el minimo considerado).
#   >population: lista con las soluciones que forman la generacion.
#   >train_seed: semilla de entrenamiento.
#   >threshold: umbral del metodo de biseccion con el que se ira actualizando el intervalo que contiene el valor de accuracy optimo.
# Devuelve: 
#   >prev_m: valor de accuracy seleccionado como optimo.
#   >last_time_acc_increase: tiempo de evaluacion consumido en la ultima iteracion del metodo de biseccion.

def bisection_method(lower_time,upper_time,population,train_seed,sample_size,interpolation_pts,threshold=0.95):

    # Inicializar limite inferior y superior.
    time0=lower_time
    time1=upper_time   

    # Punto intermedio.
    prev_m=time0
    m=(time0+time1)/2

    # Funcion para calcular la correlacion entre los rankings de las sample_size superficies aleatorias
    # usando el accuracy actual y el maximo.
    def similarity_between_current_best_acc(acc,population,train_seed,first_iteration):
        global time_acc

        # Seleccionar de forma aleatoria sample_size superficies que forman la generacion.
        random.seed(train_seed)
        ind_sol=random.sample(range(len(population)),sample_size)
        list_solutions=list(np.array(population)[ind_sol])

        # Guardar los scores asociados a cada solucion seleccionada.
        t=time.time()
        best_scores=generation_score_list(list_solutions,1,count_time_acc=first_iteration)# Con el maximo accuracy. 
        new_scores=generation_score_list(list_solutions,acc)# Accuracy nuevo. 
        last_time_acc_increase=time.time()-t

        # Obtener vectores de rankings asociados.
        new_ranking=from_scores_to_ranking(new_scores)# Accuracy nuevo. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximo accuracy. 
                
        # Comparar ambos rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)

        return metric_value,last_time_acc_increase

    # Reajustar limites del intervalo hasta que este tenga un rango lo suficientemente pequeno.
    first_iteration=True
    stop_threshold=(time1-time0)*0.1
    while time1-time0>stop_threshold:
        metric_value,last_time_acc_increase=similarity_between_current_best_acc(np.interp(m,interpolation_pts[0],interpolation_pts[1]),population,train_seed,first_iteration)
        if metric_value>=threshold:
            time1=m
        else:
            time0=m

        prev_m=m
        m=(time0+time1)/2

        first_iteration=False
    return np.interp(prev_m,interpolation_pts[0],interpolation_pts[1]),last_time_acc_increase

# FUNCION 5 (Ejecutar heuristicos durante el proceso de entrenamiento)
# Parametros:
#   >gen: numero de generacion en el algoritmo CMA-ES.
#   >min_acc: accuracy minimo a considerar en el metodo de biseccion.
#   >acc: accuracy asociado a la generacion anterior.
#   >population: lista con las soluciones que forman la generacion.
#   >train_seed: semilla de entrenamiento.
#   >list_variances: lista con las varianzas de los scores de las anteriores generaciones.
#   >heuristic: numero que identifica al heuristico que se desea considerar.
#   >param: valor del parametro asociado al heuristico que se desea aplicar.
# Devuelve: 
#   >acc: valor de accuracy seleccionado como optimo.
#   >time_best_acc: tiempo de evaluacion consumido en la ultima iteracion del metodo de biseccion.

def execute_heuristic(gen,acc,population,train_seed,list_variances,heuristic,param):

    global time_proc
    global time_acc,time_best_acc 
    global max_time
    global last_time_heuristic_accepted
    global unused_bisection_executions

    time_best_acc=0

    # Para el metodo de biseccion: tamano de muestra, frecuencia y expresion de interpolacion.
    df_sample_freq=pd.read_csv('results/data/general/sample_size_freq_'+str(sample_size_freq)+'.csv',index_col=0)
    df_interpolation=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/df_Bisection.csv')
    sample_size=int(df_sample_freq[df_sample_freq['env_name']=='Turbines']['sample_size'])
    heuristic_freq=float(df_sample_freq[df_sample_freq['env_name']=='Turbines']['frequency_time'])
    interpolation_acc=list(df_interpolation['accuracy'])
    interpolation_time=list(df_interpolation['cost_per_eval'])
    lower_time=min(interpolation_time)
    upper_time=max(interpolation_time)

   
    # HEURISTICO I de Symbolic Regressor: Biseccion de generacion en generacion (el umbral es el parametro).
    if heuristic=='I': 
        if gen==0:
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=param)            
        else:
            if (time_acc+time_proc)-last_time_heuristic_accepted>=heuristic_freq:
                acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=param)

    # HEURISTICO II de Symbolic Regressor: Biseccion con definicion automatica para frecuencia 
    # de actualizacion de accuracy (depende de parametro) y umbral del metodo de biseccion fijado en 0.95.
    if heuristic=='II': 
        if gen==0: 
            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
            unused_bisection_executions=0
        else:
            if len(list_variances)>=param+1:
                # Funcion para calcular el intervalo de confianza.
                def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                    mean_list=[]
                    for i in range(bootstrap_iterations):
                        sample = np.random.choice(data, len(data), replace=True) 
                        mean_list.append(np.mean(sample))
                    return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

                variance_q05,variance_q95=bootstrap_confidence_interval(list_variances[(-2-param):-2])
                last_variance=list_variances[-1]
                
                # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
                if last_variance<variance_q05 or last_variance>variance_q95:

                    if (time_proc+time_acc)-last_time_heuristic_accepted>=heuristic_freq:   
                        unused_bisection_executions+=int((time_proc+time_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                        acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
                    else:
                        if unused_bisection_executions>0:
                            acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
                            unused_bisection_executions-=1
            else:
                if (time_acc+time_proc)-last_time_heuristic_accepted>=heuristic_freq:
                    acc,time_best_acc=bisection_method(lower_time,upper_time,population,train_seed,sample_size,[interpolation_time,interpolation_acc])
                            
    return acc,time_best_acc
    
#--------------------------------------------------------------------------------------------------
# Funciones para el proceso de aprendizaje o busqueda del optimo diseno de turbina.
#--------------------------------------------------------------------------------------------------
def transform_turb_params(scaled_x, blade_number):

    # Definir rangos de los parametros que definen el diseno de la turbina.
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
    real_x = scaled_x * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

    return [blade_number]+list(real_x[:-1])+[round(real_x[-1])]


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

def learn(seed,blade_number,heuristic,heuristic_param,accuracy=1,popsize=20):
    global time_best_acc, last_time_heuristic_accepted
    
    # Inicializar CMA-ES.
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(5), 0.33,inopts={'bounds': [0, 1],'seed':seed,'popsize':popsize})

    # Inicializar contadores de tiempo.
    global time_proc,time_acc
    time_proc=0
    time_acc=0

    # Evaluar los disenos de las generaciones hasta agotar el tiempo maximo definido por el accuracy maximo.
    n_gen=0
    while time_proc+time_acc<max_time:

        # Nueva generacion.
        solutions = es.ask()

        # Transformar los valores escalados de los parametros a los valores reales.
        list_turb_params=[transform_turb_params(x, blade_number) for x in solutions]

        # Aplicar el heuristico.
        if n_gen==0:
            accuracy,time_best_acc=execute_heuristic(n_gen,accuracy,list_turb_params,seed,[],heuristic,heuristic_param)
        else:
            df_seed=pd.DataFrame(df)
            df_seed=df_seed[df_seed[1]==seed]
            accuracy,time_best_acc=execute_heuristic(n_gen,accuracy,list_turb_params,seed,list(df_seed[6]),heuristic,heuristic_param)


        # Obtener scores por evaluacion y actualizar contadores de tiempo.
        list_scores=generation_score_list(list_turb_params,accuracy,count_time_gen=True)
        if time_best_acc!=0:
            time_acc-=time_best_acc
            last_time_heuristic_accepted=time_proc+time_acc
            readjustement=True
        else:
            readjustement=False

        # Para construir la siguiente generacion.
        es.tell(solutions, list_scores)

        # Acumular datos de interes.
        test_score= fitness_function(transform_turb_params(es.result.xbest,blade_number))
        df.append([heuristic_param,seed,n_gen,-test_score,readjustement,accuracy,np.var(list_scores),time_proc,time_acc,time_acc+time_proc])

        n_gen+=1
        
#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Lista de semillas.
list_seeds=np.range(2,102,1)

# Parametros.
blade_number=3
default_N=100
sample_size_freq='BisectionOnly'
max_time=np.load('results/data/Turbines/ConstantAccuracyAnalysis/max_time.npy')

# Definicion de heuristicos que se desean ejecutar.
list_args=[['I',0.8],['I',0.95],['II',5],['II',10]]

# Guardar base de datos con informacion de interes asociada al entrenamiento.
for arg in list_args:
    heuristic=arg[0]
    heuristic_param=arg[1]

    df=[]

    for seed in tqdm(list_seeds):
        learn(seed,blade_number,heuristic,heuristic_param)

    df=pd.DataFrame(df,columns=['heuristic_param','seed','n_gen','score','readjustement','accuracy','variance','elapsed_time_proc','elapsed_time_acc','elapsed_time'])
    df.to_csv('results/data/Turbines/OptimalAccuracyAnalysis/df_OptimalAccuracyAnalysis_h'+str(heuristic)+'_param'+str(heuristic_param)+'.csv')


