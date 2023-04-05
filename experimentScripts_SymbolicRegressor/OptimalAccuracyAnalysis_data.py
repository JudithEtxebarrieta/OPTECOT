# En este script se comienza por implementar un total de 14 heuristicos con intencion de automatizar el 
# ajuste del accuracy durante el proceso de entrenamiento. Tras observar las ventajas y desventajas de
# los heuristicos anteriores, finalmente se diseñan 2 heuristicos finales (I y II) con las caracteristicas 
# resaltadas de los anteriores. Se almacenan y guardan los datos  obtenidos durante la ejecucion de cada 
# uno de los heurśiticos.

#==================================================================================================
# LIBRERIAS
#==================================================================================================
# Para mi codigo.
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pandas as pd
from tqdm import tqdm
import scipy as sc
import random

# Para las modificaciones.
import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from gplearn._program import _Program
from gplearn.fitness import _fitness_map, _Fitness
from gplearn.functions import _function_map, _Function, sig1 as sigmoid
from gplearn.utils import _partition_estimators
from gplearn.utils import check_random_state

from gplearn.genetic import _parallel_evolve, MAX_INT
from gplearn.genetic import BaseSymbolic
import multiprocessing as mp
from itertools import combinations
import os

#==================================================================================================
# NUEVAS FUNCIONES
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Funciones para el proceso de aprendizaje o busqueda de la superficie.
#--------------------------------------------------------------------------------------------------
# FUNCION 1
# Parametros:
#   >z_test: terceras coordenadas reales de los puntos de la superficie.
#   >z_pred: terceras coordenadas obtenidas a partir de la superficie predicha.
# Devuelve: el error absoluto medio de las dos listas anteriores.

def mean_abs_err(z_test,z_pred):
    mae=sum(abs(z_test-z_pred))/len(z_test)
    return mae

# FUNCION 2
# Parametros:
#   >n_sample: numero de puntos que se desean construir.
#   >seed: semilla para la seleccion aleatoria de los puntos.
#   >expr_surf: expresion de la superficie de la cual se quiere extraer la muestra de puntos.
# Devuelve: base de datos con las tres coordenadas de los puntos de la muestra.

def build_pts_sample(n_sample,seed,expr_surf):

    # Fijar la semilla.
    rng = check_random_state(seed)

    # Mallado aleatorio (x,y).
    xy_sample=rng.uniform(-1, 1, n_sample*2).reshape(n_sample, 2)
    x=xy_sample[:,0]
    y=xy_sample[:,1]

    # Calcular alturas correspondientes (valor z).
    z_sample=eval(expr_surf)

    # Todos los datos en un array.
    pts_sample=np.insert(xy_sample, xy_sample.shape[1], z_sample, 1)

    return pts_sample

# FUNCION 3
# Parametros:
#   >pts_set: conjunto de punts del que se desea extraer una submuestra.
#   >n_sample: numero de puntos con el que estara formada la submuestra.
# Devuelve: base de datos con las tres coordenadas de los puntos de la submuestra (una submuestra de las filas de pts_set).

def select_pts_sample(pts_set,n_sample):
    pts_sample=pts_set[:n_sample]
    return np.array(pts_sample)

# FUNCION 4
# Parametros:
#   >df_test_pts: base de datos con las tres coordenadas de los puntos que forman el 
#    conjunto de validacion.
#   >est_surf: superficie seleccionada en el proceso GA de entrenamiento.
# Devuelve: error absoluto medio.

def evaluate(df_test_pts,est_surf):

    # Dividir base de datos con las coordenadas de los puntos.
    xy_test=df_test_pts[:,[0,1]]
    z_test=df_test_pts[:,2]

    # Calcular el valor de las terceras coordenadas con las superficie seleccionada.
    z_pred=est_surf.predict(xy_test)

    # Calcular score asociado al conjunto de puntos para la superficie seleccionada.
    score=mean_abs_err(z_test, z_pred)

    return score   

# FUNCION 5
# Parametros:
#   >inti_acc: valor del accuracy inicial.
#   >train_seed: semilla de entrenamiento.
#   >df_test_pts: base de datos con las tres coordenadas de los puntos que forman el conjunto de validacion.
#   >heuristic: identificador del heuristico que se esta aplicando.
#   >heuristic_param: parametro del heuristico que se esta considerando.
# Devuelve: superficie seleccionada.

def learn(init_acc,train_seed,df_test_pts,heuristic,heuristic_param):

    # Cambiar cardinal predefinido.
    train_n_pts=int(default_train_n_pts*init_acc)

    # Inicializar conjunto de entrenamiento.
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)

    # Definicion del algoritmo genetico con el cual se encontraran la superficie.
    est_surf=SymbolicRegressor(random_state=train_seed)

    # Ajustar la superficie a los puntos.
    xy_train=df_train_pts[:,[0,1]]
    z_train=df_train_pts[:,2]
    est_surf.fit(init_acc,xy_train, z_train,train_seed,df_test_pts,heuristic,heuristic_param)    

    return est_surf._program 

#--------------------------------------------------------------------------------------------------
# Funciones auxiliares para definir el accuracy apropiado en cada momento del proceso.
#--------------------------------------------------------------------------------------------------
# FUNCION 6 (Calculo de la correlacion de Spearman entre dos secuencias)
def spearman_corr(x,y):
    return sc.stats.spearmanr(x,y)[0]

# FUNCION 7 (Calculo dela distancia Tau Kendall inversa normalizada entre dos secuencias)
def inverse_normalized_tau_kendall(x,y):
    # Numero de pares con orden inverso.
    pairs_reverse_order=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            case1 = x[i] < x[j] and y[i] > y[j]
            case2 = x[i] > x[j] and y[i] < y[j]

            if case1 or case2:
                pairs_reverse_order+=1  
    
    # Numero de pares total.
    total_pairs=len(list(combinations(x,2)))

    # Distancia tau Kendall normalizada.
    tau_kendall=pairs_reverse_order/total_pairs

    return 1-tau_kendall

# FUNCION 8 (Convertir vector de scores en vector de rankings)
def from_scores_to_ranking(list_scores):
    list_pos_ranking=np.argsort(np.array(list_scores))
    ranking=[0]*len(list_pos_ranking)
    i=0
    for j in list_pos_ranking:
        ranking[j]=i
        i+=1
    return ranking

# FUNCION 9 (Generar lista con los scores asociados a cada superficie que forma la generacion)
# Parametros:
#   >list_surfaces: lista con las expresiones de las superficies que forman la generacion.
#   >df_pts: conjunto de puntos sobre el que se desea evaluar cada superficie.
#   >count_evaluations_acc: True o False en el caso en que se quiera o no sumar el numero de evaluaciones gastadas como 
#    evaluaciones extra, respectivamente.
#   >all_gen_evaluation: True o False en el caso en que se quiera o no sumar el numero de evaluaciones gastadas como 
#    evaluaciones de procedimiento, respectivamente.
#   >gen_variance: True o False ne el caso en que se quiera o no devolver la varianza de los scores de la generacion.
# Devuelve: lista de scores junto a la varianza de los mismos (en caso que asi se indique).

def generation_score_list(list_surfaces,df_pts,
                          count_evaluations_acc=True,all_gen_evaluation=False,
                          gen_variance=False):
    
    # Inicializar lista de scores.
    list_scores=[]

    # Dividir base de datos con las coordenadas de los puntos.
    X=df_pts[:,[0,1]]
    y=df_pts[:,2]

    # Evaluar cada superficie que forma la generacion con el accuracy indicado.
    for expr_surf in list_surfaces:

        # Calcular el valor de las terceras coordenadas con las superficie seleccionada.
        y_pred=expr_surf.execute(X)

        # Calcular score asociado al conjunto de puntos para la superficie seleccionada.
        score=mean_abs_err(y, y_pred)

        # Añadir score a la lista.
        list_scores.append(score)

        # Actualizar contadores de evaluaciones dadas.
        if all_gen_evaluation:
            global n_evaluations
            n_evaluations+=len(y)
        else:
            if count_evaluations_acc:
                global n_evaluations_acc
                n_evaluations_acc+=len(y)
    
    if gen_variance:
        variance=np.var(list_scores)
        return list_scores,variance
    else:
        return list_scores

# FUNCION 10 (Eliminar de una lista en elemento de una posicion)
def idx_remove(list,idx):
    new_list=[]
    for i in range(len(list)):
        if i!=idx:
            new_list.append(list[i])
    return new_list

#--------------------------------------------------------------------------------------------------
# Funciones asociadas al metodo de biseccion para los heuristicos de prueba 
# (1,2,3,4,5,6,7,8,9,10,11,12,13 y 14).
#--------------------------------------------------------------------------------------------------
# FUNCION 11 (Implementacion adaptada del metodo de biseccion)
# Parametros:
#   >init_acc: accuracy inicial (el minimo considerado).
#   >current_acc: accuracy actual (el definido mas recientemente).
#   >list_surf_gen: lista con las expresiones de las superficies que forman la generacion.
#   >train_seed: semilla de entrenamiento.
#   >threshold: umbral del metodo de biseccion con el que se ira actualizando el intervalo que contiene el valor de accuracy optimo.
#   >metric: 'spearman' o 'taukendall'.
#   >random_sample: True o False, en caso de que se quiera aplicar el metodo de biseccion con el 10% aleatorio o mejor de las superficies
#    que forman la generacion, respectivamente.
#   >fitness: None o lista con los scores asociados a la generacion.
#   >first_gen: True o False en caso que se este en la primera o siguientes iteraciones del metodo de biseccion, respectivamente.
#   >change_threshold: 'None' en caso de que se quiera considerar un valor constante para el umbral; 'IncreasingMonotone' en caso de que se
#    quiera actualizar el umbral con el ultimo valor de la metrica que a superado el umbral anterior; 'NonMonotone' en caso de que se quiera 
#    actualizar el umbral con el valor de la metrica asociado al accuracy seleccionado como optimo.
# Devuelve: valor de accuracy seleccionado como optimo junto al nuevo umbral para el metodo de biseccion (en caso que asi se indique).

def bisection_method(init_acc,current_acc,list_surf_gen,train_seed,threshold,metric,
                     random_sample=True,fitness=None,first_gen=False,change_threshold='None'):

    # Inicializar limite inferior y superior.
    acc0=init_acc
    acc1=1    

    # Punto intermedio.
    prev_m=current_acc
    m=(acc0+acc1)/2
    
    # Funcion para calcular la correlacion entre los rankings del 10% aleatorio/mejor de las superficies
    # usando el accuracy actual y el maximo.
    def similarity_between_current_best_acc(current_acc,acc,list_surf_gen,train_seed,metric,first_iteration,actual_n_evaluations_acc,random_sample,fitness):

        if random_sample:
            # Seleccionar de forma aleatoria el 10% de las superficies que forman la generacion.
            random.seed(train_seed)
            ind_surf=random.sample(range(len(list_surf_gen)),int(len(list_surf_gen)*0.1))
            list_surfaces=list(np.array(list_surf_gen)[ind_surf])


        else:
            # Seleccionar el 10% mejor de las superficies que forman la generacion segun 
            # el accuracy de la generacion anterior.
            all_current_ranking=from_scores_to_ranking(fitness)
            if first_iteration:
                global n_evaluations_acc
                n_evaluations_acc+=int(default_train_n_pts*current_acc)*len(list_surf_gen)

            list_surfaces=list_surf_gen
            for ranking_pos in range(int(len(list_surf_gen)*0.1),len(list_surf_gen)):
                # Eliminar posiciones y superficies que no se usaran.
                ind_remove=all_current_ranking.index(ranking_pos)
                all_current_ranking.remove(ranking_pos)
                list_surfaces=idx_remove(list_surfaces,ind_remove)

        # Guardar los scores asociados a cada superficie seleccionada.
        best_scores=generation_score_list(list_surfaces,default_df_train_pts,count_evaluations_acc=first_iteration)# Con el maximo accuracy. 
        new_df_train_pts=select_pts_sample(default_df_train_pts,int(default_train_n_pts*acc))
        new_scores=generation_score_list(list_surfaces,new_df_train_pts)# Accuracy nuevo. 

        # Obtener vectores de rankings asociados.
        new_ranking=from_scores_to_ranking(new_scores)# Accuracy nuevo. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximo accuracy. 
                
        # Comparar ambos rankings.
        if metric=='spearman':
            metric_value=spearman_corr(new_ranking,best_ranking)
        if metric=='taukendall':
            metric_value=inverse_normalized_tau_kendall(new_ranking,best_ranking)

        return metric_value, n_evaluations_acc-actual_n_evaluations_acc

    # Reajustar limites del intervalo hasta que este tenga un rango lo suficientemente pequeño o hasta alcanzar
    # el numero maximo de evaluaciones.
    global n_evaluations_acc
    first_iteration=True
    continue_bisection_method=True
    max_n_evaluations=default_train_n_pts*len(list_surf_gen)# Para no superar las evaluaciones realizadas por defecto (accuracy=1)
    next_upper_threshold=[]

    while acc1-acc0>0.1 and continue_bisection_method:
        metric_value,extra_n_evaluations_acc=similarity_between_current_best_acc(current_acc,m,list_surf_gen,train_seed,metric,first_iteration,n_evaluations_acc,random_sample,fitness)
        if metric_value>=threshold:
            acc1=m

            next_upper_threshold.append(metric_value)

        else:
            acc0=m

        if first_gen and n_evaluations_acc+int(default_train_n_pts*m)*len(list_surf_gen)>max_n_evaluations:
            continue_bisection_method=False
            n_evaluations_acc-=extra_n_evaluations_acc
        else:
            prev_m=m
            m=(acc0+acc1)/2
        
        first_iteration=False
    
    if change_threshold=='IncreasingMonotone':
        if len(next_upper_threshold)==0:
            next_threshold=threshold
        else:
            next_threshold=next_upper_threshold[-1]
        return prev_m,next_threshold
    if change_threshold=='NonMonotone':
        next_threshold=metric_value
        return prev_m,next_threshold
    if change_threshold=='None':
        return prev_m

# FUNCION 12 (ajustar con el metodo de biseccion el accuracy en la primera generacion)
# Parametros:
#   >init_acc: accuracy inicial (el minimo considerado).
#   >list_surf_gen: lista con las expresiones de las superficies que forman la generacion.
#   >train_seed: semilla de entrenamiento.
#   >metric: 'spearman' o 'taukendall'.
#   >threshold: umbral del metodo de biseccion con el que se ira actualizando el intervalo que contiene el valor de accuracy optimo.
# Devuelve: 
#   >acc: accuracy seleccionado como optimo.
#   >X,y: coordenadas del conjunto de puntos de entrenamiento asociado al accuracy acc.
#   >fitness: lista de scores de la generacion calculados a partir del conjunto de puntos de entrenamiento recien fijado.
#   >acc_split: el 10% del accuracy restante que queda (1-acc).
#   >threshold: umbral del metodo de biseccion empleado.
#   >variance: varianza de los scores de la generacion.

def set_initial_accuracy(init_acc,list_surf_gen,train_seed,metric,threshold=0.95,change_threshold='None',sample_size=None):

    # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
    if change_threshold !='None':
        acc,next_threshold=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,metric,first_gen=True,change_threshold=change_threshold,sample_size=sample_size)
    else:
        acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,metric,first_gen=True)

    # Calcular conjunto de entrenamiento correspondiente.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # Calcular vector fitness de la generacion usando el accuracy definido.
    fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
    global n_evaluations_acc
    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

    # Definir el incremento de accuracy que se usara en las siguientes iteraciones en caso de 
    # tener que incrementar el accuracy.
    global acc_split
    acc_split=(1-acc)*0.1

    if change_threshold !='None':
        return acc,X,y,fitness,acc_split,threshold,next_threshold,variance
    else:
        return acc,X,y,fitness,acc_split,threshold,variance
#--------------------------------------------------------------------------------------------------
# Funciones asociadas al metodo de biseccion para los heuristicos finales (I y II)
#--------------------------------------------------------------------------------------------------
# FUNCION 13 (Implementacion adaptada del metodo de biseccion)
def customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,threshold,sample_size,interpolation_pts):

    # Inicializar limite inferior y superior.
    time0=lower_time
    time1=upper_time    

    # Punto intermedio.
    prev_m=lower_time
    m=(time0+time1)/2
    
    # Funcion para calcular la correlacion entre los rankings de las sample_size superficies
    # usando el accuracy actual y el maximo.
    def similarity_between_current_best_acc(time,list_surf_gen,train_seed,first_iteration):

        # Seleccionar de forma aleatoria sample_size superficies que forman la generacion.
        random.seed(train_seed)
        ind_surf=random.sample(range(len(list_surf_gen)),sample_size)
        list_surfaces=list(np.array(list_surf_gen)[ind_surf])

        # Guardar los scores asociados a cada superficie seleccionada.
        best_scores=generation_score_list(list_surfaces,default_df_train_pts,count_evaluations_acc=first_iteration)# Con el maximo accuracy. 
        new_df_train_pts=select_pts_sample(default_df_train_pts,int(default_train_n_pts*time))
        new_scores=generation_score_list(list_surfaces,new_df_train_pts)# Accuracy nuevo. 

        # Obtener vectores de rankings asociados.
        new_ranking=from_scores_to_ranking(new_scores)# Accuracy nuevo. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximo accuracy. 
                
        # Comparar ambos rankings.
        metric_value=spearman_corr(new_ranking,best_ranking)

        return metric_value

    # Reajustar limites del intervalo hasta que este tenga un rango lo suficientemente pequeño (el 10% de la longitud maxima).
    global n_evaluations_acc
    first_iteration=True
    continue_bisection_method=True
    stop_threshold=(time1-time0)*0.1

    while time1-time0>stop_threshold and continue_bisection_method:
        metric_value=similarity_between_current_best_acc(np.interp(m,interpolation_pts[0],interpolation_pts[1]),list_surf_gen,train_seed,first_iteration)
        if metric_value>=threshold:
            time1=m
        else:
            time0=m

        prev_m=m
        m=(time0+time1)/2
        
        first_iteration=False


    return np.interp(prev_m,interpolation_pts[0],interpolation_pts[1])

# FUNCION 14 (ajustar con el metodo de biseccion el accuracy en la primera generacion)
def customized_set_initial_accuracy(lower_time,upper_time,list_surf_gen,train_seed,sample_size,interpolation_pts,threshold=0.95):

    # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
    acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,threshold,sample_size,interpolation_pts)

    # Calcular conjunto de entrenamiento correspondiente.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # Calcular vector fitness de la generacion usando el accuracy definido.
    fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
    global n_evaluations_acc
    n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

    return acc,X,y,fitness,threshold,variance


#--------------------------------------------------------------------------------------------------
# Funciones que implementan diferentes heuristicos para actualizar el accuracy.
#--------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________
# Heuristicos de prueba diseñados para la busqueda de la definicion del heuristico final.

# HEURISTICO 1 (accuracy ascendente)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Valoracion de cambio de accuracy (depende de parametro): se comprobara si la correlacion entre rankings
# (accuracy actual y maximo) del 10% aleatorio de las superficies de la generacion supera o no cierto umbral (parametro). 
# >>Definicion de cambio de accuracy: funcion dependiente de la correlacion anterior.
def update_accuracy_heuristic1(acc,X,y,population,fitness,train_seed,param):

    global n_evaluations_acc
    global n_evaluations

    if acc<1:
        # Funcion para el calculo de incremento de accuracy.
        def acc_split(corr,acc_rest,param):
            if param=='logistic':
                split=(1/(1+np.exp(12*(corr-0.5))))*acc_rest
            else:
                if corr<=param[0]:
                    split=acc_rest
                else:
                    split=-acc_rest*(((corr-param[0])/(1-param[0]))**(1/param[1]))+acc_rest
            return split
    
        # Seleccionar de forma aleatoria el 10% de las superficies que forman la generacion.
        random.seed(train_seed)
        ind_surf=random.sample(range(len(population)),int(len(population)*0.1))
        list_surfaces=list(np.array(population)[ind_surf])

        # Guardar los scores asociados a cada superficie seleccionada.
        best_scores=generation_score_list(list_surfaces,default_df_train_pts)# Con el maximo accuracy. 
        current_scores=list(np.array(fitness)[ind_surf])# Accuracy actual.

        # Actualizar numero de evaluaciones extra empleadas para la definicion del accuracy.
        n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)

        # Obtener vectores de rankings asociados.
        current_ranking=from_scores_to_ranking(current_scores)# Accuracy actual. 
        best_ranking=from_scores_to_ranking(best_scores)# Maximo accuracy. 
                
        # Comparar ambos rankings (calcular coeficiente de correlacion de Spearman).
        corr=spearman_corr(current_ranking,best_ranking)

        # Dependiendo de la similitud entre los rankings calcular el split en el accuracy para la siguiente generacion.
        split=acc_split(corr,1-acc,param)

        # Modificar accuracy.
        prev_acc=acc
        acc=acc+split

        # Calcular nuevo conjunto de entrenamiento.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # Si el accuracy asciende y si no lo hace.
        if prev_acc!=acc:
            # Calcular vector fitness de la generacion usando el accuracy definido.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            # Actualizar numero de evaluaciones del proceso.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
            n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)
    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

# HEURISTICO 2 (accuracy ascendente)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Valoracion de cambio de accuracy (depende de parametro): se comprobara si la correlacion entre rankings 
# (accuracy actual y siguientes en una lista predefinida) del 10% aleatorio de las superficies de la generacion supera o no cierto umbral (parametro). 
# >>Definicion de cambio de accuracy: accuracy actual mas el split constante definido tras aplicar el metodo de biseccion
# (con umbral 0.95) al comienzo.
def update_accuracy_heuristic2(acc,init_acc,X,y,population,fitness,train_seed,param):
    global n_evaluations_acc
    global n_evaluations

    if acc<1:
        # Definir lista de accuracys duplicando el valor del accuracy actual sucesivamente hasta
        # llegar al maximo.
        list_acc=[init_acc]
        next_acc=list_acc[-1]*2
        while next_acc<1:
            list_acc.append(next_acc)
            next_acc=list_acc[-1]*2
        if 1 not in list_acc:
            list_acc.append(1)
    
        # Seleccionar de forma aleatoria el 10% de las superficies que forman la generacion.
        random.seed(train_seed)
        ind_surf=random.sample(range(len(population)),int(len(population)*0.1))
        list_surfaces=list(np.array(population)[ind_surf])

        # Guardar los scores asociados a cada superficie seleccionada y calcular el ranking con el 
        # accuracy actual.
        current_scores=list(np.array(fitness)[ind_surf])
        current_ranking=from_scores_to_ranking(current_scores)

        # Actualizar numero de evaluaciones extra empleadas para la definicion del accuracy.
        n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)

        # Mientras la correlacion del ranking actual con el ranking asociado a un accuracy mayor no 
        # sea inferior al umbral seguir probando con el resto de accuracys.
        possible_acc=list(np.array(list_acc)[np.array(list_acc)>acc])
        ind_next_acc=0
        corr=1
        while corr>param and ind_next_acc<len(possible_acc):

            # Nuevo conjunto de puntos para evaluar las superficies.
            next_train_n_pts=int(default_train_n_pts*possible_acc[ind_next_acc])
            next_df_train_pts=select_pts_sample(default_df_train_pts,next_train_n_pts)

            # Guardar scores de las superficies seleccionadas calculados con el accuracy siguiente y
            # obtener el ranking correspondiente.
            next_scores=generation_score_list(list_surfaces,next_df_train_pts)
            next_ranking=from_scores_to_ranking(next_scores)

            # Comparar ambos rankings (calcular coeficiente de correlacion de Spearman).
            corr=spearman_corr(current_ranking,next_ranking)

            # Actualizacion de indice de accuracy.
            ind_next_acc+=1
        
        # Modificar accuracy.
        prev_acc=acc
        if corr<param:
            acc+=acc_split
            if acc>1:
                    acc=1

        # Calcular nuevo conjunto de entrenamiento.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # Si el accuracy asciende y si no lo hace.
        if prev_acc!=acc:
            # Calcular vector fitness de la generacion usando el accuracy definido.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            # Actualizar numero de evaluaciones del proceso.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
            n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)
    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

# HEURISTICO 3 (accuracy ascendente)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Valoracion de cambio de accuracy (depende de parametro):  se comprobara si la correlacion entre rankings
# (accuracy minimo y actual) de las superficies de la generacion supera o no cierto umbral (parametro). 
# >>Definicion de cambio de accuracy: accuracy actual mas el split constante definido tras aplicar el metodo de biseccion
# (con umbral 0.95) al comienzo.
def update_accuracy_heuristic3(acc,init_acc,X,y,population,fitness,param):

    global n_evaluations_acc
    global n_evaluations

    if acc<1:
        # Guardar los scores asociados a cada superficie.
        list_surfaces=list(population)
        worst_df_train_pts=select_pts_sample(default_df_train_pts,int(default_train_n_pts*init_acc))
        worst_scores=generation_score_list(list_surfaces,worst_df_train_pts)# Con el minimo accuracy. 
        current_scores=fitness# Accuracy actual.

        # Actualizar numero de evaluaciones extra empleadas para la definicion del accuracy.
        n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)
        
        # Obtener vectores de rankings asociados.
        current_ranking=from_scores_to_ranking(current_scores)# Accuracy actual. 
        worst_ranking=from_scores_to_ranking(worst_scores)# Minimo accuracy. 
                
        # Comparar ambos rankings (calcular coeficiente de correlacion de Spearman).
        corr=spearman_corr(current_ranking,worst_ranking)

        # Dependiendo de la similitud entre los rankings considerar un accuracy mayor para la siguiente generacion.
        prev_acc=acc
        if corr>param:
            acc+=acc_split
            if acc>1:
                    acc=1

        # Calcular nuevo conjunto de entrenamiento.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # Si el accuracy asciende y si no lo hace.
        if prev_acc!=acc:
            # Calcular vector fitness de la generacion usando el accuracy definido.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surfaces)
            n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)
    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=default_train_n_pts*len(list(population))
    return acc,X,y,fitness

# HEURISTICO 4 (accuracy ascendente)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Valoracion de cambio de accuracy (depende de parametro): cada cierta frecuencia (parametro) correlacion 
# entre rankings (accuracy actual y maximo) del mejor 10% de superficies de la generacion. 
# >>Definicion de cambio de accuracy: accuracy actual mas el split constante definido tras aplicar el metodo de biseccion
# (con umbral 0.95) al comienzo.
def update_accuracy_heuristic4(acc,X,y,population,fitness,param):
    global last_optimal_evaluations
    global n_evaluations_acc
    global n_evaluations
    if acc<1:
        if (n_evaluations+n_evaluations_acc)-last_optimal_evaluations>=param:

            # Valorar si se debe ascender el accuracy.
            list_surfaces=list(population)
            all_current_ranking=from_scores_to_ranking(fitness)

            n_evaluations_acc+=int(default_train_n_pts*acc)*len(list_surfaces)

            for ranking_pos in range(int(len(population)*0.1),len(population)):
                # Eliminar posiciones y superficies que no se usaran.
                ind_remove=all_current_ranking.index(ranking_pos)
                all_current_ranking.remove(ranking_pos)
                list_surfaces=idx_remove(list_surfaces,ind_remove)
            current_ranking=all_current_ranking

            best_scores=generation_score_list(list_surfaces,default_df_train_pts) 
            best_ranking=from_scores_to_ranking(best_scores)

            corr=spearman_corr(current_ranking,best_ranking)

            # Definir ascenso de accuracy en caso de que se deba ascender.
            prev_acc=acc
            if corr<1:
                acc+=acc_split
                if acc>1:
                    acc=1

            # Calcular nuevo conjunto de entrenamiento.
            train_n_pts=int(default_train_n_pts*acc)
            df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
            X=df_train_pts[:,[0,1]]
            y=df_train_pts[:,2]
            
            # Si el accuracy asciende y si no lo hace.
            if prev_acc!=acc:
                # Calcular vector fitness de la generacion usando el accuracy definido.
                fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
            else:
                n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
                n_evaluations_acc-=int(default_train_n_pts*acc)*len(list_surfaces)

            # Actualizar numero de evaluaciones en los que se ha actualizado el accuracy.
            last_optimal_evaluations=n_evaluations+n_evaluations_acc
        else:
            # Actualizar numero de evaluaciones.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

# HEURISTICO 5 (accuracy ascendente)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Valoracion de cambio de accuracy (depende de parametro): mirar el ultimo descenso de score en que posicion 
# del intervalo de confianza se encuentra. El numero de descensos asociados a las generaciones anteriores
# es el parametro a definir, y a partir de estos descensos es de donde se calculara el intervalo de confianza.. 
# >>Definicion de cambio de accuracy: accuracy actual mas el split constante definido tras aplicar el metodo de biseccion al comienzo.
def update_accuracy_heuristic5(acc,X,y,list_scores,population,fitness,param):
    global n_evaluations
    global n_evaluations_acc

    if acc<1:
        if len(list_scores)>param+1:

            # Funcion para calcular el intervalo de confianza.
            def bootstrap_confidence_interval(data,bootstrap_iterations=1000):
                mean_list=[]
                for i in range(bootstrap_iterations):
                    sample = np.random.choice(data, len(data), replace=True) 
                    mean_list.append(np.mean(sample))
                return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

            # Calcular intervalo de confianza de los descensos anteriores.
            list_scores1=list_scores[(-2-param):-2]
            list_scores2=list_scores[(-1-param):-1]

            list_score_falls=list(np.array(list_scores1)-np.array(list_scores2))
            conf_interval_q05,conf_interval_q95=bootstrap_confidence_interval(list_score_falls[0:-1])
            last_fall=list_score_falls[-1]

            # Actualizar numero de evaluaciones extra empleadas para la definicion del accuracy.
            n_evaluations_acc+=default_train_n_pts*(param+1)
            
            # Definir ascenso de accuracy en caso de que se deba ascender.
            prev_acc=acc
            if last_fall<conf_interval_q05:
                acc+=acc_split
                if acc>1:
                    acc=1

            # Calcular nuevo conjunto de entrenamiento.
            train_n_pts=int(default_train_n_pts*acc)
            df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
            X=df_train_pts[:,[0,1]]
            y=df_train_pts[:,2]

            # Si el accuracy asciende y si no lo hace.
            if prev_acc!=acc:
                # Calcular vector fitness de la generacion usando el accuracy definido.
                fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
            else:
                # Actualizar numero de evaluaciones del proceso.
                n_evaluations+=int(default_train_n_pts*acc)*len(list(population))

        else:
            # Actualizar numero de evaluaciones del proceso.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))
    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=default_train_n_pts*len(list(population))

    return acc,X,y,fitness

# HEURISTICO 6 (accuracy ascendente)
# >>Accuracy inicial: el definido con el metodo de biseccion (el umbral es el parametro).
# >>Actualizacion de accuracy (depende de parametro): por generacion, aplicando el metodo de biseccion con el 
# 10% aleatorio (el umbral es el parametro) y usando como limite inferior del intervalo el accuracy de la 
# iteracion anterior.
def update_accuracy_heuristic6(acc,list_surf_gen,train_seed,fitness,heuristic_param):
    # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
    prev_acc=acc
    acc=bisection_method(acc,acc,list_surf_gen,train_seed,heuristic_param,'spearman')

    # Calcular nuevo conjunto de entrenamiento.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # Si el accuracy asciende y si no lo hace.
    if prev_acc!=acc:
        # Calcular vector fitness de la generacion usando el accuracy definido.
        fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
        global n_evaluations_acc
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
    else:
        # Actualizar numero de evaluaciones del proceso.
        global n_evaluations
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)

    return acc,X,y,fitness

# HEURISTICO 7 (accuracy optimo)
# >>Accuracy inicial: el definido con el metodo de biseccion (el umbral es el parametro).
# >>Actualizacion de accuracy (depende de parametro): por generacion, aplicando el metodo de biseccion 
# (el umbral es el parametro) desde cero con el 10% aleatorio de las superficies que forman la generacion.
def update_accuracy_heuristic7(acc,init_acc,list_surf_gen,train_seed,fitness,heuristic_param):
    global n_evaluations
    global n_evaluations_acc

    # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
    prev_acc=acc
    acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,heuristic_param,'spearman')

    # Calcular nuevo conjunto de entrenamiento.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # Si el accuracy cambia y si no lo hace.
    if prev_acc!=acc:
        # Calcular vector fitness de la generacion usando el accuracy definido.
        fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        
    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

    return acc,X,y,fitness

# HEURISTICO 8 (accuracy optimo)
# >>Accuracy inicial: el definido con el metodo de biseccion (el umbral es el parametro).
# >>Actualizacion de accuracy (depende de parametro): por generacion, aplicando el metodo de 
# biseccion (el umbral es el parametro) desde cero con el 10% mejor de las superficies que 
# forman la generacion.
def update_accuracy_heuristic8(acc,init_acc,list_surf_gen,train_seed,fitness,heuristic_param):
    global n_evaluations
    global n_evaluations_acc
    
    # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
    prev_acc=acc
    acc=bisection_method(init_acc,acc,list_surf_gen,train_seed,heuristic_param,'spearman',random_sample=False,fitness=fitness)

    # Calcular nuevo conjunto de entrenamiento.
    train_n_pts=int(default_train_n_pts*acc)
    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
    X=df_train_pts[:,[0,1]]
    y=df_train_pts[:,2]

    # Si el accuracy cambia y si no lo hace.
    if prev_acc!=acc:
        # Calcular vector fitness de la generacion usando el accuracy definido.
        fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
        n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

    return acc,X,y,fitness

# HEURISTICO 9 (heuristico 7 con frecuencia de actualizacion de accuracy predefinida)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Actualizacion de accuracy (depende de parametro): cada cierta frecuencia (param) se aplicara 
# el metodo de biseccion (con umbral 0.95) desde cero con el 10% aleatorio de las superficies que
# forman la generacion.
def update_accuracy_heuristic9(acc,init_acc,X,y,list_surf_gen,train_seed,fitness,param):
    global n_evaluations
    global n_evaluations_acc
    global last_optimal_evaluations

    if (n_evaluations+n_evaluations_acc)-last_optimal_evaluations>=param:

        # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
        prev_acc=acc
        acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,0.95,'spearman')

        # Calcular nuevo conjunto de entrenamiento.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # Si el accuracy cambia y si no lo hace.
        if prev_acc!=acc:
            # Calcular vector fitness de la generacion usando el accuracy definido.
            fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            
        else:
            # Actualizar numero de evaluaciones del proceso.
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

        # Actualizar numero de evaluaciones en los que se a actualizado el accuracy.
        last_optimal_evaluations=n_evaluations+n_evaluations_acc
    else:
        # Actualizar numero de evaluaciones.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)


    return acc,X,y,fitness

# HEURISTICO 10 (heuristico 9 usando la distancia tau Kendall en lugar de la correlacion de Spearman)
def update_accuracy_heuristic10(acc,init_acc,X,y,list_surf_gen,train_seed,fitness,param):
    global n_evaluations
    global n_evaluations_acc
    global last_optimal_evaluations
    if (n_evaluations+n_evaluations_acc)-last_optimal_evaluations>=param:

        # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
        prev_acc=acc
        acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,0.95,'taukendall')

        # Calcular nuevo conjunto de entrenamiento.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # Si el accuracy cambia y si no lo hace.
        if prev_acc!=acc:
            # Calcular vector fitness de la generacion usando el accuracy definido.
            fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            
        else:
            # Actualizar numero de evaluaciones del proceso.
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)

        # Actualizar numero de evaluaciones en los que se a actualizado el accuracy.
        last_optimal_evaluations=n_evaluations+n_evaluations_acc
    else:
        # Actualizar numero de evaluaciones.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)


    return acc,X,y,fitness


# HEURISTICO 11 (heuristico 7 con frecuencia de actualizacion de accuracy definida por heuristico 5)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Valoracion de cambio de accuracy (depende de parametro): por generacion se comprobara si el nuevo 
# descenso de score se encuentra por debajo del intervalo de confianza de los param descensos anteriores. 
# El parametro param indica el numero de descensos a partir del cual se quiere calcular el intervalo de confianza.
# >>Definicion de cambio de accuracy: en caso de que haya que modificar el accuracy, se aplicara el metodo
# de biseccion (umbral 0.95) desde cero con el 10% aleatorio de las superficies que forman la generacion.
def update_accuracy_heuristic11(init_acc,acc,X,y,list_scores,population,train_seed,fitness,param):
    global n_evaluations

    if len(list_scores)>param+1:

        # Funcion para calcular el intervalo de confianza.
        def bootstrap_confiance_interval(data,bootstrap_iterations=1000):
            mean_list=[]
            for i in range(bootstrap_iterations):
                sample = np.random.choice(data, len(data), replace=True) 
                mean_list.append(np.mean(sample))
            return np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

        # Calcular intervalo de confianza de los descensos anteriores.
        list_scores1=list_scores[(-2-param):-2]
        list_scores2=list_scores[(-1-param):-1]

        list_score_falls=list(np.array(list_scores1)-np.array(list_scores2))
        conf_interval_q05,conf_interval_q95=bootstrap_confiance_interval(list_score_falls[0:-1])
        last_fall=list_score_falls[-1]

        # Actualizar numero de evaluaciones extra empleadas para la definicion del accuracy.
        global n_evaluations_acc
        n_evaluations_acc+=default_train_n_pts*(param+1)
        
        # Definir ascenso de accuracy en caso de que se deba ascender.
        prev_acc=acc
        if last_fall<conf_interval_q05:
            acc=bisection_method(init_acc,init_acc,list(population),train_seed,0.95,'spearman')

        # Calcular nuevo conjunto de entrenamiento.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # Si el accuracy cambia y si no lo hace.
        if prev_acc!=acc:
            # Calcular vector fitness de la generacion usando el accuracy definido.
            fitness=generation_score_list(list(population),df_train_pts,all_gen_evaluation=True) 
        else:
            # Actualizar numero de evaluaciones del proceso.
            n_evaluations+=int(default_train_n_pts*acc)*len(list(population))

    else:
        # Actualizar numero de evaluaciones del proceso.
        n_evaluations+=int(default_train_n_pts*acc)*len(list(population))


    return acc,X,y,fitness

# HEURISTICO 12 (heuristico 7 con frecuencia de actualizacion de accuracy definida automaticamente)
# >>Accuracy inicial: el definido con el metodo de biseccion (umbral 0.95).
# >>Valoracion de cambio de accuracy (depende de parametro): se van almacenando las varianzas de cada 
# generacion calculadas con el accuracy optimo, y con una cierta cantidad de las mas recientes entre ellas
# (param) se calcula un intervalo de confianza. Si la ultima varianza registrada se situa fuera del intervalo
# se procedera a reajustar el accuracy.
# >>Definicion de cambio de accuracy: en caso de que haya que modificar el accuracy, se aplicara el metodo
# de biseccion (umbral 0.95) desde cero con el 10% aleatorio de las superficies que forman la generacion.
def update_accuracy_heuristic12(acc,init_acc,X,y,list_variances,list_surf_gen,train_seed,fitness,param):
    global n_evaluations
    global n_evaluations_acc
    threshold=None

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

        if last_variance<variance_q05 or last_variance>variance_q95:

            # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
            prev_acc=acc
            acc=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,0.95,'spearman')

            # Calcular nuevo conjunto de entrenamiento.
            train_n_pts=int(default_train_n_pts*acc)
            df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
            X=df_train_pts[:,[0,1]]
            y=df_train_pts[:,2]

            fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
        variance=np.var(fitness)

    return acc,X,y,fitness,threshold,variance

# HEURISTICO 13 (heuristico 12 con definicion automatica monotono ascendente para el umbral del metodo de biseccion)
# Cada vez que haya que reajustar el accuracy (definido por el heuristico 12), se recalculara el umbral a considerar
# en el metodo de biseccion para el siguiente reajuste del accuracy. El nuevo umbral sera la correlacion de Spearman 
# con la que se ha superado el umbral actual por ultima vez al aplicar el metodo de biseccion. En caso de que no se supere 
# ninguna vez, el umbral se mantendra.
def update_accuraccy_heuristic13(acc,init_acc,X,y,list_variances,list_surf_gen,train_seed,fitness,threshold,param):
    global n_evaluations
    global n_evaluations_acc
    next_threshold=threshold

    if threshold<1:

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

            if last_variance<variance_q05 or last_variance>variance_q95:

                # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
                prev_acc=acc
                acc,next_threshold=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,'spearman',change_threshold='IncreasingMonotone')

                # Calcular nuevo conjunto de entrenamiento.
                train_n_pts=int(default_train_n_pts*acc)
                df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                X=df_train_pts[:,[0,1]]
                y=df_train_pts[:,2]

                fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        
                # Si el accuracy cambia y si no lo hace.
                if prev_acc!=acc:
                    # Calcular vector fitness de la generacion usando el accuracy definido.
                    fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
                    
                else:
                    # Actualizar numero de evaluaciones del proceso.
                    n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            else:
                n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                variance=np.var(fitness)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        n_evaluations+=default_train_n_pts*len(list_surf_gen)
        variance=np.var(fitness)

    return acc,X,y,fitness,threshold,next_threshold,variance

# HEURISTICO 14 (heuristico 12 con definicion automatica no monotona para el umbral del metodo de biseccion)
# Cada vez que haya que reajustar el accuracy (definido por el heuristico 12), se recalculara el umbral a considerar
# en el metodo de biseccion para el siguiente reajuste del accuracy. El nuevo umbral sera la correlacion de Spearman 
# asociada al accuracy seleccionado como optimo en el metodo de biseccion.
def update_accuraccy_heuristic14(acc,init_acc,X,y,list_variances,list_surf_gen,train_seed,fitness,threshold,param):
    global n_evaluations
    global n_evaluations_acc
    next_threshold=threshold

    if threshold<1:

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

            if last_variance<variance_q05 or last_variance>variance_q95:

                # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
                prev_acc=acc
                acc,next_threshold=bisection_method(init_acc,init_acc,list_surf_gen,train_seed,threshold,'spearman',change_threshold='NonMonotone')

                # Calcular nuevo conjunto de entrenamiento.
                train_n_pts=int(default_train_n_pts*acc)
                df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                X=df_train_pts[:,[0,1]]
                y=df_train_pts[:,2]

                fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
        
                # Si el accuracy cambia y si no lo hace.
                if prev_acc!=acc:
                    # Calcular vector fitness de la generacion usando el accuracy definido.
                    fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
                    
                else:
                    # Actualizar numero de evaluaciones del proceso.
                    n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                    n_evaluations_acc-=int(default_train_n_pts*acc)*int(len(list_surf_gen)*0.1)
            else:
                n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                variance=np.var(fitness)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        n_evaluations+=default_train_n_pts*len(list_surf_gen)
        variance=np.var(fitness)

    return acc,X,y,fitness,threshold,next_threshold,variance

#__________________________________________________________________________________________________
# Heuristicos finales (los heuristico 7 y 12 anteriores pero considerando el tamaño de muestra y
# la frecuencia de aceptacion de los heuristicos predefinidos, y la biseccion sobre el coste por evaluacion para
# aplicar el metodo de biseccion).

# HEURISTICO I (aplicar el metodo de biseccion todas las veces que sea posible)
def update_accuracy_heuristicI(acc,lower_time,upper_time,X,y,list_surf_gen,train_seed,fitness,heuristic_param,sample_size,heuristic_freq,interpolation_pts):
    global n_evaluations
    global n_evaluations_acc
    global last_time_heuristic_accepted

    if (n_evaluations+n_evaluations_acc)-last_time_heuristic_accepted>=heuristic_freq:
        # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
        prev_acc=acc
        acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,heuristic_param,sample_size,interpolation_pts)

        # Calcular nuevo conjunto de entrenamiento.
        train_n_pts=int(default_train_n_pts*acc)
        df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
        X=df_train_pts[:,[0,1]]
        y=df_train_pts[:,2]

        # Si el accuracy cambia y si no lo hace.
        if prev_acc!=acc:
            # Calcular vector fitness de la generacion usando el accuracy definido.
            fitness=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size
            
        else:
            # Actualizar numero de evaluaciones del proceso.
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

        last_time_heuristic_accepted=n_evaluations+n_evaluations_acc
    else:
        # Actualizar numero de evaluaciones.
        n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)

    return acc,X,y,fitness


# HEURISTICO II (aplicar el metodo de biseccion con frecuencia de actualizacion de accuracy definida automaticamente)
def update_accuracy_heuristicII(acc,lower_time,upper_time,X,y,list_variances,list_surf_gen,train_seed,fitness,param,sample_size,heuristic_freq,interpolation_pts):
    global n_evaluations
    global n_evaluations_acc
    global last_time_heuristic_accepted
    global unused_bisection_executions
    threshold=None


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

        if last_variance<variance_q05 or last_variance>variance_q95:
            if (n_evaluations+n_evaluations_acc)-last_time_heuristic_accepted>=heuristic_freq:

                unused_bisection_executions+=int((n_evaluations+n_evaluations_acc-last_time_heuristic_accepted)/heuristic_freq)-1

                # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
                acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,0.95,sample_size,interpolation_pts)

                # Calcular nuevo conjunto de entrenamiento.
                train_n_pts=int(default_train_n_pts*acc)
                df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                X=df_train_pts[:,[0,1]]
                y=df_train_pts[:,2]

                fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

                last_time_heuristic_accepted=n_evaluations+n_evaluations_acc
            else:
                if unused_bisection_executions>0:
                    
                    # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
                    acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,0.95,sample_size,interpolation_pts)

                    # Calcular nuevo conjunto de entrenamiento.
                    train_n_pts=int(default_train_n_pts*acc)
                    df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
                    X=df_train_pts[:,[0,1]]
                    y=df_train_pts[:,2]

                    fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
                    n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

                    unused_bisection_executions-=1
                    last_time_heuristic_accepted=n_evaluations+n_evaluations_acc

                else:
                    n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
                    variance=np.var(fitness)
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)
    else:
        if (n_evaluations+n_evaluations_acc)-last_time_heuristic_accepted>=heuristic_freq:

            # Calcular el minimo accuracy con el que se obtiene la maxima calidad.
            acc=customized_bisection_method(lower_time,upper_time,list_surf_gen,train_seed,0.95,sample_size,interpolation_pts)

            # Calcular nuevo conjunto de entrenamiento.
            train_n_pts=int(default_train_n_pts*acc)
            df_train_pts=select_pts_sample(default_df_train_pts,train_n_pts)
            X=df_train_pts[:,[0,1]]
            y=df_train_pts[:,2]

            fitness,variance=generation_score_list(list_surf_gen,df_train_pts,all_gen_evaluation=True,gen_variance=True) 
            n_evaluations_acc-=int(default_train_n_pts*acc)*sample_size

            last_time_heuristic_accepted=n_evaluations+n_evaluations_acc
            
        else:
            n_evaluations+=int(default_train_n_pts*acc)*len(list_surf_gen)
            variance=np.var(fitness)

    return acc,X,y,fitness,threshold,variance

# TODOS LOS HEURISTICOS (ha esta funcion se le llamara desde la funcion fit (modificada por new_fit))
def execute_heuristic(heuristic,heuristic_param,train_seed,gen,population,init_acc,acc,X,y,fitness):
    global train_pts_seed
    global last_optimal_evaluations
    global acc_split
    global next_threshold
    global sample_size,heuristic_freq,last_time_heuristic_accepted
    global unused_bisection_executions

    threshold=None
    variance=None

    # Para el metodo de biseccion en los heuristicos finales (tamaño de muestra, frecuencia y expresion de interpolacion).
    df_sample_freq=pd.read_csv('results/data/general/sample_size_freq_'+str(sample_size_freq)+'.csv',index_col=0)
    df_interpolation=pd.read_csv('results/data/SymbolicRegressor/UnderstandingAccuracy/df_Bisection.csv')
    sample_size=int(df_sample_freq[df_sample_freq['env_name']=='SymbolicRegressor']['sample_size'])
    heuristic_freq=int(df_sample_freq[df_sample_freq['env_name']=='SymbolicRegressor']['frequency_time'])
    interpolation_acc=list(df_interpolation['accuracy'])
    interpolation_time=list(df_interpolation['cost_per_eval'])
    lower_time=min(interpolation_time)
    upper_time=max(interpolation_time)

 
    # Fijar accuracy de la generacion inicial.
    if gen==0:
        if heuristic in [1,2,3,4,5,9,11,12]:
            if heuristic==11:
                train_pts_seed=gen
            acc,X,y,fitness,acc_split,threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman')
        if heuristic in [6,7,8]:
            acc,X,y,fitness,acc_split,threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman',threshold=heuristic_param)
        if heuristic==10:
            acc,X,y,fitness,acc_split,threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'taukendall')
        if heuristic==13:
            acc,X,y,fitness,acc_split,threshold,next_threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman',threshold=heuristic_param[0],change_threshold='IncreasingMonotone')
        if heuristic==14:
            acc,X,y,fitness,acc_split,threshold,next_threshold,variance=set_initial_accuracy(init_acc,list(population),train_seed,'spearman',threshold=heuristic_param[0],change_threshold='NonMonotone')

        # Actualizar ultimo numero de evaluaciones en el que se a actualizado el accuracy para los heuristicos que lo necesiten.    
        if heuristic in [4,9,10]:
            last_optimal_evaluations=n_evaluations+n_evaluations_acc


        if heuristic in ['I','II']:
            acc,X,y,fitness,threshold,variance=customized_set_initial_accuracy(lower_time,upper_time,list(population),train_seed,sample_size,[interpolation_time,interpolation_acc],threshold=0.95)
            last_time_heuristic_accepted=n_evaluations+n_evaluations_acc
            if heuristic=='II':
                unused_bisection_executions=0
        


    # Actualizar accuracy en el resto de generaciones.
    else:
        if heuristic==1:
            acc,X,y,fitness=update_accuracy_heuristic1(acc,X,y,population,fitness,train_seed,heuristic_param)               
        if heuristic==2:
            acc,X,y,fitness=update_accuracy_heuristic2(acc,init_acc,X,y,population,fitness,train_seed,heuristic_param)
        if heuristic==3:
            acc,X,y,fitness=update_accuracy_heuristic3(acc,init_acc,X,y,population,fitness,heuristic_param)
        if heuristic==4:
            acc,X,y,fitness=update_accuracy_heuristic4(acc,X,y,population,fitness,heuristic_param)
        if heuristic==5:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness=update_accuracy_heuristic5(acc,X,y,list(df_seed[6]),population,fitness,heuristic_param)
        if heuristic==6:
            acc,X,y,fitness=update_accuracy_heuristic6(acc,list(population),train_seed,fitness,heuristic_param)
        if heuristic==7:
            acc,X,y,fitness=update_accuracy_heuristic7(acc,init_acc,list(population),train_seed,fitness,heuristic_param)
        if heuristic==8:
            acc,X,y,fitness=update_accuracy_heuristic8(acc,init_acc,list(population),train_seed,fitness,heuristic_param)
        if heuristic==9:
            acc,X,y,fitness=update_accuracy_heuristic9(acc,init_acc,X,y,list(population),train_seed,fitness,heuristic_param)
        if heuristic==10:
            acc,X,y,fitness=update_accuracy_heuristic10(acc,init_acc,X,y,list(population),train_seed,fitness,heuristic_param)
        if heuristic==11:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness=update_accuracy_heuristic11(init_acc,acc,X,y,list(df_seed[6]),population,train_seed,fitness,heuristic_param)
        if heuristic==12:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,variance=update_accuracy_heuristic12(acc,init_acc,X,y,list(df_seed[3]),list(population),train_seed,fitness,heuristic_param)
        if heuristic==13:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,next_threshold,variance=update_accuraccy_heuristic13(acc,init_acc,X,y,list(df_seed[3]),list(population),train_seed,fitness,next_threshold,heuristic_param[1])
        if heuristic==14:
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,next_threshold,variance=update_accuraccy_heuristic14(acc,init_acc,X,y,list(df_seed[3]),list(population),train_seed,fitness,next_threshold,heuristic_param[1])
    
        if heuristic=='I':
            acc,X,y,fitness=update_accuracy_heuristicI(acc,lower_time,upper_time,X,y,list(population),train_seed,fitness,heuristic_param,sample_size,heuristic_freq,[interpolation_time,interpolation_acc])
        if heuristic=='II':
            df_seed=pd.DataFrame(df_train)
            df_seed=df_seed[df_seed[1]==train_seed]
            acc,X,y,fitness,threshold,variance=update_accuracy_heuristicII(acc,lower_time,upper_time,X,y,list(df_seed[3]),list(population),train_seed,fitness,heuristic_param,sample_size,heuristic_freq,[interpolation_time,interpolation_acc])


    return acc,X,y,fitness,threshold,variance

#==================================================================================================
# FUNCIONES DISEÑADAS A PARTIR DE ALGUNAS YA EXISTENTES
#==================================================================================================

# FUNCION 
# -Original: raw_fitness
# -Script: _Program.py
# -Clase: _Program
def new_raw_fitness(self, X, y, sample_weight):
    
    y_pred = self.execute(X)
    if self.transformer:
        y_pred = self.transformer(y_pred)
    raw_fitness = self.metric(y, y_pred, sample_weight)
    
    # MODIFICACION: Sumar el numero de evaluaciones realizadas (tantas como puntos en el 
    # conjunto de entrenamiento).
    if count_evaluations:
        global n_evaluations
        n_evaluations+=X.shape[0]

    return raw_fitness

# FUNCION 
# -Original: _parallel_evolve
# -Script: genetic.py
def new_parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
   
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']

    max_samples = int(max_samples * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []
    i=0# MODIFICACION: inicializar contador de forma manual.
    while i<n_programs and n_evaluations<max_n_eval:#MODIFICACION: añadir nueva restriccion para terminar el bucle.

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        
        
        program.raw_fitness_=program.raw_fitness(X, y, curr_sample_weight)
         
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_= program.raw_fitness(X, y, oob_sample_weight)
            

        programs.append(program)

        i+=1# MODIFICACION: actualizar contador de forma manual.
    return programs

# FUNCION 
# Esta funcion contiene una parte del codigo interno de una funcion ya existente.
# -Original: fit
# -Script: genetic.py 
def find_best_individual_final_generation(self,fitness):

    if isinstance(self, TransformerMixin):
        # Find the best individuals in the final generation
        fitness = np.array(fitness)
        if self._metric.greater_is_better:
            hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
        else:
            hall_of_fame = fitness.argsort()[:self.hall_of_fame]
        evaluation = np.array([gp.execute(X) for gp in
                                [self._programs[-1][i] for
                                i in hall_of_fame]])
        if self.metric == 'spearman':
            evaluation = np.apply_along_axis(rankdata, 1, evaluation)

        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.abs(np.corrcoef(evaluation))
        np.fill_diagonal(correlations, 0.)
        components = list(range(self.hall_of_fame))
        indices = list(range(self.hall_of_fame))
        # Iteratively remove least fit individual of most correlated pair
        while len(components) > self.n_components:
            most_correlated = np.unravel_index(np.argmax(correlations),
                                                correlations.shape)
            # The correlation matrix is sorted by fitness, so identifying
            # the least fit of the pair is simply getting the higher index
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))
        self._best_programs = [self._programs[-1][i] for i in
                                hall_of_fame[components]]

    else:
        # Find the best individual in the final generation
        if self._metric.greater_is_better:
            self._program = self._programs[-1][np.argmax(fitness)]
        else:
            self._program = self._programs[-1][np.argmin(fitness)]

# FUNCION 
# -Original: fit
# -Script: genetic.py
# -Clase: BaseSymbolic
def new_fit(self,init_acc, X, y, train_seed,df_test_pts,heuristic,heuristic_param,sample_weight=None):# MODIFICACION: añadir nuevos argumentos.

    random_state = check_random_state(self.random_state)

    # Check arrays
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    if isinstance(self, ClassifierMixin):
        X, y = self._validate_data(X, y, y_numeric=False)
        check_classification_targets(y)

        if self.class_weight:
            if sample_weight is None:
                sample_weight = 1.
            # modify the sample weights with the corresponding class weight
            sample_weight = (sample_weight *
                                compute_sample_weight(self.class_weight, y))

        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes != 2:
            raise ValueError("y contains %d class after sample_weight "
                                "trimmed classes with zero weights, while 2 "
                                "classes are required."
                                % n_trim_classes)
        self.n_classes_ = len(self.classes_)

    else:
        X, y = self._validate_data(X, y, y_numeric=True)

    hall_of_fame = self.hall_of_fame
    if hall_of_fame is None:
        hall_of_fame = self.population_size
    if hall_of_fame > self.population_size or hall_of_fame < 1:
        raise ValueError('hall_of_fame (%d) must be less than or equal to '
                            'population_size (%d).' % (self.hall_of_fame,
                                                    self.population_size))
    n_components = self.n_components
    if n_components is None:
        n_components = hall_of_fame
    if n_components > hall_of_fame or n_components < 1:
        raise ValueError('n_components (%d) must be less than or equal to '
                            'hall_of_fame (%d).' % (self.n_components,
                                                    self.hall_of_fame))

    self._function_set = []
    for function in self.function_set:
        if isinstance(function, str):
            if function not in _function_map:
                raise ValueError('invalid function name %s found in '
                                    '`function_set`.' % function)
            self._function_set.append(_function_map[function])
        elif isinstance(function, _Function):
            self._function_set.append(function)
        else:
            raise ValueError('invalid type %s found in `function_set`.'
                                % type(function))
    if not self._function_set:
        raise ValueError('No valid functions found in `function_set`.')

    # For point-mutation to find a compatible replacement node
    self._arities = {}
    for function in self._function_set:
        arity = function.arity
        self._arities[arity] = self._arities.get(arity, [])
        self._arities[arity].append(function)

    if isinstance(self.metric, _Fitness):
        self._metric = self.metric
    elif isinstance(self, RegressorMixin):
        if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                'spearman', 'spearman'):
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]
    elif isinstance(self, ClassifierMixin):
        if self.metric != 'log loss':
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]
    elif isinstance(self, TransformerMixin):
        if self.metric not in ('spearman', 'spearman'):
            raise ValueError('Unsupported metric: %s' % self.metric)
        self._metric = _fitness_map[self.metric]

    self._method_probs = np.array([self.p_crossover,
                                    self.p_subtree_mutation,
                                    self.p_hoist_mutation,
                                    self.p_point_mutation])
    self._method_probs = np.cumsum(self._method_probs)

    if self._method_probs[-1] > 1:
        raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                            'p_hoist_mutation and p_point_mutation should '
                            'total to 1.0 or less.')

    if self.init_method not in ('half and half', 'grow', 'full'):
        raise ValueError('Valid program initializations methods include '
                            '"grow", "full" and "half and half". Given %s.'
                            % self.init_method)

    if not((isinstance(self.const_range, tuple) and
            len(self.const_range) == 2) or self.const_range is None):
        raise ValueError('const_range should be a tuple with length two, '
                            'or None.')

    if (not isinstance(self.init_depth, tuple) or
            len(self.init_depth) != 2):
        raise ValueError('init_depth should be a tuple with length two.')
    if self.init_depth[0] > self.init_depth[1]:
        raise ValueError('init_depth should be in increasing numerical '
                            'order: (min_depth, max_depth).')

    if self.feature_names is not None:
        if self.n_features_in_ != len(self.feature_names):
            raise ValueError('The supplied `feature_names` has different '
                                'length to n_features. Expected %d, got %d.'
                                % (self.n_features_in_,
                                len(self.feature_names)))
        for feature_name in self.feature_names:
            if not isinstance(feature_name, str):
                raise ValueError('invalid type %s found in '
                                    '`feature_names`.' % type(feature_name))

    if self.transformer is not None:
        if isinstance(self.transformer, _Function):
            self._transformer = self.transformer
        elif self.transformer == 'sigmoid':
            self._transformer = sigmoid
        else:
            raise ValueError('Invalid `transformer`. Expected either '
                                '"sigmoid" or _Function object, got %s' %
                                type(self.transformer))
        if self._transformer.arity != 1:
            raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                'got %d.' % (self._transformer.arity))

    params = self.get_params()
    params['_metric'] = self._metric
    if hasattr(self, '_transformer'):
        params['_transformer'] = self._transformer
    else:
        params['_transformer'] = None
    params['function_set'] = self._function_set
    params['arities'] = self._arities
    params['method_probs'] = self._method_probs

    if not self.warm_start or not hasattr(self, '_programs'):
        # Free allocated memory, if any
        self._programs = []
        self.run_details_ = {'generation': [],
                                'average_length': [],
                                'average_fitness': [],
                                'best_length': [],
                                'best_fitness': [],
                                'best_oob_fitness': [],
                                'generation_time': []}

    prior_generations = len(self._programs)
    n_more_generations = self.generations - prior_generations

    if n_more_generations < 0:
        raise ValueError('generations=%d must be larger or equal to '
                            'len(_programs)=%d when warm_start==True'
                            % (self.generations, len(self._programs)))
    elif n_more_generations == 0:
        fitness = [program.raw_fitness_ for program in self._programs[-1]]
        warn('Warm-start fitting without increasing n_estimators does not '
                'fit new programs.')

    if self.warm_start:
        # Generate and discard seeds that would have been produced on the
        # initial fit call.
        for i in range(len(self._programs)):
            _ = random_state.randint(MAX_INT, size=self.population_size)

    if self.verbose:
        # Print header fields
        self._verbose_reporter()

    start_total_time=time() #MODIFICACION: empezar a contar el tiempo de entrenamiento.
    gen=0# MODIFICACION: para que el procedimiento no termine cuando se alcance un numero de generaciones, las generaciones se cuentan con un contador independiente.

    # MODIFICACION: variable global mediante la cual se iran contando el numero de evaluaciones realizadas,
    # entendiendo por evaluacion cada evaluacion de un punto en una expresion de una superficie.
    global n_evaluations
    n_evaluations=0
    global n_evaluations_acc
    n_evaluations_acc=0
    acc=init_acc
    global count_evaluations
    count_evaluations=False# Para que en la primera generacion no se cuenten las evaluaciones hechas al calcular la poblacion.
    
    global train_pts_seed
    while n_evaluations+n_evaluations_acc < max_n_eval:# MODIFICACION: modificar el limite de entrenamiento.
        
        start_time = time()

        if gen == 0:
            parents = None
        else:
            parents = self._programs[gen - 1]

        # Parallel loop
        n_jobs, n_programs, starts = _partition_estimators(
            self.population_size, self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.population_size)

        # MODIFICACION: contar las evaluaciones hechas al definir la poblacion.
        if heuristic=='None':
            threshold=None
            variance=None
            count_evaluations=True

        population = Parallel(n_jobs=n_jobs,
                                verbose=int(self.verbose > 1))(
            delayed(_parallel_evolve)(n_programs[i],
                                        parents,
                                        X,
                                        y,
                                        sample_weight,
                                        seeds[starts[i]:starts[i + 1]],
                                        params)
            for i in range(n_jobs))

        # Reduce, maintaining order across different n_jobs
        population = list(itertools.chain.from_iterable(population))
        fitness = [program.raw_fitness_ for program in population]

        # MODIFICACION: cuando se aplique algun heuristico.
        if heuristic !='None':
            acc,X,y,fitness,threshold,variance=execute_heuristic(heuristic,heuristic_param,train_seed,gen,population,init_acc,acc,X,y,fitness)

        length = [program.length_ for program in population]

        parsimony_coefficient = None
        if self.parsimony_coefficient == 'auto':
            parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                        np.var(length))
        for program in population:
            program.fitness_ = program.fitness(parsimony_coefficient)

        self._programs.append(population)

        # Remove old programs that didn't make it into the new population.
        if not self.low_memory:
            for old_gen in np.arange(gen, 0, -1):
                indices = []
                for program in self._programs[old_gen]:
                    if program is not None:
                        for idx in program.parents:
                            if 'idx' in idx:
                                indices.append(program.parents[idx])
                indices = set(indices)
                for idx in range(self.population_size):
                    if idx not in indices:
                        self._programs[old_gen - 1][idx] = None
        elif gen > 0:
            # Remove old generations
            self._programs[gen - 1] = None

        # Record run details
        if self._metric.greater_is_better:
            best_program = population[np.argmax(fitness)]
        else:
            best_program = population[np.argmin(fitness)]

        self.run_details_['generation'].append(gen)
        self.run_details_['average_length'].append(np.mean(length))
        self.run_details_['average_fitness'].append(np.mean(fitness))
        self.run_details_['best_length'].append(best_program.length_)
        self.run_details_['best_fitness'].append(best_program.raw_fitness_)
        oob_fitness = np.nan
        if self.max_samples < 1.0:
            oob_fitness = best_program.oob_fitness_
        self.run_details_['best_oob_fitness'].append(oob_fitness)
        generation_time = time() - start_time
        self.run_details_['generation_time'].append(generation_time)

        if self.verbose:
            self._verbose_reporter(self.run_details_)

        # Check for early stopping
        if self._metric.greater_is_better:
            best_fitness = fitness[np.argmax(fitness)]
        else:
            best_fitness = fitness[np.argmin(fitness)]

   
        find_best_individual_final_generation(self,fitness) # MODIFICACION: para poder evaluar la mejor superficie durante el proceso.
        
        # MODIFICACION: ir guardando los datos de interes durante el entrenamiento. 
        score=evaluate(df_test_pts,self)
        elapsed_time=time()-start_total_time   
        df_train.append([heuristic_param,train_seed,threshold,variance,acc,gen,score,elapsed_time,generation_time,n_evaluations,n_evaluations_acc,n_evaluations+n_evaluations_acc])

        # print('n_eval_PROC: '+str(n_evaluations)+'n_evaluations_acc: '+str(n_evaluations_acc)+' acc:'+str(acc))   
        gen+=1# MODIFICACION: actualizar numero de generaciones.

    find_best_individual_final_generation(self,fitness)# MODIFICACION: para obtener el mejor individuo de la ultima generacion.
    
    return self

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Para usar la funcion de ajuste modificada.
_Program.raw_fitness=new_raw_fitness
_parallel_evolve=new_parallel_evolve
BaseSymbolic.fit=new_fit

# Superficie.
expr_surf_real='x**2-y**2+y-1'

# Mallados.
list_train_seeds=range(1,101,1)# Semillas de entrenamiento.

# Parametros y conjunto de entrenamiento.
default_train_n_pts=50# Cardinal de conjunto inicial predefinido.
train_pts_seed=0
default_df_train_pts=build_pts_sample(default_train_n_pts,train_pts_seed,expr_surf_real)
max_n_eval=20*50*1000# Equivalente a 20 generaciones con el maximo accuracy.

# Parametros y conjunto de validacion.
test_n_pts=default_train_n_pts
test_pts_seed=1
df_test_pts=build_pts_sample(test_n_pts,test_pts_seed,expr_surf_real)

# Otros parametros.
sample_size_freq='BisectionAndPopulation'
# sample_size_freq='BisectionOnly'

# Funcion para realizar la ejecucion en paralelo.
def parallel_processing(arg):

    # Extraer informacion de heuristico del argumento de la funcion.
    heuristic=arg[0]
    param=arg[1]

    # Guardar datos de entrenamiento.
    global df_train
    df_train=[]

    # Accuracy inicial.
    if heuristic=='None':
        init_acc=1
    else:
        init_acc=1/default_train_n_pts# El correspondiente a un conjunto formado por un unico punto.


    for train_seed in tqdm(list_train_seeds):
        #Entrenamiento.
        learn(init_acc,train_seed,df_test_pts,heuristic,param)     

    # Guardar base de datos construida.
    df_train=pd.DataFrame(df_train,columns=['heuristic_param','train_seed','threshold','variance','acc','n_gen','score','elapsed_time','time_gen','n_eval_proc','n_eval_acc','n_eval'])

    if heuristic=='None':
        df_train.to_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_ConstantAccuracy1.csv')
    else:
        df_train.to_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(heuristic)+'_param'+str(param)+'.csv')

    # Guardar expresion de superficie.
    np.save('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/expr_surf',expr_surf_real)

    

# Preparar argumentos de la funcion para procesamiento en paralelo.
# Forma de argumento: [identificador de heuristico, valor de parametro]
list_arg=[
    # [1,(0,1)],[1,(0,3)],[1,(0.5,3)],[1,(0.5,1)],[1,(0,0.3)],[1,'logistic'],
    # [2,0.8],[2,0.6],
    # [3,0.8],[3,0.6],
    # [4,100000],[4,500000],[4,50000],
    # [5,10],[5,20],
    # [6,0.95],[6,0.8],
    # [7,0.95],[7,0.8],
    # [8,0.95],[8,0.8],
    # [9,100000],[9,500000],
    # [10,100000],[10,500000],
    # [11,5],[11,10],
    # [12,5],[12,10],
    # [13,(0.5,5)],[13,(0.5,10)],[13,(0.95,5)],[13,(0.95,10)],[13,(0.8,5)],[13,(0.8,10)],
    # [14,(0.5,5)],[14,(0.5,10)],[14,(0.95,5)],[14,(0.95,10)],[14,(0.8,5)],[14,(0.8,10)],
    # ['None',''],
    # ['I',0.95],['I',0.8],['II',5],['II',10]
    ['II',5],['II',10]

    ]

# Procesamiento en paralelo.
pool=mp.Pool(mp.cpu_count())
pool.map(parallel_processing,list_arg)
pool.close()

# Agrupar bases de datos.
def concat_same_heuristic_df(list_arg):
    heuristic_param_dict={}
    for arg in list_arg:
        heuristic=arg[0]
        parameter=arg[1]
        if heuristic not in heuristic_param_dict:
            heuristic_param_dict[heuristic]=[parameter]
        else:
            heuristic_param_dict[heuristic].append(parameter)

    dict_keys=list(heuristic_param_dict.keys())
    # dict_keys.remove('None')

    for key in dict_keys:
        list_param=heuristic_param_dict[key]
        first=True
        for param in list_param:

            if first:
                df=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv')
                first=False
            else:
                df_new=pd.read_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv', index_col=0)
                os.remove('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_param'+str(param)+'.csv')
                df=pd.concat([df,df_new],ignore_index=True)


        df.to_csv('results/data/SymbolicRegressor/OptimalAccuracyAnalysis/df_train_OptimalAccuracy_heuristic'+str(key)+'_'+str(sample_size_freq)+'.csv')


concat_same_heuristic_df(list_arg)



