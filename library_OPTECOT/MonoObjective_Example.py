#==================================================================================================
# LIBRERIAS
#==================================================================================================
import pandas as pd
import numpy as np
from termcolor import colored
from tqdm import tqdm
from MonoObjective_OPTECOT import OPTECOT, ExperimentalGraphs

import os
import sys
sys.path.append("OptimizationAlgorithms_KONFLOT/packages")

import turbine_classes
import MathTools as mt

#==================================================================================================
# DATOS Y REQUISITOS NECESARIOS
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Parametros
#--------------------------------------------------------------------------------------------------
# Obligatorio definir.
popsize=20 # Tamaño de poblacion.
theta1=100 # Valor original del parametro del que depende la funcion objetivo y con el que controlamos su coste.
theta0=10 # Valor del parametro del que depende la funcion objetivo al que se le asocia el menor coste.
xdim=6 # Dimention of x (una solucion).
max_time=60*60 # Tiempo maximo de ejecucion disponible en segundos(1h per seed).  
objective_min=False

# Vienen definidos por defecto pero se pueden modificar.
alpha=0.95 # Umbral de precision de la aproximacion.
beta=5 # Numero de varianzas consideradas para calcular el intervalo de confianza.
kappa=3 # Numero de accuracies previos a comparar para valorar la interrupcion del heuristico.
min_sample_size=10 # Minimo tamaño de muestra de la poblacion.
perc_cost=0.25 # Porcentaje del tiempo total que se empleara en el peor caso para reajustar el coste.

#--------------------------------------------------------------------------------------------------
# Funcion objetivo
#--------------------------------------------------------------------------------------------------
'''
Objective function (takes a solution and parameter value to give its score value)

Inputs: solution, theta
Ouput: solution score

Note: remind that the score asociated with objective function value must match with the optimization algorithm optimization
direction (maximization or minimization)
'''
def fitness_function(turb_params,theta=100):

    '''Evaluating a turbine design.'''

    # Build dictionary of constant parameters.
    def build_constargs_dict(N):
        # Define constant parameters.
        omega = 2100# Rotational speed.
        rcas = 0.4# Casing radius.
        airfoils = ["NACA0015", "NACA0018", "NACA0021"]# Set of possible airfoils.
        polars = turbine_classes.polar_database_load(filepath="OptimizationAlgorithms_KONFLOT/", pick=False)# Polars.
        cpobjs = [933.78, 1089.41, 1089.41, 1011.59, 1011.59, 1011.59, 933.78, 933.78, 933.78, 855.96]# Target dumping coefficients.
        devobjs = [2170.82, 2851.59, 2931.97, 2781.80, 2542.296783, 4518.520988, 4087.436172, 3806.379812, 5845.986619, 6745.134759]# Input sea-state standard pressure deviations.
        weights = [0.1085, 0.1160, 0.1188, 0.0910, 0.0824, 0.1486, 0.0882, 0.0867, 0.0945, 0.0652]# Input sea-state weights.
        Nmin = 1000#Max threshold rotational speeds
        Nmax = 3200#Min threshold rotational speeds

        # Construct the dictionary needed by the fitness function.
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

    constargs=build_constargs_dict(N=theta)

    # Create instantaneous turbine.
    os.chdir('OptimizationAlgorithms_KONFLOT')
    turb = turbine_classes.instantiate_turbine(constargs, turb_params)	
    os.chdir('../')

    # Calculate evaluation.
    scores = turbine_classes.fitness_func(constargs=constargs, turb=turb, out='brfitness')

    return scores[1]
#--------------------------------------------------------------------------------------------------
# Generador de soluciones aleatorias (funcion que permite generar un conjunto de posibles soluciones aleatorias).
#--------------------------------------------------------------------------------------------------
def generate_random_configurations(n_sample=100):   
    '''Create a random set of turbine designs.'''
        
    # List of seeds.
    list_seeds=range(0,n_sample)

    # Define ranges of parameters defining the turbine design.
    blade_number = [3,5,7]# Blade-number gene.
    sigma_hub = [0.4, 0.7]# Hub solidity gene.
    sigma_tip = [0.4, 0.7]# Tip solidity gene.
    nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
    tip_clearance=[0,3]# Tip-clearance gene.	  
    airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

    # Save as many random configurations as seeds in list_seeds.
    list_turb_params=[]
    for seed in list_seeds: 
        np.random.seed(seed)
        if type(blade_number)==list:
            select_blade_number=np.random.choice(blade_number)
        if type(blade_number)==int:
            select_blade_number=blade_number
        turb_params = [select_blade_number,
                np.random.uniform(sigma_hub[0], sigma_hub[1]),
                np.random.uniform(sigma_tip[0], sigma_tip[1]),
                np.random.uniform(nu[0], nu[1]),
                np.random.uniform(tip_clearance[0], tip_clearance[1]),
                np.random.choice(airfoil_dist)]
        list_turb_params.append(turb_params)

    return list_turb_params

#--------------------------------------------------------------------------------------------------
# Transformador de soluciones escaladas.
#--------------------------------------------------------------------------------------------------
def transform_scaled_solution(scaled_x):
    '''Transform the scaled values of a solution to the real values.'''

    # Set the ranges of the parameters defining the turbine design.
    blade_number = [3,5,7]# Blade-number gene.
    sigma_hub = [0.4, 0.7]# Hub solidity gene.
    sigma_tip = [0.4, 0.7]# Tip solidity gene.
    nu = [0.4, 0.75] # Hub-to-tip-ratio gene.
    tip_clearance=[0,3]# Tip-clearance gene.	  
    airfoil_dist = np.arange(0, 27)# Airfoil dist. gene.  

    # List with ranges.
    bounds=np.array([
    [sigma_hub[0]    , sigma_hub[1]],
    [sigma_tip[0]    , sigma_tip[1]],
    [nu[0]           , nu[1]],
    [tip_clearance[0], tip_clearance[1]],
    [0               , 26]
    ])

    # To transform the discrete parameter blade-number.
    def blade_number_transform(posible_blade_numbers,scaled_blade_number):
        discretization=np.arange(0,1+1/len(posible_blade_numbers),1/len(posible_blade_numbers))
        detection_list=discretization>scaled_blade_number
        return posible_blade_numbers[list(detection_list).index(True)-1]

    # Transformation.
    real_x = scaled_x[1:] * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    real_bladenumber= blade_number_transform(blade_number,scaled_x[0])

    return [real_bladenumber]+list(real_x[:-1])+[round(real_x[-1])]

#==================================================================================================
# USO DE LA LIBRERIA
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Inicializar OPTECOT y las graficas con los requerimientos definidos.
#--------------------------------------------------------------------------------------------------
optecot=OPTECOT(popsize=popsize,
                xdim=xdim,
                max_time=max_time,
                theta0=theta0,
                theta1=theta1,
                objective_min=False,
                objective_function=fitness_function,
                scaled_solution_transformer=transform_scaled_solution,
                set_solutions=generate_random_configurations(),
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                min_sample_size=min_sample_size,
                perc_cost=perc_cost#,
                #customized_df_acc_time=pd.read_csv('library_OPTECOT/results/auxiliary_data/df_acc_time.csv',index_col=0),
                #list_costs=[1.0,0.78,0.56,0.33,0.22,0]
                )

# !!!!!!ELIMINAR LOS ULTIMOS DOS ARGUMENTOS DE LA CLASE (solo era para probar que funciona el codigo)

#--------------------------------------------------------------------------------------------------
# Ejecutar RBEA usando funciones objetivo aproximadas.
#--------------------------------------------------------------------------------------------------
# Ejecutar CMA-ES usando diferentes semillas y aproximaciones de diferentes costes.
optecot.execute_CMAES_with_approximations(2,[1.0,0.78,0.56,0.33,0.22,0])

# Dibujar resultados.
ExperimentalGraphs.illustrate_approximate_objective_functions_use(optecot)   

#--------------------------------------------------------------------------------------------------
# Ejecutar RBEA usando OPTECOT.  
#--------------------------------------------------------------------------------------------------
# Ejecutar CMA-ES usando diferentes semillas y OPTECOT.
optecot.execute_CMAES_with_OPTECOT(100)

# Dibujar resultados.
ExperimentalGraphs.illustrate_OPTECOT_application_results(optecot)








