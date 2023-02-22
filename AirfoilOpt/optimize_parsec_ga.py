"""
Optimize the airfoil shape directly using genetic algorithm, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s):
    Viswanath, A., J. Forrester, A. I., Keane, A. J. (2011). Dimension Reduction for Aerodynamic Design Optimization.
    AIAA Journal, 49(6), 1256-1266.
    Grey, Z. J., Constantine, P. G. (2018). Active subspaces of airfoil shape parameterizations.
    AIAA Journal, 56(5), 2003-2017.
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from genetic_alg import generate_first_population, select, create_children, mutate_population
from parsec.synthesis import synthesize
from utils import mean_err



def optimize(x0, syn_func, perturb_type, perturb, n_eval, run_id, n_points, n_iter):
    # Optimize using GA
    n_best = 30
    n_random = 10
    n_children = 5
    chance_of_mutation = 0.1
    population_size = int((n_best+n_random)/2*n_children)
    population = generate_first_population(x0, population_size, perturb_type, perturb)
    best_inds = []
    best_perfs = []
    opt_perfs = [0]
    i = 0
    while 1:
        breeders, best_perf, best_individual = select(population, n_best, n_random, syn_func, n_iter)
        best_inds.append(best_individual)
        best_perfs.append(best_perf)
        opt_perfs += [np.max(best_perfs)] * population_size # Best performance so far
        print('PARSEC-GA %d-%d: fittest %.2f' % (run_id, i+1, best_perf))
        # No need to create next generation for the last generation
        if i < n_eval/population_size-1:
            next_generation = create_children(breeders, n_children)
            population = mutate_population(next_generation, chance_of_mutation, perturb_type, perturb)
            i += 1
        else:
            break
    
    opt_x = best_inds[np.argmax(best_perfs)]
    opt_airfoil = synthesize(opt_x, n_points)
    print('Optimal CL/CD: {}'.format(opt_perfs[-1]))
    
    return opt_airfoil, opt_perfs
