import os
import sys
import numpy as np


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'AirfoilOpt'))
os.chdir('AirfoilOpt')

import optimize_parsec_ga
from parsec.synthesis import synthesize

n_eval = 1000

# Airfoil parameters
n_points = 192

# NACA 0012 as the original airfoil
x0 = np.array([0.0147, 0.2996, -0.06, 0.4406, 7.335, 0.3015, 0.0599, -0.4360, -7.335]) # NACA 0012

perturb_type = 'relative'
perturb = 0.2
syn_func = lambda x: synthesize(x, n_points)

n_iter = 20
opt_airfoil, opt_perfs = optimize_parsec_ga.optimize(x0, syn_func, perturb_type, perturb, n_eval, 1, n_points, n_iter)


n_iter = 200
opt_airfoil, opt_perfs = optimize_parsec_ga.optimize(x0, syn_func, perturb_type, perturb, n_eval, 1, n_points, n_iter)


