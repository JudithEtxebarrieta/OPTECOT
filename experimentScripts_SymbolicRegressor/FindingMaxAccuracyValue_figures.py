'''
This script is used to graphically represent the numerical results obtained in 
"FindingMaxAccuracyValue.py".
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import pandas as pd
import tqdm

#==================================================================================================
# FUNCTIONS
#==================================================================================================

def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
    '''
    The 95% confidence interval of a given data sample is calculated.

    Parameters
    ==========
    data (list): Data on which the range between percentiles will be calculated.
    bootstrap_iterations (int): Number of subsamples of data to be considered to calculate the percentiles of their means. 

    Return
    ======
    The mean of the original data together with the percentiles of the means obtained from the subsampling of the data. 
    '''
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True)
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def from_data_to_figure(df,position_mae):
    '''
    Construct graph from stored data.

    Parameters
    ==========
    df: Database with information on the scores associated with different sets of points of different sizes.
    position_mae: Numerical code with the position where you want to plot the graph of the mean absolute errors.
    '''

    # Initializations.
    all_mean_mae=[]
    all_q05_mae=[]
    all_q95_mae=[]

    # list of seeds. 
    list_train_seeds=list(set(df['seed']))
    list_train_n_pts=list(set(df['n_pts']))

    # Fill in the initialized lists.
    for n_pts in list_train_n_pts:

        # Select the data of all the seeds associated with the number of points set.
        scores_mae=df[df['n_pts']==n_pts]['score_mae']

        # Calculate confidence interval and mean.
        mean_mae,q05_mae,q95_mae=bootstrap_mean_and_confidence_interval(scores_mae)

        # Accumulate data.
        all_mean_mae.append(mean_mae)
        all_q05_mae.append(q05_mae)
        all_q95_mae.append(q95_mae)

    # Draw graph.
    ax1=plt.subplot(position_mae)
    ax1.fill_between(list_train_n_pts,all_q05_mae,all_q95_mae, alpha=.5, linewidth=0)
    plt.plot(list_train_n_pts,all_mean_mae)
    ax1.set_xlabel("Size of train point set")
    ax1.set_ylabel("Mean MAE("+str(len(list_train_seeds))+' seeds)')
    ax1.set_title('Behavior of the MAE of the optimal solution\n depending on the size of the train point set')

def draw_surface(position,eval_expr):
    
    '''Draw surface and place the obtained graph in the indicated position.'''

    # Calculate coordinates of the points to be drawn.
    x = np.arange(-1, 1, 1/10.)
    y = np.arange(-1, 1, 1/10.)
    x, y= np.meshgrid(x, y)
    z = eval(eval_expr)

    # Draw all the points.
    ax = plt.subplot(position,projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.plot_surface(x, y, z, rstride=1, cstride=1,color='green', alpha=0.5)
    ax.set_title('Real surface: '+eval_expr)    

#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# EXAMPLE 1 (Hyperbolic paraboloid)
# Solution: the size of the set of points at which the MAE begins to converge is 30.
#--------------------------------------------------------------------------------------------------
# Read data.
df=pd.read_csv('results/data/SymbolicRegressor/FindingMaxAccuracyValue/FindingMaxAccuracyValue1.csv',index_col=0)
eval_expr=str(np.load('results/data/SymbolicRegressor/FindingMaxAccuracyValue/eval_expr1.npy'))

# Build graph.
plt.figure(figsize=[10,5])
plt.subplots_adjust(left=0.07,bottom=0.11,right=0.94,top=0.88,wspace=0.2,hspace=0.2)

from_data_to_figure(df,121)
draw_surface(122,eval_expr)

plt.savefig('results/figures/SymbolicRegressor/FindingMaxAccuracyValue/FindingAccuracyConvergenceSURF1.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# EXAMPLE 2 (Plane)
# Solution: the size of the set of points at which the MAE begins to converge is 6.
#--------------------------------------------------------------------------------------------------

# Read data.
df=pd.read_csv('results/data/SymbolicRegressor/FindingMaxAccuracyValue/FindingMaxAccuracyValue2.csv',index_col=0)
eval_expr=str(np.load('results/data/SymbolicRegressor/FindingMaxAccuracyValue/eval_expr2.npy'))

# Build graph.
plt.figure(figsize=[10,5])
plt.subplots_adjust(left=0.07,bottom=0.11,right=0.94,top=0.88,wspace=0.2,hspace=0.2)

from_data_to_figure(df,121)
draw_surface(122,eval_expr)

plt.savefig('results/figures/SymbolicRegressor/FindingMaxAccuracyValue/FindingAccuracyConvergenceSURF2.png')
plt.show()
plt.close()
