'''
This script graphically represents two interpolations between the evaluation times and their 
accuracy values: one is polynomial and the other is linear. The polynomial experiments are 
stored in a database.
'''

#==================================================================================================
# LIBRARIES
#==================================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==================================================================================================
# FUNCTIONS
#==================================================================================================
def polinomial_interpolation(x,y,var_name='m'):
    '''
    Obtain the polynomial expression that interpolates a set of points (the degree 
    of the polynomial is equivalent to the cardinal of the set of points).
    '''
    vandermonde=np.vander(x,increasing=True)
    coef=np.dot(np.linalg.inv(vandermonde),y)
    expression=''
    for i in range(len(x)):
        sign=''
        if coef[i]>=0:
            sign='+'        
        expression+=sign+str(coef[i])+'*'+str(var_name)+'**'+str(i)

    return  expression

def draw_expression(expression,x,y,env_name):
    '''Graph the polynomial that interpolates a set of points.'''
    expression_x=np.arange(min(x),max(x)+(max(x)-min(x))/100,(max(x)-min(x))/100)
    expression_y=[]
    for m in expression_x:
        expression_y.append(eval(expression))
    
    plt.figure(figsize=[6,6])
    ax=plt.subplot(111)
    plt.plot(expression_x,expression_y,color='blue',label='Interpolation')
    plt.scatter(x,y,color='red',marker='x',label='Interpolation points')
    plt.title(str(env_name))
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Evaluation cost')
    ax.legend()
    plt.savefig('results/figures/general/PolinomioalInterpolation_'+str(env_name)+'.png')
    plt.show()

def draw_linear_interpolation(x,y,env_name):
    '''Graph the polygonal line that interpolates a set of points.'''
    plt.figure(figsize=[6,6])
    ax=plt.subplot(111)
    plt.plot(x,y,color='blue',label='Interpolation')
    plt.scatter(x,y,color='red',marker='x',label='Interpolation points')
    plt.title(str(env_name))
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Evaluation cost')
    ax.legend()
    plt.savefig('results/figures/general/LinearInterpolation_'+str(env_name)+'.png')
    plt.show()


#==================================================================================================
# MAIN PROGRAM
#==================================================================================================
list_env_names=['SymbolicRegressor','WindFLO','MuJoCo','Turbines']
df=[]

# Polynomial interpolation.
for env_name in list_env_names:
    df_acc_eval_cost=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_Bisection.csv')
    expression=polinomial_interpolation(df_acc_eval_cost['cost_per_eval'],df_acc_eval_cost['accuracy'])
    inverse_expression=polinomial_interpolation(df_acc_eval_cost['accuracy'],df_acc_eval_cost['cost_per_eval'],var_name='m_inv')
    draw_expression(expression,df_acc_eval_cost['cost_per_eval'],df_acc_eval_cost['accuracy'],env_name)
    df.append([env_name,expression,inverse_expression,min(df_acc_eval_cost['cost_per_eval']),max(df_acc_eval_cost['cost_per_eval'])])

df=pd.DataFrame(df,columns=['env_name','interpolation_expression','inverse_expression','lower_time','upper_time'])
df.to_csv('results/data/general/IntervalInterpolation_bisection_method.csv')

# Linear interpolation.
for env_name in list_env_names:
    df_acc_eval_cost=pd.read_csv('results/data/'+str(env_name)+'/UnderstandingAccuracy/df_Bisection.csv')
    draw_linear_interpolation(df_acc_eval_cost['cost_per_eval'],df_acc_eval_cost['accuracy'],env_name)
