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
import math

#==================================================================================================
# FUNCIONES
#==================================================================================================

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

def common_n_gen_per_seed(blade_number,list_seeds):
    n_gen_per_seed=math.inf

    for seed in list_seeds:
        df=pd.read_csv('results/data/Turbines/CMA_ES_GenerationAnalysis/df_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv',index_col=0)
        max_n_gen=int(df['n_gen'].max())

        if max_n_gen<n_gen_per_seed:
            n_gen_per_seed=max_n_gen

    return n_gen_per_seed

def from_argsort_to_ranking(list):
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

def ranking_matrix(df,gen):

    # Reducir df a filas de interés.
    df=df[df['n_gen']==gen]

    # Guardar rankings de cada N en una matriz por filas.
    matrix=[]
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)
 
    for N in list_N:
        df_N=df[df['N']==N]
        list_scores=df_N.sort_values('n_eval')['score']
        ranking=from_argsort_to_ranking(np.argsort(-np.array(list_scores)))# Orden descendente.
        matrix.append(ranking)

    # Reordenar filas de la matriz para que la fila de la máxima precisión represente un ranking 
    # ordenado de forma perfecta (de mejor a peor).
    matrix_sort=[]
    ind_sort=np.argsort(matrix[0])
    for row in range(len(matrix)):

        new_row=[]
        for i in ind_sort:
            new_row.append(matrix[row][i])

        matrix_sort.append(new_row)
    return matrix_sort


def draw_rankings_per_generation(size,blade_number,seed):

    # Cargar base de datos.
    df=pd.read_csv('results/data/Turbines/CMA_ES_GenerationAnalysis/df_blade_number'+str(blade_number)+'_seed'+str(seed)+'.csv',index_col=0)

    # Número de evaluaciones total.
    total_n_eval=df['n_eval'].max()

    # Valores de N considerados.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Por cada generación dibujar la matriz de rankings.
    for gen in range(1,size[1]+1):
        ax=plt.subplot(size[0],size[1],gen+seed*size[1])

        # Crear matriz de rankings.
        matrix=np.matrix(ranking_matrix(df,gen))

        # Dibujar la matriz
        color = sns.color_palette("deep",total_n_eval )
        color = ListedColormap(color)

        ax = sns.heatmap(matrix, cmap=color,linewidths=.5, linecolor='lightgray',cbar=False)
        plt.xticks([])

        # if seed==size[0]-1:
        #     ax.set_xlabel('Generation '+str(gen))
        if seed==size[0]-2:
            ax.set_xlabel('Generation '+str(gen))

        if gen ==1:
            ax.set_ylabel('Seed '+str(seed)+'\n N')
            ax.set_yticks(np.arange(.5,matrix.shape[0]+.5))
            ax.set_yticklabels(list_N, rotation=0)

        if gen !=1:
            plt.yticks([])
    return color
    
def pearson_corr(x,y):
    return sc.stats.pearsonr(x,y)[0]

def join_df(blade_number,list_seeds):

    # Primera base de datos.
    df=pd.read_csv('results/data/Turbines/CMA_ES_GenerationAnalysis/df_blade_number'+str(blade_number)+'_seed'+str(list_seeds[0])+'.csv',index_col=0)
    
    # Las demás.
    for seed in list_seeds[1:]:
        new_df=pd.read_csv('results/data/Turbines/CMA_ES_GenerationAnalysis/df_blade_number'+str(blade_number)+'_seed'+str(list_seeds[seed])+'.csv',index_col=0)
        df=pd.concat([df,new_df])    
    return df

def score_matrix(df_seed,gen):

     # Reducir df a filas de interés.
    df_seed_gen=df_seed[df_seed['n_gen']==gen]

    # Guardar scores de cada N en una matriz por filas.
    matrix=[]
    list_N=list(set(df_seed_gen['N']))
    list_N.sort(reverse=True)

    for N in list_N:
        df_N=df_seed_gen[df_seed_gen['N']==N]
        list_scores=df_N.sort_values('n_eval')['score']
        matrix.append(list(list_scores))

    return matrix


def draw_ranking_comparison_per_gen(blade_number,list_seeds,type,what):

    # Juntar todas las bases de datos en una.
    df=join_df(blade_number,list_seeds)

    # Valores de N considerados.
    list_N=list(set(df['N']))
    list_N.sort(reverse=True)

    # Número de generaciones a considerar.
    n_gen_considered=min(df.groupby('seed')['n_gen'].max())

    # Inicializar gráfica.
    ax=plt.subplot(111)

    # Para cada valor de N dibujar una curva.
    for ind_N in range(len(list_N)):
        # Inicializar listas.
        all_mean=[]
        all_q05=[]
        all_q95=[]

        # Por generación ir actualizando las listas anteriores.
        for gen in range(1,n_gen_considered+1):
            list_metric=[]
            for seed in list_seeds:
                df_seed=df[df['seed']==seed]
                if what=='scores':
                    matrix=score_matrix(df_seed,gen)
                if what=='positions':
                    matrix=ranking_matrix(df_seed,gen)
                if type=='pearson':
                    list_metric.append(pearson_corr(matrix[0],matrix[ind_N]))


            mean,q05,q95=bootstrap_median_and_confiance_interval(list_metric)
            all_mean.append(mean)
            all_q05.append(q05)
            all_q95.append(q95)
        
        # Dibujar curva.
        ax.fill_between(range(1,n_gen_considered+1),all_q05,all_q95,alpha=.5,linewidth=0)
        plt.plot(range(1,n_gen_considered+1),all_mean,linewidth=2,label=str(list_N[ind_N]))

    # Detalles de la gráfica.
    ax.set_ylabel('Pearson correlation')
    ax.set_xlabel('Generation')
    ax.set_title('Comparing similarity between the perfect \n and each '+str(what)+' ranking (blade-number '+str(blade_number)+')')
    ax.legend(title="N",bbox_to_anchor=(1.2, 0, 0, 1), loc='center')

    return ax

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Gráfica 1: rankings para blade-number 3
#--------------------------------------------------------------------------------------------------
# Cargar datos.
list_seeds=np.load('results/data/Turbines/CMA_ES_GenerationAnalysis/list_seeds.npy')

# Blade-number.
blade_number=3

# Número de generaciones en común para todas las semillas.
n_gen_per_seed=common_n_gen_per_seed(blade_number,list_seeds)

# Inicializar estructura de la gráfica.
plt.figure(figsize=[15,9])
plt.subplots_adjust(left=0.1,bottom=0.05,right=0.95,top=0.93,wspace=0.14,hspace=0.3)

for seed in list_seeds:
    #color=draw_rankings_per_generation([len(list_seeds),n_gen_per_seed],blade_number,seed)
    color=draw_rankings_per_generation([len(list_seeds)+1,n_gen_per_seed],blade_number,seed)

# Barra de colores.
ax = plt.subplot(len(list_seeds)+1,n_gen_per_seed,(len(list_seeds)*n_gen_per_seed+2,(len(list_seeds)+1)*n_gen_per_seed-1))
mpl.colorbar.ColorbarBase(ax, cmap=color,orientation='horizontal')
ax.set_xlabel('Ranking position')
ax.set_xticks(np.arange(.05,1.05,0.1))
ax.set_xticklabels(range(1,10+1,1))


plt.figtext(0.5, 0.95, 'Rankings per generation (blade-number '+str(blade_number)+')', ha='center', va='center')
#plt.savefig('results/figures/Turbines/CMA_ES_GenerationAnalysis_rankings.png')
plt.savefig('results/figures/Turbines/CMA_ES_GenerationAnalysis_rankings_legend.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# Gráfica 2: comparación de rankings (usando los scores) por generación blade-number 3
#--------------------------------------------------------------------------------------------------
# Inicializar gráfica.
plt.figure(figsize=[7,5])
plt.subplots_adjust(left=0.13,bottom=0.11,right=0.77,top=0.87,wspace=0.14,hspace=0.3)

# Crear gráfica.
draw_ranking_comparison_per_gen(blade_number,list_seeds,'pearson','scores')

def forward(x):
    return -np.log(x)

def inverse(x):
    return np.exp(-x)
#ax.set_yscale('function', functions=(forward, inverse))

# Guardar gráfica.
plt.savefig('results/figures/Turbines/CMA_ES_GenerationAnalysis_comparison_scores.png')
plt.show()
plt.close()

#--------------------------------------------------------------------------------------------------
# Gráfica 3: comparación de rankings (usando los posiciones) por generación blade-number 3
#--------------------------------------------------------------------------------------------------
# Inicializar gráfica.
plt.figure(figsize=[7,5])
plt.subplots_adjust(left=0.13,bottom=0.11,right=0.77,top=0.87,wspace=0.14,hspace=0.3)


# Crear gráfica.
draw_ranking_comparison_per_gen(blade_number,list_seeds,'pearson','positions')

def forward(x):
    return -np.log(x)

def inverse(x):
    return np.exp(-x)
#ax.set_yscale('function', functions=(forward, inverse))

# Guardar gráfica.
plt.savefig('results/figures/Turbines/CMA_ES_GenerationAnalysis_comparison_positions.png')
plt.show()
plt.close()


