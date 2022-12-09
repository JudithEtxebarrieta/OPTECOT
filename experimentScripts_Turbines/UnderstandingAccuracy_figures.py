# Mediante este script se representan gráficamente los resultados numéricos obtenidos al 
# ejecutar "UnderstandingAccuracy_data.py".

#==================================================================================================
# LIBRERÍAS
#==================================================================================================
import numpy as np
import matplotlib as mpl
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCIÓN 1
# Parámetros:
#   >data: datos sobre los cuales se calculará el rango entre percentiles.
#   >bootstrap_iterations: número de submuestras que se considerarán de data para poder calcular el 
#    rango entre percentiles de sus medias.
# Devolver: la media de los datos originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre data.

def bootstrap_median_and_confiance_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

# FUNCIÓN 2
# Parámetros:
#   >str_col: columna de la base de datos cuyos elementos son originariamente listas de números 
#    (float o integer), pero al guardar la base de datos y leerla se consideran como strings.
#   >type_elems: tipo de números que guardan las listas de las columnas, float o integer.
# Devolver: una lista de listas.

def form_str_col_to_float_list_col(str_col,type_elems):
    float_list_col=[]
    for str in str_col:
        #Eliminar corchetes string.
        str=str.replace('[','')
        str=str.replace(']','')

        #Convertir str en lista de floats.
        float_list=str.split(", ")
        if len(float_list)==1:
            float_list=str.split(" ")
        
        #Acumular conversiones.
        if type_elems=='float':
            float_list=[float(i) for i in float_list]
        if type_elems=='int':
            float_list=[int(i) for i in float_list]
        float_list_col.append(float_list)

    return float_list_col

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================

# Lectura de datos.
df=pd.read_csv('results/data/Turbines/UnderstandingAccuracy.csv',index_col=0)

# Poner en formato adecuado las columnas que contienen listas
df['all_scores']=form_str_col_to_float_list_col(df['all_scores'],'float')
df['ranking']=form_str_col_to_float_list_col(df['ranking'],'int')
df['all_times']=form_str_col_to_float_list_col(df['all_times'],'float')


# Inicializar figura (tamaño y margenes)
plt.figure(figsize=[12,9])
plt.subplots_adjust(left=0.09,bottom=0.11,right=0.95,top=0.88,wspace=0.3,hspace=0.4)

#--------------------------------------------------------------------------------------------------
# GRÁFICA 1 
# Repercusión de la precisión de N en el tiempo requerido para hacer una evaluación.
#--------------------------------------------------------------------------------------------------
x=df['N']
y_mean=[]
y_q05=[]
y_q95=[]
for times in df['all_times']:
    mean,q05,q95=bootstrap_median_and_confiance_interval(times)
    y_mean.append(mean)
    y_q05.append(q05)
    y_q95.append(q95)

ax1=plt.subplot(221)
ax1.fill_between(x,y_q05,y_q95,alpha=.5,linewidth=0)
plt.plot(x,y_mean,linewidth=2)
ax1.set_xlabel('N')
ax1.set_ylabel('Time per evaluation')
ax1.set_title('Evaluation time depending on N')

#--------------------------------------------------------------------------------------------------
# GRÁFICA 2 
# Repercusión de la precisión de N en la calidad de la solución del problema de optimización
# (comparando rankings).
#--------------------------------------------------------------------------------------------------
x=range(1,len(df['all_scores'][0])+1)
ax2=plt.subplot(224)

matrix=[]
for i in df['ranking']:
    matrix.append(i)
matrix=np.matrix(matrix)

color = sns.color_palette("deep", len(df['ranking'][0]))
ax2 = sns.heatmap(matrix, cmap=color,linewidths=.5, linecolor='lightgray')

colorbar=ax2.collections[0].colorbar
colorbar.set_label('Ranking position', rotation=270)
colorbar.set_ticks(np.arange(.5,len(df['ranking'][0])-.9,.9))
colorbar.set_ticklabels(range(len(df['ranking'][0]),0,-1))

ax2.set_xlabel('Turbine design')
ax2.set_xticklabels(range(1,len(df['ranking'][0])+1))

ax2.set_ylabel('N')
ax2.set_yticks(np.arange(0.5,df.shape[0]+0.5))
ax2.set_yticklabels(df['N'],rotation=0)

ax2.set_title('Comparing rankings depending on N')

#--------------------------------------------------------------------------------------------------
# GRÁFICA 3 
# Repercusión de la precisión de N en la calidad de la solución del problema de optimización
# (comparando perdidas de score).
#--------------------------------------------------------------------------------------------------
x=[str(i) for i in df['N']]
y=[]
best_scores=df['all_scores'][0]
for i in range(0,df.shape[0]):
    ind_best_turb=df['ranking'][i][-1]
    quality_loss=(max(best_scores)-best_scores[ind_best_turb])/max(best_scores)
    y.append(quality_loss)


ax3=plt.subplot(223)
plt.scatter(x,y)
ax3.set_xlabel('N')
ax3.set_ylabel('Score loss (%)')
plt.xticks(rotation = 45)
ax3.set_title('Comparing loss of score quality depending on N')


#--------------------------------------------------------------------------------------------------
# GRÁFICA 4 
# Evaluaciones extra que se pueden hacer al considerar un N menos preciso.
#--------------------------------------------------------------------------------------------------
x=[str(i) for i in df['N']]
y=[]
for i in range(0,df.shape[0]):
    extra_eval=(df['total_time'][0]-df['total_time'][i])/df['time_per_eval'][i]
    y.append(extra_eval)

ax4=plt.subplot(222)
plt.bar(x,y)
ax4.set_xlabel('N')
ax4.set_ylabel('Number of extra evaluations')
plt.xticks(rotation = 45)
ax4.set_title('Extra evaluations in the same time required by maximum N')



plt.savefig('results/figures/Turbines/UnderstandingAccuracy.png')
plt.show()
