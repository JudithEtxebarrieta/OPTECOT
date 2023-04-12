# Mediante este script se representan graficamente los resultados numericos obtenidos al 
# ejecutar "UnderstandingAccuracy_data.py".

#==================================================================================================
# LIBRERIAS
#==================================================================================================
import numpy as np
import matplotlib as mpl
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from itertools import combinations
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

#==================================================================================================
# FUNCIONES
#==================================================================================================
# FUNCION 1
# Parametros:
#   >data: datos sobre los cuales se calculara el rango entre percentiles.
#   >bootstrap_iterations: numero de submuestras que se consideraran de data para poder calcular el 
#    rango entre percentiles de sus medias.
# Devolver: la media de los datos originales junto a los percentiles de las medias obtenidas del 
# submuestreo realizado sobre data.

def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
    mean_list=[]
    for i in range(bootstrap_iterations):
        sample = np.random.choice(data, len(data), replace=True) 
        mean_list.append(np.mean(sample))
    return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)

# FUNCION 2
# Parametros:
#   >str_col: columna de la base de datos cuyos elementos son originariamente listas de numeros 
#    (float o integer), pero al guardar la base de datos y leerla se consideran como strings.
#   >type_elems: tipo de numeros que guardan las listas de las columnas, float o integer.
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

# FUNCION 3 (Obtener ranking a partir de lista conseguida tras aplicar "np.argsort" sobre una lista original)
def from_argsort_to_ranking(list):
    new_list=[0]*len(list)
    i=0
    for j in list:
        new_list[j]=i
        i+=1
    return new_list

# FUNCION 4 (Ordenar lista segun el orden para cada posicion definido en argsort)
def custom_sort(list,argsort):
    new_list=[]
    for index in argsort:
        new_list.append(list[index])
    return new_list


# FUNCION 5 (Calculo de la distancia inversa normalizada de tau kendall entre dos rankings)
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

# FUNCION 6 (Eliminar elementos en ciertas posiciones de una lista)
def remove_element_in_idx(list_elements,list_idx):
    new_list=[]
    for i in range(len(list_elements)):
        if i not in list_idx:
            new_list.append(list_elements[i])
    return new_list

# FUNCION 7 (Grafica de motivacion I)
def from_data_to_figuresI(df):

    # Poner en formato adecuado las columnas que contienen listas
    df['all_scores']=form_str_col_to_float_list_col(df['all_scores'],'float')
    df['ranking']=form_str_col_to_float_list_col(df['ranking'],'int')
    df['all_times']=form_str_col_to_float_list_col(df['all_times'],'float')

    # Inicializar figura (tamano y margenes)
    plt.figure(figsize=[12,9])
    plt.subplots_adjust(left=0.09,bottom=0.11,right=0.95,top=0.88,wspace=0.3,hspace=0.4)

    #--------------------------------------------------------------------------------------------------
    # GRAFICA 1 
    # Repercusion de la precision de N en el tiempo requerido para hacer una evaluacion.
    #--------------------------------------------------------------------------------------------------
    x=df['N']
    y_mean=[]
    y_q05=[]
    y_q95=[]
    for times in df['all_times']:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(times)
        y_mean.append(mean)
        y_q05.append(q05)
        y_q95.append(q95)

    ax1=plt.subplot(221)
    ax1.fill_between(x,y_q05,y_q95,alpha=.5,linewidth=0)
    plt.plot(x,y_mean,linewidth=2)
    ax1.set_xlabel('N')
    ax1.set_ylabel('Time per evaluation')
    ax1.set_title('Evaluation time depending on N')

    #Zoom
    axins = zoomed_inset_axes(ax1,6,loc='upper left',borderpad=1)
    axins.fill_between(x,y_q05,y_q95,alpha=.5,linewidth=0)
    plt.plot(x,y_mean,linewidth=2)
    axins.set_xlim(0,60)
    axins.set_ylim(2,7)
    axins.yaxis.tick_right()
    #plt.xticks(visible=False)
    #plt.yticks(visible=False)
    #mark_inset(ax1,axins,loc1=2,loc2=4)

    #--------------------------------------------------------------------------------------------------
    # GRAFICA 2 
    # Repercusion de la precision de N en la calidad de la solucion del problema de optimizacion
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
    # GRAFICA 3 
    # Repercusion de la precision de N en la calidad de la solucion del problema de optimizacion
    # (comparando perdidas de score).
    #--------------------------------------------------------------------------------------------------
    x=[str(i) for i in df['N']]
    y=[]
    best_scores=df['all_scores'][0]
    for i in range(0,df.shape[0]):
        ind_best_turb=df['ranking'][i].index(max(df['ranking'][i]))
        quality_loss=(max(best_scores)-best_scores[ind_best_turb])/max(best_scores)
        y.append(quality_loss)


    ax3=plt.subplot(223)
    plt.scatter(x,y)
    ax3.set_xlabel('N')
    ax3.set_ylabel('Score loss (%)')
    plt.xticks(rotation = 45)
    ax3.set_title('Comparing loss of score quality depending on N')


    #--------------------------------------------------------------------------------------------------
    # GRAFICA 4 
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



    plt.savefig('results/figures/Turbines/UnderstandingAccuracyI.png')
    plt.show()

# FUNCION 8 (Grafica de motivacion II)
def from_data_to_figuresII(df,blade_number):

    # Inicializar grafica.
    plt.figure(figsize=[20,10])
    plt.subplots_adjust(left=0.06,bottom=0.11,right=0.95,top=0.88,wspace=0.3,hspace=0.69)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: tiempo de ejecucion por accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(311)

    list_acc=list(set(df['accuracy']))
    list_acc.sort()


    all_means=[]
    all_q05=[]
    all_q95=[]

    for accuracy in list_acc:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list(df[df['accuracy']==accuracy]['time']))
        all_means.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    ax.fill_between(list_acc,all_q05,all_q95, alpha=.5, linewidth=0)
    plt.plot(list_acc, all_means, linewidth=2) 
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Time per evaluation")
    ax.set_title('Evaluation cost depending on accuracy')


    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: extra de evaluaciones.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(312)

    list_extra_eval=[]
    for i in range(0,len(all_means)):
        # Ahorro de tiempo.
        save_time=all_means[-1]-all_means[i]
        # Numero de evaluaciones extra que se podrian hacer para agotar el tiempo que se necesita por defecto.
        extra_eval=save_time/all_means[i]
        if extra_eval<0:
            extra_eval=0
        list_extra_eval.append(extra_eval)

    plt.bar([str(acc) for acc in list_acc],list_extra_eval)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Number of extra evaluations')
    plt.xticks(rotation = 45)
    ax.set_title('Extra evaluations in the same amount of time required for maximum accuracy')


    #----------------------------------------------------------------------------------------------
    # GRAFICA 3: soluciones nulas por valor de accuracy.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(313)

    list_null_scores=[]
    for accuracy in list_acc:
        list_null_scores.append(sum(np.array(list(df[df['accuracy']==accuracy]['score']))==0)/100)

    plt.bar([str(acc) for acc in list_acc],list_null_scores)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Percentaje of null solutions')
    plt.xticks(rotation = 45)
    ax.set_title('Presence of null solutions')


    # Guardar imagen.
    plt.savefig('results/figures/Turbines/UnderstandingAccuracyII_nullsolutions_bladenumber'+str(blade_number)+'.png')
    plt.show()
    plt.close()

# FUNCION 9 (Grafica para el paper)
def from_data_to_figures_paper(df,blade_number=None):

    # Inicializar grafica.
    if blade_number==None:
        fig=plt.figure(figsize=[15,3.5],constrained_layout=True)
    else:
        fig=plt.figure(figsize=[15,15],constrained_layout=True)

    # plt.subplots_adjust(left=0.06,bottom=0.21,right=0.95,top=0.78,wspace=0.1,hspace=0.69)
    subfigs = fig.subfigures(1, 4, width_ratios=[2,2,0.1,4], wspace=0)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: tiempo de ejecucion por accuracy y extra de evaluaciones.
    #----------------------------------------------------------------------------------------------
    list_acc=list(set(df['accuracy']))
    list_acc.sort()
    list_acc_str=[str(acc) for acc in list_acc]
    list_acc_str.reverse()

    # Tiempos de ejecucion por evaluacion.
    all_means=[]
    all_q05=[]
    all_q95=[]

    for accuracy in list_acc:
        mean,q05,q95=bootstrap_mean_and_confidence_interval(list(df[df['accuracy']==accuracy]['time']))
        all_means.append(mean)
        all_q05.append(q05)
        all_q95.append(q95)

    # Evaluaciones extra.
    list_extra_eval=[]
    for i in range(0,len(all_means)):
        # Ahorro de tiempo.
        save_time=all_means[-1]-all_means[i]
        # Numero de evaluaciones extra que se podrian hacer para agotar el tiempo que se necesita por defecto.
        extra_eval=save_time/all_means[i]
        list_extra_eval.append(extra_eval)

    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,3))

    ax=subfigs[0].subplots()
    y=[i*(-1) for i in all_means]
    y.reverse()
    plt.barh(list_acc_str, y, align='center',color=color[0])
    plt.title("\n Cost per evaluation")
    plt.ylabel("Accuracy")
    plt.xlabel("Time")
    ax.set_xticks(np.arange(0,-4,-1))
    ax.set_xticklabels(np.arange(0,4,1),rotation=0)
    

    ax=subfigs[1].subplots()
    for i in range(len(list_extra_eval)):
        if list_extra_eval[i]<0:
            list_extra_eval[i]=0
    list_extra_eval.reverse()
    plt.barh(list_acc_str, list_extra_eval, align='center',color=color[1])
    plt.yticks([])
    plt.title("Extra evaluations in the same amount of \n time required for maximum accuracy")
    # plt.ylabel("Accuracy")
    plt.xlabel("Evaluations")


    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: perdida de calidad en las evaluaciones considerar accuracys menores y existencia 
    # de un accuracy menor con el que se obtiene la maxima calidad.
    #----------------------------------------------------------------------------------------------
    def ranking_matrix_sorted_by_max_acc(ranking_matrix):
        # Argumentos asociados al ranking del accuracy 1 ordenado de menor a mayor.
        argsort_max_acc=np.argsort(ranking_matrix[-1])

        # Reordenar filas de la matriz.
        sorted_ranking_list=[]
        for i in range(len(ranking_list)):
            sorted_ranking_list.append(custom_sort(ranking_list[i],argsort_max_acc))

        return sorted_ranking_list

    def absolute_distance_matrix_between_rankings(ranking_matrix):
        abs_dist_matrix=[]
        for i in range(len(ranking_matrix)):
            abs_dist_matrix.append(np.abs(list(np.array(ranking_matrix[i])-np.array(ranking_matrix[-1]))))

        max_value=max(max(i) for i in abs_dist_matrix)
        norm_abs_dist_matrix=list(np.array(abs_dist_matrix)/max_value)
        
        return norm_abs_dist_matrix

    ax=subfigs[3].subplots()

    ranking_list=[]

    for accuracy in list_acc:
        df_acc=df[df['accuracy']==accuracy]
        ranking=from_argsort_to_ranking(list(df_acc['score'].argsort()))
        ranking_list.append(ranking)

    ranking_matrix=np.matrix(absolute_distance_matrix_between_rankings(ranking_matrix_sorted_by_max_acc(ranking_list)))
    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)

    ax = sns.heatmap(ranking_matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    colorbar=ax.collections[0].colorbar
    colorbar.set_label('Normalized distance')

    ax.set_xlabel('Solution (sorted by the ranking of maximum accuracy)')
    ax.set_xticks(range(0,ranking_matrix.shape[1],10))
    ax.set_xticklabels(range(1,ranking_matrix.shape[1]+1,10),rotation=0)

    ax.set_ylabel('Accuracy')
    ax.set_title('Normalized absolute distances between the respective positions \n of the best and the rest rankings')
    ax.set_yticks(np.arange(0.5,len(list_acc)+0.5,1))
    ax.set_yticklabels(list_acc,rotation=0)

    # Guardar imagen.
    if blade_number==None:
        plt.savefig('results/figures/Turbines/UnderstandingAccuracy_paper.png')
    else:
        plt.savefig('results/figures/Turbines/UnderstandingAccuracy_bladenumber'+str(blade_number)+'.png')
    plt.show()
    plt.close()

#==================================================================================================
# PROGRAMA PRINCIPAL
#==================================================================================================
# Lectura de datos.
df1=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyI.csv',index_col=0)
df2=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII.csv',index_col=0)
df2_bladenumber3=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber3.csv',index_col=0)
df2_bladenumber5=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber5.csv',index_col=0)
df2_bladenumber7=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumber7.csv',index_col=0)
df2_bladenumberAll=pd.read_csv('results/data/Turbines/UnderstandingAccuracy/UnderstandingAccuracyII_bladenumberAll.csv',index_col=0)

# Dibujar graficas.
from_data_to_figuresI(df1)
from_data_to_figuresII(df2_bladenumberAll,'All')
from_data_to_figuresII(df2_bladenumber3,3)
from_data_to_figuresII(df2_bladenumber5,5)
from_data_to_figuresII(df2_bladenumber7,7)
from_data_to_figures_paper(df2_bladenumberAll,blade_number='All')
from_data_to_figures_paper(df2_bladenumber3,blade_number=3)
from_data_to_figures_paper(df2_bladenumber5,blade_number=5)
from_data_to_figures_paper(df2_bladenumber7,blade_number=7)
from_data_to_figures_paper(df2)

