import numpy as np
from matplotlib import pyplot as plt


data_small = [0.06019538, 0.89107288, 0.56234304, 0.89412721, 0.98606047,
       0.37555595, 0.92892888, 0.58679572, 0.68545189, 0.25607071]

data_medium = [0.06019538, 0.89107288, 0.56234304, 0.89412721, 0.98606047,
       0.37555595, 0.92892888, 0.58679572, 0.68545189, 0.25607071,
       0.87620604, 0.94756544, 0.43744465, 0.2378663 , 0.60806766,
       0.15646704, 0.60516878, 0.11537174, 0.7594546 , 0.4862541 ,
       0.49312795, 0.40778569, 0.72928631, 0.01037447, 0.68604081]

data_big = [0.06019538, 0.89107288, 0.56234304, 0.89412721, 0.98606047,
       0.37555595, 0.92892888, 0.58679572, 0.68545189, 0.25607071,
       0.87620604, 0.94756544, 0.43744465, 0.2378663 , 0.60806766,
       0.15646704, 0.60516878, 0.11537174, 0.7594546 , 0.4862541 ,
       0.49312795, 0.40778569, 0.72928631, 0.01037447, 0.68604081,
       0.70817771, 0.11845191, 0.52949286, 0.73652986, 0.63216692,
       0.00838538, 0.47049212, 0.11220318, 0.28533206, 0.73748116,
       0.73933534, 0.10334525, 0.14020212, 0.76507966, 0.29459818,
       0.6379741 , 0.25812304, 0.67432954, 0.95168174, 0.57229172,
       0.03063801, 0.91621206, 0.4494862 , 0.75340248, 0.11987101]


bootstrap_iterations = 1000
for j in range(3):
  data = [data_small, data_medium, data_big][j]
  data_label = ["small", "medium", "large"][j]

  mean_list = np.zeros(bootstrap_iterations)
  for i in range(bootstrap_iterations):
    y = np.random.choice(data, len(data), replace=True) # generate bootstrap sample
    avg = np.mean(y) # calculate average of sample
    mean_list[i] = avg
  plt.scatter([j,j], [np.quantile(mean_list, 0.05), np.quantile(mean_list, 0.95)], marker="_", label="95% Confidence interval", color="black")
  plt.scatter([j], [np.quantile(mean_list, 0.5)], marker="x", label="Estimation", color="black")
  plt.annotate(data_label, (j, 0.1))
  plt.ylim((0,1))

# Remove duplicate legends
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
  
plt.show()

