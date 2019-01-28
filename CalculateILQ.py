import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_excel('./data/datacity.xlsx',index_col=0)
dataset=data.values
all=np.sum(dataset)
k=(dataset[:,3]/all)
ilq=k*(dataset[:,0]/dataset[:,1])/(np.sum(dataset[:,0])/np.sum(dataset[:,1]))
plt.scatter(ilq,dataset[:,-3], color='g',linewidth = 3)
plt.ylabel('EnvirenmentOut')
plt.xlabel('ILQ')
plt.title('E-L figure')
plt.savefig('./picture/E-L figure.png')
sns.pairplot(data,kind='reg')
#plt.hist(ilq, bins =7, color = 'r',alpha=0.5,rwidth= 0.9, normed=True)
plt.savefig('./picture/cityinChina.png')
plt.show()