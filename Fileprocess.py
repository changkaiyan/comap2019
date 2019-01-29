import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('./data/optcommunity.csv',index_col=0)
process=data.values
sns.pairplot(data,x_vars='averageGDP',y_vars='averageSO2',kind='reg')
plt.savefig('./picture/beijingGDPSO2.png')
sns.pairplot(data,kind='reg')
plt.savefig('./picture/beijingall.png')