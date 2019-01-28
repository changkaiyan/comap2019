import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data=pd.read_excel('./data/datacity.xlsx')
data2010=pd.read_excel('./data/2010environment.xlsx')
environmentout=sorted(data.values[:,2],reverse=True)
environmentout2010=sorted(data2010.values[:,1]/10000,reverse=True)
n=[i for i in range(len(environmentout))]
plt.plot(n,environmentout,label='2017')
plt.plot(n,environmentout2010,label='2010')
plt.legend()
plt.xlabel('number')
plt.ylabel('price')
plt.title('Environmental demand curve ')