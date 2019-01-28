import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def getscore(gdp,water,air,struct,people,k):
    waterweight=0.3218
    airweight=0.6751
    averageGDP=0.2914
    structure=0.1336
    peopledense=0.5750
    gdp=(gdp-min(gdp))/(max(gdp)-min(gdp))
    water=(water-min(water))/(max(water)-min(water))
    air=(air-min(air))/(max(air)-min(air))
    struct=(struct-min(struct))/(max(struct)-min(struct))
    people=(people-min(people))/(max(people)-min(people))
    return (1/(1+k))*(k*(waterweight*water+air*airweight)+(gdp*averageGDP+struct*structure+people*peopledense))
data=pd.read_excel('./data/citytocheck.xlsx',index_col=0)
dataset=data.values
ran=np.linspace(0.6,1.4,9)
score=np.array([])
for i,k in enumerate(ran):
    score=np.append(score,np.array(getscore(dataset[:,1],dataset[:,2],dataset[:,3],dataset[:,4],dataset[:,-2],k)),axis=0)
t=[]
for i in range(dataset.shape[0]):
    m=[]
    for j in range(dataset.shape[1]):
        m.append(score[i+j*dataset.shape[0]])
    t.append(m)
for k,i in enumerate(t):
    plt.plot(ran,i,label=data.index[k])
plt.xlabel('k')
plt.ylabel('score')
plt.legend(loc='upper right')
plt.savefig('./picture/1.png')
plt.title(' ')