import numpy as np
import pandas as pd
fp="./data/develop.csv"
data=pd.read_csv(fp,index_col=0,encoding='utf8')
m,n=data.shape
#data=data.as_matrix(columns=None)
data=data.values[:,[1,4,7]]
k=1/np.log(m)
yij=data.sum(axis=0)
pij=data/yij
test=pij*np.log(pij)
test=np.nan_to_num(test)
ej=-k*(test.sum(axis=0))
wi=(1-ej)/np.sum(1-ej)
wi
