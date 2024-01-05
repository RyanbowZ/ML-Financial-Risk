import numpy as np
import pandas as pd

qh=pd.read_csv('qhpca.csv',index_col='time')[['y1','y2','y3','y4']]
k=20
#print(np.concatenate((np.array(qh.iloc[0]),np.array(qh.iloc[1]))))
l=[]
for i in range(k,len(qh)):
    a=np.array(qh.iloc[i-k])
    for j in range(i-k+1,i):
        b=np.array(qh.iloc[j])
        a=np.concatenate((a,b))
    l.append(a)

pd.DataFrame(l).to_csv('qhconcat.csv')

