import numpy as np
import pandas as pd


qq=pd.read_csv('hhpca_53.csv')

k=20
l=[]
for i in range(k,len(qq)+1):
    a=np.array(qq.iloc[i-k])
    for j in range(i-k+1,i):
        b=np.array(qq.iloc[j])
        a=np.concatenate((a,b))
    l.append(a)

pd.DataFrame(l).to_csv('hhcon_0503.csv')