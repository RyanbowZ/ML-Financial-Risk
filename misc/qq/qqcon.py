import numpy as np
import pandas as pd


qq=pd.read_csv('qqpca_427td.csv',index_col="trade_date")

k=20
l=[]
for i in range(k,len(qq)):
    a=np.array(qq.iloc[i-k])
    for j in range(i-k+1,i):
        b=np.array(qq.iloc[j])
        a=np.concatenate((a,b))
    l.append(a)

pd.DataFrame(l).to_csv('qqconcat_427.csv')