import numpy as np
import pandas as pd

qd=pd.read_csv('hhx0503.csv')
aa=pd.read_csv('aa2.csv')
bb=pd.read_csv('bb.csv')
ll=[[] for i in range(5)]

#print(qd.iloc[0])
#print(aa.iloc[3])
for i in range(len(qd)):
    c=qd.iloc[i]
    #print(np.array(c)@np.array(aa.iloc[0]))
    for j in range(5):
        ll[j].append(float(np.array(c)@np.array(aa.iloc[j])))

#print(ll)
for j in range(5):
    qd['feature_'+str(j)]=ll[j]
qd.to_csv('hhpca_0503.csv')