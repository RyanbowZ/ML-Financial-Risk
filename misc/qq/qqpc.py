import numpy as np
import pandas as pd

qd=pd.read_csv('yhat_427.csv')
aa=pd.read_csv('aa2.csv')
bb=pd.read_csv('bb2.csv')
ll=[[] for i in range(9)]

#print(qd.iloc[0])
#print(aa.iloc[3])
for i in range(len(qd)):
    c=qd.iloc[i]
    #print(np.array(c)@np.array(aa.iloc[0]))
    for j in range(9):
        ll[j].append(float(np.array(c)@np.array(aa.iloc[j]))+float(bb.iloc[j]))

print(ll)
for j in range(9):
    qd['feature_'+str(j)]=ll[j]
qd.to_csv('qqpca_427.csv')