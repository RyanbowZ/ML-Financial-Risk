import numpy as np
import pandas as pd

ir=pd.read_csv('ff007merge2.csv')
am=pd.read_csv('ff007amount.csv',index_col='time')

l=[]
for i in range(len(ir)):
    c=ir.iloc[i]
    l.append(am.loc[int(str(c['time'])[:4])]['amount'])
ir['amount']=l
ir.to_csv('fr007x.csv')
