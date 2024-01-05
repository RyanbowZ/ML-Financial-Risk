import numpy as np
import pandas as pd

qd=pd.read_csv('hh_week.csv',index_col="time")
aa=pd.read_csv('aaweek.csv')
bb=pd.read_csv('bbweek.csv')

l1=[]
l2=[]
l3=[]
#print(np.array([-0.014375049,1.301368626,1.27,-204.288,-1.165953225])@np.array([-5.961604651,0.66780515,0.645820606,-0.001867175,0.309359015]))
l4=[]
l5=[]
l6=[]
l7=[]
#print(qd.iloc[0])
#print(aa.iloc[3])
for i in range(len(qd)):
    c=qd.iloc[i]
    #print(np.array(c)@np.array(aa.iloc[0]))
    l1.append(np.array(c)@np.array(aa.iloc[0])+float(bb.iloc[0]))
    l2.append(np.array(c) @ np.array(aa.iloc[1])+float(bb.iloc[1]))
    l3.append(np.array(c) @ np.array(aa.iloc[2])+float(bb.iloc[2]))
    l4.append(np.array(c) @ np.array(aa.iloc[3])+float(bb.iloc[3]))
    l5.append(np.array(c) @ np.array(aa.iloc[4]) + float(bb.iloc[4]))
#print(l)
qd['y1']=l1
qd['y2']=l2
qd['y3']=l3
qd['y4']=l4
qd['y5']=l5

qd.to_csv('hhwpcaresult.csv')
qh=pd.read_csv('hhwpcaresult.csv',index_col='time')[['y1','y2']]
k=12
#print(np.concatenate((np.array(qh.iloc[0]),np.array(qh.iloc[1]))))
l=[]
for i in range(k,len(qh)+1):
    a=np.array(qh.iloc[i-k])
    for j in range(i-k+1,i):
        b=np.array(qh.iloc[j])
        a=np.concatenate((a,b))
    l.append(a)

pd.DataFrame(l).to_csv('hhwconcat.csv')