import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm

do=pd.read_csv("qqname3.csv")[["index","price","st","ed","type"]]
lst=0
led=0
ltype="1"
ct=0
data=[]
stl={}
ctl=[]

for i in range(0,len(do)):
    c=do.iloc[i]
    if c["st"]!=lst:
        lst=c["st"]
        stl[c["st"]]=[]
        led=0
        ltype = "1"
        if ct!=0:
            ctl.append(ct)
        ct=0
    if c["ed"]!=led or c["type"]!=ltype:
        if ct!=0:
            ctl.append(ct)
        led=c["ed"]
        ltype= c["type"]
        stl[c["st"]].append(c["ed"])
        ct=0
    ct+=1
ctl.append(ct)
i=0
row=0
lrow=0
for cc in stl:

    for ci in stl[cc]:
        row+=ctl[i]
        r=int((lrow+row)*0.5)
        #print(r+1)
        i+=1
        data.append({"st":cc,"ed":ci,"index":do.iloc[r]["index"],"price":do.iloc[r]["price"],"type":do.iloc[r]["type"]})
        lrow=row

df=pd.DataFrame(data)
df.to_csv("qqname4.csv",encoding='utf_8_sig')
#print(stl)
#print(len(ctl))
#print(i)


