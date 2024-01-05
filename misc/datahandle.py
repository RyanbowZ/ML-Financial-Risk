import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
#%%
Model=2

ys = pd.read_csv("D:\Documents\毕业设计\data\month\hs300.csv",index_col="time")[["val","ml","mla","theta","cc"]]

x1s = pd.read_csv("D:\Documents\毕业设计\data\daily\qh.csv")[["time","per","lncjl","lncjje","jc","lnccl"]]
#%%
train_y=[]
train_x=[]

row_x=[]
lTime=0
rep=1
for i in range(0,len(x1s)):
    if str(x1s.iloc[i]["lnccl"])[:1]=="#":
        continue
    if lTime != int(str(x1s.iloc[i]["time"])[:4]):
        lTime = int(str(x1s.iloc[i]["time"])[:4])
        #print(lTime)
        if len(row_x)==5:
            #print(rep)
            for j in range(5):
                row_x[j]=row_x[j]/rep
            train_x.append(row_x)
        rep=1
        row_x=[float(x1s.iloc[i]["per"]),float(x1s.iloc[i]["lncjl"]),float(x1s.iloc[i]["lncjje"]),float(x1s.iloc[i]["jc"]),float(x1s.iloc[i]["lnccl"])]
    else:
        row_x[0]+=float(x1s.iloc[i]["per"])
        row_x[1] += float(x1s.iloc[i]["lncjl"])
        row_x[2] += float(x1s.iloc[i]["lncjje"])
        row_x[3] += float(x1s.iloc[i]["jc"])
        row_x[4] += float(x1s.iloc[i]["lnccl"])
        rep+=1

#print(train_x)

sp=int(len(train_x)*0.6)
test_x=train_x[sp:len(train_x)-1]
train_x=train_x[0:sp]
for i in ys["cc"]:
    train_y.append(i)
train_y=train_y[0:sp]
#print(train_y)
print(len(train_x))
print(len(train_y))
#%%
if Model==1:
    clf=svm.SVC(kernel='rbf')
elif Model==2:
    clf = RandomForestClassifier(n_estimators=200)
elif Model==3:
    clf = GaussianNB()
elif Model==4:
    clf=KNeighborsClassifier()
elif Model==5:
    clf=hmm.GaussianHMM(n_components=2)

clf.fit(train_x,train_y)

y_pre = clf.predict(test_x)

print(y_pre)
