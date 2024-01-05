import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from hmmlearn import hmm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#%%
Model=11
train_x=[]
train_y=[]

qh=pd.read_csv("qhxxx.csv",index_col="time")
hs=pd.read_csv("qhy.csv",index_col="time")

for i in range(len(qh)):
    c=qh.iloc[i]
    train_x.append(c.tolist())
    train_y.append(int(hs.loc[qh.index[i]]["rc2"]))
#print(train_x)
#print(train_y)
'''sp=int(len(train_x)*0.6)
test_x=train_x[sp:len(train_x)-1]
train_x=train_x[0:sp]
train_y=train_y[0:sp]'''
x_train0, x_test0, y_train0, y_test0 =  train_test_split( train_x, train_y, shuffle=False,train_size=0.8)

if Model==1:
    #clf=svm.SVC(kernel='rbf',class_weight='balanced')
    clf=svm.LinearSVC(C=0.2,class_weight='balanced')
elif Model==2:
    clf = RandomForestClassifier(n_estimators=25)
elif Model==3:
    clf = GaussianNB()
elif Model==4:
    clf=KNeighborsClassifier()
elif Model==5:
    clf=hmm.GaussianHMM(n_components=2)#problem
elif Model==6:
    clf=LinearDiscriminantAnalysis()
elif Model==7:
    clf=MLPClassifier()
elif Model==8:
    clf=SelfTrainingClassifier(svm.LinearSVC(C=0.2))
elif Model==9:
    clf = HistGradientBoostingClassifier()
elif Model==10:
    clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
elif Model==11:
    clf = LogisticRegression()

clf.fit(x_train0,y_train0)

y_pre = clf.predict(x_test0)

print(y_pre)
print(y_test0)
print(clf.score(x_train0,y_train0))
print(metrics.log_loss(y_test0,y_pre))
print(metrics.roc_auc_score(y_test0,y_pre))
print(Model)
y=pd.DataFrame(y_pre)
y['y_test']=y_test0
y.to_csv(str(Model)+'y_result.csv')

