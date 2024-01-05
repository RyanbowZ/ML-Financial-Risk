import numpy as np
import pandas as pd
import shap
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
import lime
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt


def acu_curve(y, prob,m):
    fpr, tpr, threshold = metrics.roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = metrics.auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    #print(fpr,tpr,threshold)
    #plt.text(0,0.2,fpr)
    #plt.text(0, 0.3, tpr)
    plt.legend(loc="lower right")
    plt.savefig(str(m)+'_bd_roc.png')
    plt.close("all")
    #plt.show()


#%%
#Model=11
train_x=[]
train_y=[]

qh=pd.read_csv("qhmconcat.csv",index_col="time")
hs=pd.read_csv("bd.csv",index_col="time")


for i in range(len(qh)):
    c=qh.iloc[i]
    train_x.append(c.tolist())
    train_y.append(int(hs.iloc[i]["bd"]))
x_train0=train_x
y_train0=train_y
d=[]
#x_train0, x_test0, y_train0, y_test0 =  train_test_split( train_x, train_y, shuffle=False,train_size=1)
for Model in range(1,12):
    print(Model)
    if Model==1:
        #clf=svm.SVC(kernel='rbf',class_weight='balanced')
        clf=svm.LinearSVC(C=0.35,class_weight='balanced')


    elif Model==2:
        clf = RandomForestClassifier(n_estimators=15,max_depth=8)

    elif Model==3:
        clf = GaussianNB()

    elif Model==4:
        clf=KNeighborsClassifier(n_neighbors=2)

    elif Model==6:
        #clf=LinearDiscriminantAnalysis(store_covariance=True,tol=1e-3)
        clf=LinearDiscriminantAnalysis()
    elif Model==7:
        #clf=MLPClassifier(alpha=0.03,max_iter=150)
        clf=MLPClassifier(alpha=0.05,max_iter=150)

    elif Model==9:
        #clf = HistGradientBoostingClassifier(max_iter=25,l2_regularization=0.4,max_depth=18,learning_rate=0.06)
        clf = HistGradientBoostingClassifier(l2_regularization=0.3,max_iter=35,max_depth=25)

    elif Model==11:
        clf = LogisticRegression(C=0.25,class_weight='balanced')
    else:
        continue

    clf.fit(x_train0,y_train0)
    if Model in [1, 6, 8,9,10]:
        y_score = clf.decision_function(x_train0)
    else:
        y_score = clf.predict_proba(x_train0)[:,1]
    x = np.array(x_train0)

    sp = 91
    sp2 = 416




    acu_curve(y_train0,y_score,Model)

    #y_pre = clf.predict(x_test0)
    y_pre = clf.predict(x_train0)
    CM=metrics.confusion_matrix(y_train0,y_pre)
    auc=metrics.roc_auc_score(y_train0,y_pre)
    log_loss=metrics.log_loss(y_train0,y_pre)
    (tn, fp, fn, tp)= CM.ravel()
    acc=metrics.accuracy_score(y_train0,y_pre)
    ppv=tp/(tp+fp)
    tpr=tp/(tp+fn)
    fpr=fp/(fp+tn)
    forr=fn/(fn+tn)
    tnr=1-fpr
    fnr=1-tpr
    fdr=1-ppv
    npv=1-forr
    lrp=tpr/fpr
    lrn=fnr/tnr
    f1=2*ppv*tpr/(ppv+tpr)
    lift=ppv/((tp+fn)/(tp+tn+fp+fn))
    youden=tpr-fpr
    '''pd.DataFrame({'Model':Model,'tn':tn,'fp':fp,'fn':fn,'tp':tp,'auc':auc,'log_loss':log_loss,'acc':acc,
                  'ppv':ppv,'tpr':tpr,'fpr':fpr,'forr':forr,
                  'tnr':tnr,'fnr':fnr,'fdr':fdr,'npv':npv,
                  'f1':f1,'lrp':lrp,'lrn':lrn,'lift':lift,'youden':youden},index=[0]).to_csv('./result/'+str(Model)+'modelresult2.csv')'''
    d.append({'Model': Model, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'auc': auc, 'log_loss': log_loss, 'acc': acc,
                  'ppv': ppv, 'tpr': tpr, 'fpr': fpr, 'forr': forr,
                  'tnr': tnr, 'fnr': fnr, 'fdr': fdr, 'npv': npv,
                  'f1': f1, 'lrp': lrp, 'lrn': lrn, 'lift': lift, 'youden': youden})
    #print(y_pre)
    #print(y_test0)
    #print(clf.score(x_train0,y_train0))
    #print(metrics.log_loss(y_train0,y_pre))
    #print(metrics.roc_auc_score(y_train0,y_pre))
    #print(Model)
    #y=pd.DataFrame(y_pre)
    '''y_ans=[]
    y_ans.append(clf.score(x_train0,y_train0))
    y_ans.append(metrics.log_loss(y_train0,y_pre))
    y_ans.append(metrics.roc_auc_score(y_train0,y_pre))'''
    qh['y_test_model'+str(Model)]=y_pre
    #hs['y_test_model' + str(Model)+'rate'] = y_ans
pd.DataFrame(d).to_csv('bd_SumModelResult.csv')
qh['y_real']=train_y
qh.to_csv('bd_result_compare.csv')

