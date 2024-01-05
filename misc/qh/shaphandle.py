import numpy as np
import pandas as pd
import shap
from shap._explanation import Explanation
import matplotlib.pyplot as plt

class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self.get(name)
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value


#x_train0, x_test0, y_train0, y_test0 =  train_test_split( train_x, train_y, shuffle=False,train_size=1)
for Model in range(1,12):
    l = []
    ll = [[] for i in range(5)]
    if Model==5:
        continue
    sv=pd.read_csv(str(Model) + 'shap_value_values.csv')
    j=1
    l0=0
    l1=0
    l2=0
    l3=0
    for i in range(0,20):
        l0+=sv.iloc[:,j]
        l1 += sv.iloc[:,1+j]
        l2 += sv.iloc[:,2+j]
        l3 += sv.iloc[:,3+j]
        j+=4
    l.append(l0)
    l.append(l1)
    l.append(l2)
    l.append(l3)
    lc=pd.DataFrame(l).T
    lc.columns=['y1','y2','y3','y4']
    #lc.to_csv(str(Model) + 'shap_l.csv')
    lc['y5']=np.zeros(len(lc))
    an=pd.read_csv('aa2ni.csv')
    #bb=pd.read_csv('bb.csv')
    for ii in range(len(lc)):
        c = lc.iloc[ii]
        for jj in range(5):
            '''print(np.array(an.iloc[jj]))
            print(c)
            print(np.array(an.iloc[jj])@np.array(bb))'''
            ll[jj].append(float(np.array(an.iloc[jj])@np.array(c) ) )
            #ll[jj].append(float(np.array(an.iloc[jj]) @ np.array(c)) )
    for jj in range(5):
        lc['feature_' + str(jj)] = ll[jj]
    #lc.to_csv(str(Model) + 'shap_ni_real_2.csv')
    dt=pd.read_csv('qhhpca2.csv')
    shap_values=[]
    sv1=[]
    sv2=[]
    sv3=[]

    for ii in range(len(lc)):
        #svv=ObjDict()
        svv={}
        svv['values']=np.array(lc.iloc[ii,5:10])
        sv1.append(np.array(lc.iloc[ii, 5:10]))
        svv['data']=np.array(dt.iloc[ii])
        sv2.append(np.array(dt.iloc[ii]))
        svv['base_values']=np.array(lc.iloc[ii,5:10]).sum()
        sv3.append(np.array(lc.iloc[ii,5:10]).sum())
        svv['feature_names']=['x1','x2','x3','x4','x5']
        shap_values.append(svv)
    #sa=pd.DataFrame(shap_values)
    #print(shap_values[91]['values'])
    shap_values_explain=Explanation(np.array(sv1),base_values=np.array(sv3),data=np.array(sv2),feature_names=['x1','x2','x3','x4','x5'])

    train_x = []
    #qh=pd.read_csv("qhxa.csv", index_col="time")
    for ii in range(len(dt)):
        c = dt.iloc[i]
        train_x.append(c.tolist())
    x = np.array(train_x)
    sp=91
    sp2=416
    #print(shap_values[15])
    '''shap.summary_plot(shap_values_explain, x, show=False)
    plt.savefig(str(Model) + "shap_summary3.png")
    plt.close()'''
    shap.dependence_plot(shap_values_explain,x,interaction_index='Capital Gain')
    plt.savefig(str(Model) + "shap_pdp.png")
    plt.close()
    '''shap.plots.waterfall(shap_values[sp],show = False)
    plt.savefig(str(Model) + "shap_waterfall2.png")
    plt.close()'''
    '''shap.plots.force(shap_values[sp],show = False)
    plt.savefig(str(Model) + "shap_force2.png")
    plt.close()
    shap.summary_plot(shap_values,x,show = False)
    plt.savefig(str(Model) + "shap_summary2.png")
    plt.close()
    shap.summary_plot(shap_values, x,plot_type="bar",show = False)
    plt.savefig(str(Model) + "shap_summary_bar2.png")
    plt.close()
    shap.plots.heatmap(shap_values,show = False)
    plt.savefig(str(Model) + "shap_heatmap2.png")
    plt.close()'''

    '''shap.force_plot(shap_values.base_values,shap_values.values,x,show = False)
    plt.savefig(str(Model) + "shap_force_plot2.png")
    plt.close()
    shap_interaction_values=expshap.shap_interaction_values(x_train0)
    shap.summary_plot(shap_interaction_values,x_train0,show = False)
    plt.savefig(str(Model) + "shap_interaction.png")
    plt.close()
    shap.summary_plot(shap_interaction_values, x_train0,max_display=20,plot_type="compact_dot",show = False)
    plt.savefig(str(Model) + "shap_interaction_compact.png")
    plt.close()'''







