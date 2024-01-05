import numpy as np
import pandas as pd
import tushare as ts
import config
import os
import datetime
from scipy import stats
from math import log, sqrt, exp
qq=pd.read_csv('qq3forlstm.csv')

nr=[]
ns=[]
ne=[]
nj=[]
nd=[]
ng=[]
nq=[]
nv=[]

a=qq.price
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normprice']=nr
nr=[]
a=qq.settle
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normsettle']=nr
nr=[]
a=qq.per
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normper']=nr
nr=[]
a=qq.jc
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normjc']=nr
nr=[]
a=qq.dt
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normdt']=nr
nr=[]
a=qq.g_settle
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normgse']=nr
nr=[]
a=qq.q
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normq']=nr
nr=[]
a=qq.iv
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qq['normiv']=nr

qq.to_csv('q4lstm.csv')