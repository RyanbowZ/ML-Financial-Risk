import numpy as np
import pandas as pd
import tushare as ts
import os
import datetime
from scipy import stats
from math import log, sqrt, exp

qh=pd.read_csv('qhw.csv')

nr=[]

a=qh.per
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qh['normper']=nr
nr=[]

a=qh.cjl
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qh['normcjl']=nr
nr=[]

a=qh.cje
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qh['normcje']=nr
nr=[]

a=qh.jc
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qh['normjc']=nr
nr=[]

a=qh.ccl
for x in a:
    x = float(x - a.mean())/a.std()
    nr.append(x)
qh['normccl']=nr
nr=[]

qh.to_csv('qhwnorm.csv')