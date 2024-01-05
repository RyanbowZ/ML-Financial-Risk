import numpy as np
import pandas as pd
import tushare as ts
import config
import os
import datetime

qq=pd.read_csv('qq19.csv')
qn=pd.read_csv('shibor.csv',index_col='date')
#pro = ts.pro_api(config.tk)
list=[]
for i in range(len(qq)):
    c=qq.iloc[i]
    #df=pro.opt_basic(ts_code=c['ts_code'],fields='maturity_date')
    #print(df.iloc[0]['maturity_date'])
    #delta=datetime.datetime.strptime(str(qn.loc[c['ts_code']]['ed']), "%Y%m%d")-datetime.datetime.strptime(str(c['trade_date']), "%Y%m%d")
    #print(delta)
    #print(delta.days)
    list.append(qn.loc[c['trade_date']]['val'])
qq['shibor']=list
qq.to_csv('qq1.csv',encoding='utf_8_sig')
