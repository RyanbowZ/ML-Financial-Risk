import numpy as np
import pandas as pd
import tushare as ts
import config
import os

qq=pd.read_csv('D:\Documents\毕业设计\data\daily\qqs.csv')
etf=pd.read_csv('50etf.csv',index_col='trade_date')

list=[]
for i in range(len(qq)):
    c=qq.iloc[i]
    #print(etf.loc[int(c['trade_date'])]['close'])
    #print(c['price'])
    list.append(etf.loc[int(c['trade_date'])]['close']-c['price'])
qq['jc']=list
qq.to_csv('qqname8.csv',encoding='utf_8_sig')