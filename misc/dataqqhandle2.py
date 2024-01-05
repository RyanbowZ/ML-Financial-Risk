import numpy as np
import pandas as pd
import tushare as ts
import config
import os

pro = ts.pro_api(config.tk)

da=pd.read_csv("qqname4.csv")[["index","price","st","ed","type"]]

for i in range(1001,1655):
    c=da.iloc[i]
    df = pro.opt_daily(ts_code=str(c["index"]))
    df = df.sort_values(by="trade_date", ascending=True)
    ty="隔季月"
    date=int(str(c["ed"])[:6])-int(str(c["st"])[:6])
    if date<=1 or (date>20 and date==89):
        ty="当月"
    elif date<=2 or (date>20 and date==90):
        ty="下月"
    elif date<5 or (date>20 and date<93):
        ty="季月"
    filepath="D:/Documents/毕业设计/data/qq/"+c["type"]+"/"+ty+"/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    df.to_csv(filepath+str(c["st"])+".csv",encoding='utf_8_sig')

