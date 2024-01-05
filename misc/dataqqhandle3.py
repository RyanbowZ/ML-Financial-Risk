import numpy as np
import pandas as pd
import tushare as ts
import config
import os

filepath="D:/Documents/毕业设计/data/qq/沽/当月"

ref=pd.read_csv("qqname4.csv",index_col="index")["price"]
#print(ref)
#print(ref["10000003.SH"])
data=[]
i=0
for root, dirs, files in os.walk(filepath):
    # print(root)
    # print(dirs)
    print(files)
    for filename in files:
        ee = pd.read_csv(filepath+"/"+filename)[["trade_date", "ts_code", "open", "settle", "close", "high", "low", "vol", "amount","oi"]]
        for i in range(0,len(ee)):
            e=ee.iloc[i]

            data.append({"trade_date":e["trade_date"],"ts_code":e["ts_code"],"price":ref[e["ts_code"]],"open":e["open"],"settle":e["settle"],"close":e["close"],"high":e["high"],"low":e["low"],
                     "vol":e["vol"],"amount":e["amount"],"oi":e["oi"]})

df=pd.DataFrame(data)
#df=df.reset_index(drop=True)
df.sort_values(by="trade_date", ascending=True)
df.to_csv("qqnameG5.csv",encoding='utf_8_sig')