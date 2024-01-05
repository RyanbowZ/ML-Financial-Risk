import numpy as np
import pandas as pd
import tushare as ts
import config
import os

pro = ts.pro_api(config.tk)

df=pro.fund_daily(ts_code='510050.SH',start_date='20150205',end_date='20180411')

df.to_csv('50etf2.csv',encoding='utf_8_sig')