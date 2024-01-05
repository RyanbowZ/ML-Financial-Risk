import numpy as np
import pandas as pd
import tushare as ts
import config
import os
import datetime
from scipy import stats
from math import log, sqrt, exp
qq=pd.read_csv('yhat2.csv')
f=pd.read_csv("qq44lstm.csv",index_col="trade_date")
qq['date']=np.unique(f.index)

qq.to_csv('yhat3.csv')