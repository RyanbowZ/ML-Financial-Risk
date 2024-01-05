import numpy as np
import pandas as pd
import tushare as ts
import config
import os
import datetime
from scipy import stats
from math import log, sqrt, exp
qq=pd.read_csv('qq1.csv')

#pro = ts.pro_api(config.tk)
listq=[]
listiv=[]
def bsm_imp_vol(s, k, t, r, c, q, option_type):
    c_est = 0  # 期权价格估计值
    top = 1  # 波动率上限
    floor = 0  # 波动率下限
    sigma = (floor + top) / 2  # 波动率初始值
    count = 0  # 计数器
    while abs(c - c_est) > 0.000001:
        c_est = bsm_value(s, k, t, r, sigma, q, option_type)
        # 根据价格判断波动率是被低估还是高估，并对波动率做修正
        count += 1
        if count > 100:  # 时间价值为0的期权是算不出隐含波动率的，因此迭代到一定次数就不再迭代了
            sigma = 0
            break

        if c - c_est > 0:  # f(x)>0
            floor = sigma
            sigma = (sigma + top) / 2
        else:
            top = sigma
            sigma = (sigma + floor) / 2
    return sigma

def bsm_value(s, k, t, r, sigma, q, option_type):
    d1 = (log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    d2 = (log(s / k) + (r - q - 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    if option_type.lower() == 'c':
        value = (s * exp(-q * t) * stats.norm.cdf(d1) - k * exp(-r * t) * stats.norm.cdf(d2))
    else:
        value = k * exp(-r * t) * stats.norm.cdf(-d2) - s * exp(-q * t) * stats.norm.cdf(-d1)
    return value

for i in range(len(qq)):
    o=qq.iloc[i]
    c=o['settle']
    cp=o['g_settle']
    k=o['price']
    s=k+o['jc']
    r=o['shibor']/100
    t=o['dt']/365
    q = -log((c + k * exp(-r * t) - cp) / (s)) / t
    if np.isnan(q) or np.isinf(q):
        q=0
    listq.append(q)
    sigma = bsm_imp_vol(s, k, t, r, c, q, 'c')
    listiv.append(sigma)
qq['q']=listq
qq['iv']=listiv

qq.to_csv('qq2.csv')