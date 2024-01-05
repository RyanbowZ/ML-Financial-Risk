import pandas as pd
import numpy as np

qh=pd.read_csv("hhcon_0503.csv",index_col="time")
hs=pd.read_csv("qhy.csv",index_col="time")

r=pd.merge(qh,hs,on='time')
r.to_csv("merge_0503.csv")