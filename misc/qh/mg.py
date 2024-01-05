import pandas as pd

x=pd.read_csv('xxt.csv')
y=pd.read_csv('qhy.csv')

pd.merge(x,y,on='time').to_csv('xxtmg.csv')