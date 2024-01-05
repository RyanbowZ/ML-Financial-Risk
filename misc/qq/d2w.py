import pandas as pd

qqd=pd.read_csv("yhat_503csv.csv",index_col="trade_date")
#qqd=pd.read_csv("yhat_503csv.csv")

t=qqd.index

l=[]
'''while i < len(qqd):
   l.append(qqd.iloc[i])
   i+=5'''
lm="02"
for i in range(len(qqd)):
   year=str(t[i])[2:4]
   month=str(t[i])[4:6]
   c=qqd.iloc[i]
   if month != lm:
      l.append(
         {'time': year + month, 'price': c["0"], 'settle':  c["1"], 'per':  c["2"], 'jc':  c["3"],
      'gse':  c["4"],'iv': c["5"],'cjl': c["6"],'cje': c["7"],'ccl': c["8"]})
   lm=month
l.append(
         {'time': year + month, 'price': c["0"], 'settle':  c["1"], 'per':  c["2"], 'jc':  c["3"],
      'gse':  c["4"],'iv': c["5"],'cjl': c["6"],'cje': c["7"],'ccl': c["8"]})
pd.DataFrame(l).to_csv("yhat_month.csv")

