import pandas as pd

qqd=pd.read_csv("hhx0503.csv",index_col="time")
#qqd=pd.read_csv("yhat_503csv.csv")

t=qqd.index
ii=0
l=[]
while ii < len(qqd):
   l.append(qqd.iloc[ii])
   ii+=5
pd.DataFrame(l).to_csv("hh_week.csv")
l=[]
lm="03"
for i in range(len(qqd)):
   year=str(t[i])[0:2]
   month=str(t[i])[2:4]
   c=qqd.iloc[i]
   if month != lm:
      l.append(
         {'time': year + month, 'val':c["normval"],'buy':c['normbuy'],'sell':c['normsell'],'avg':c['normavg'],'per':c['normper']})
   lm=month
l.append(
         {'time': year + month, 'val':c["normval"],'buy':c['normbuy'],'sell':c['normsell'],'avg':c['normavg'],'per':c['normper']})
pd.DataFrame(l).to_csv("hh_month.csv")

