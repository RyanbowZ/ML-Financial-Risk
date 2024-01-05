import pandas as pd

qhw=pd.read_csv('qhw.csv')
ccl=0
cje=0
cjl=0
jc=0
price=0
avg=0
lm=0
ct=0
l=[]
for i in range(len(qhw)):
    c=qhw.iloc[i]
    year=str(c['time'])[:2]
    month=str(c['time'])[2:4]
    if month!=lm and ct!=0:
        l.append({'time':year+month,'ccl':ccl,'cje':cje,'cjl':cjl,'avgprice':avg/ct,'jc':jc,'price':price})
        ccl = 0
        cje = 0
        cjl = 0
        avg=0
        ct=0
    ccl+=c['ccl']
    cje+=c['cje']
    cjl+=c['cjl']
    avg+=c['avgp']
    jc=c['jc']
    price=c['price']
    ct+=1
    lm=month
l.append({'time': year + month, 'ccl': ccl, 'cje': cje, 'cjl': cjl, 'avgprice': avg / ct, 'jc': jc, 'price': price})

pd.DataFrame(l).to_csv("qhm.csv")