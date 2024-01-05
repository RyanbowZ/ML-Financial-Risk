import tushare as ts
import config

pro = ts.pro_api(config.tk)

#df = pro.opt_basic(exchange='SSE', fields='ts_code,name,opt_code, opt_type, exercise_price,list_date,delist_date')

#df.to_csv('qqname2.csv', encoding='utf_8_sig')

df=pro.cn_m(start_m='201501', end_m='202303')
ds=pro.cn_gdp(start_m='2015Q1', end_m='2023Q1')
df.to_csv('m.csv', encoding='utf_8_sig')
ds.to_csv('GDP.csv', encoding='utf_8_sig')