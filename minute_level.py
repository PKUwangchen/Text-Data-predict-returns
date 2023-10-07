import time, datetime, pytz
from clickhouse_driver import Client
import pandas as pd
import numpy as np
import tqdm

client = Client('10.8.3.37', user='quant', password='pbcsf888QUANT', port='9000', database='minute_level')
# 创建时区对象
timezone = pytz.timezone('Asia/Shanghai')

query = "SELECT name FROM system.tables WHERE database = 'minute_level'"
sheet_names = client.execute(query)
print(f'# of sheets: {len(sheet_names)}')
query = "select column_name from information_schema.columns where table_schema='minute_level' and table_name='SH600000_2023'"
col_names = client.execute(query)
col_names = [i[0] for i in col_names]

stock = pd.DataFrame(columns = col_names)
tmp_ls=[]
for names in tqdm.tqdm(sheet_names):
    if names[0][-4:] == '2023':
        query = f"SELECT * FROM {names[0]} WHERE dt > '2023-01-01' and dt<'2023-02-01'"
        result = client.execute(query)
        tmp = pd.DataFrame(result, columns=col_names)
        Stkcd = int(names[0][2:8])
        tmp['Stkcd'] = Stkcd
        tmp_ls.append(tmp)
stock = pd.concat(tmp_ls, axis=0)
stock['Date'] = pd.to_datetime(stock.dt.dt.date)
client = Client('10.8.3.37', user='quant', password='pbcsf888QUANT', port='9000', database='day_level')
# 创建时区对象
timezone = pytz.timezone('Asia/Shanghai')

query = "SELECT name FROM system.tables WHERE database = 'day_level'"
sheet_names = client.execute(query)
print(f'# of sheets: {len(sheet_names)}')
query = "select column_name from information_schema.columns where table_schema='day_level' and table_name='stock'"
col_names = client.execute(query)
col_names = [i[0] for i in col_names]
query = f"SELECT * FROM {sheet_names[2][0]} WHERE dt > '2023-01-01' and dt<'2023-02-01'"
result = client.execute(query)
tmp = pd.DataFrame(result, columns=col_names)
tmp['Stkcd'] = tmp['code'].apply(lambda x: x[:6]).astype(int) 
tmp.rename(columns={'dt': 'Date'}, inplace=True)
tmp['Date'] = pd.to_datetime(tmp['Date'])
stock = pd.merge(stock, tmp[['Date', 'Stkcd', 'adj_factor']], on=['Date', 'Stkcd'])
stock = stock.sort_values(['Stkcd', 'Date']).reset_index(drop=True)
stock['price'] = stock['close'] * stock['adj_factor']

close_to_open = stock.groupby(['Stkcd', 'Date']).apply(lambda x: x.iloc[[0, -1]]).reset_index(drop=True)
close_to_open['close_shift'] = close_to_open.groupby('Stkcd')['price'].shift(1)
close_to_open['close_to_open'] = close_to_open['open'] * close_to_open['adj_factor'] / close_to_open['close_shift'] - 1
open_to_now = stock.groupby(['Stkcd', 'Date']).apply(lambda x: x['price'] / (x.iloc[0]['close'] * x.iloc[0]['adj_factor']) - 1)
now_to_close = stock.groupby(['Stkcd', 'Date']).apply(lambda x: x.iloc[-1]['close'] / x['close'] - 1)
stock['open_to_now'] = open_to_now.values
stock['now_to_close'] = now_to_close.values
stock = pd.merge(stock, close_to_open[['Stkcd', 'dt', 'close_to_open']], on=['Stkcd', 'dt'], how='left')
stock['price_15'] = stock.groupby(['Stkcd', 'Date'])['price'].shift(-15)
stock['change_15'] = stock['price_15'] / stock['price'] - 1
stock['change_15_30'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(-15)
stock['change_30_45'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(-30)
stock['change_45_60'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(-45)
stock['change_60_75'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(-60)
stock['change_75_90'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(-75)
stock['change_90_105'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(-90)
stock['change_105_120'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(-105)
stock['15_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(15)
stock['15_30_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(30)
stock['30_45_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(45)
stock['45_60_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(60)
stock['60_75_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(75)
stock['75_90_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(90)
stock['90_105_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(105)
stock['105_120_change'] = stock.groupby(['Stkcd', 'Date'])['change_15'].shift(120)
stock = stock[['dt', 'Date', 'Stkcd', '105_120_change', '90_105_change', '75_90_change', '60_75_change', '45_60_change', '30_45_change', '15_30_change'
               , '15_change', 'change_15', 'change_15_30', 'change_30_45', 'change_45_60', 'change_60_75', 'change_75_90', 'change_90_105', 'change_105_120'
               , 'open_to_now', 'now_to_close', 'close_to_open']]
for _ in ['105_120_change', '90_105_change', '75_90_change', '60_75_change', '45_60_change', '30_45_change', '15_30_change', '15_change']:
    stock[_].fillna(stock['open_to_now'], inplace=True)
for _ in ['change_15', 'change_15_30', 'change_30_45', 'change_45_60', 'change_60_75', 'change_75_90', 'change_90_105', 'change_105_120']:
    stock[_].fillna(stock['now_to_close'], inplace=True)

mark_score = pd.read_csv('/public/news_data/mark_score.csv')
data = mark_score[(mark_score['Date']>='2023-01-01') & (mark_score['Date']<'2023-02-01')]
data['dt'] = pd.to_datetime(data.time.apply(lambda x: x[:-6]))
data['Date'] = pd.to_datetime(data.Date)
data = data[['dt', 'Stkcd', 'Mktvalue', 'return', 'time', 'newsTitle', 'newsSummary', 'newsId',
       'emotionIndicator', 'label_1', 'label_2']]
not_trade_hour_1 = data[data.dt.dt.hour >= 15]
not_trade_hour_2 = data[data.dt.dt.hour < 9]
not_trade_hour_3 = data[(data.dt.dt.hour == 9) & (data.dt.dt.minute <= 30)]
not_trade_hour = pd.concat([not_trade_hour_1, not_trade_hour_2, not_trade_hour_3])
trade_hour = data[~data.index.isin(not_trade_hour.index)]
import datetime
def adjust_time(dt):
    if dt.weekday() == 6: # 如果是周日
        dt = dt + datetime.timedelta(days=1) # 加1天变为周一
        dt = datetime.datetime(dt.year, dt.month, dt.day, 9, 31)
    elif dt.weekday() == 5: # 如果是周六
        dt = dt + datetime.timedelta(days=2) # 加2天变为周一
        dt = datetime.datetime(dt.year, dt.month, dt.day, 9, 31)
    elif (dt.weekday() == 4) & (dt.hour >= 15): # 如果是周五的15点以后
        dt = dt + datetime.timedelta(days=3)
        dt = datetime.datetime(dt.year, dt.month, dt.day, 9, 31)
    elif (dt.weekday() in [0, 1, 2, 3]) & (dt.hour >= 15): # 如果是周一到周四的15点以后
        dt = dt + datetime.timedelta(days=1) # 加1天变为第二天
        dt = datetime.datetime(dt.year, dt.month, dt.day, 9, 31)
    elif (dt.weekday() in [0, 1, 2, 3, 4]) & (dt.hour < 9):
        dt = datetime.datetime(dt.year, dt.month, dt.day, 9, 31)
    elif (dt.weekday() in [0, 1, 2, 3, 4]) & (dt.hour == 9) & (dt.minute <= 30):
        dt = datetime.datetime(dt.year, dt.month, dt.day, 9, 31)
    elif (dt.weekday() in [0, 1, 2, 3, 4]) & (dt.hour == 11) & (dt.minute >= 30):
        dt = datetime.datetime(dt.year, dt.month, dt.day, 13, 1)
    elif (dt.weekday() in [0, 1, 2, 3, 4]) & (dt.hour == 12):
        dt = datetime.datetime(dt.year, dt.month, dt.day, 13, 1)
    dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)         
    return dt
trade_hour['dt'] = trade_hour.dt.apply(adjust_time)
trade_hour = pd.merge(stock, trade_hour, on=['dt', 'Stkcd'])
trade_hour.to_csv('/public/news_data/trade_hour.csv', index=False)

not_trade_hour_stock = stock.groupby(['Stkcd', 'Date']).apply(lambda x: x.iloc[[0, -1]]).reset_index(drop=True)
for _ in ['105_120_change', '90_105_change', '75_90_change', '60_75_change', '45_60_change', '30_45_change', '15_30_change', '15_change']:
    not_trade_hour_stock[f'{_}_shift'] = not_trade_hour_stock.groupby('Stkcd')[_].shift(1)
not_trade_hour['dt'] = not_trade_hour.dt.apply(adjust_time)
not_trade_hour = pd.merge(not_trade_hour_stock, not_trade_hour, on=['dt', 'Stkcd'])
not_trade_hour.to_csv('/public/news_data/not_trade_hour.csv', index=False)