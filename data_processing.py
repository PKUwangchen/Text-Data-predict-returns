import pymongo
import datetime
import pandas as pd 
import numpy as np
import time
################### news_processing
print(f'[+] {time.strftime("%c")} news_processing start')
# news_info
news_info_history = pd.read_csv('/public/news_data/history/news_info.csv')
news_info_22_8 = pd.read_csv('/public/news_data/20220801/news_info.csv')
news_info_22_9 = pd.read_csv('/public/news_data/20220901/news_info.csv')
news_info_22_10 = pd.read_csv('/public/news_data/20221001/news_info.csv')
news_info_22_11 = pd.read_csv('/public/news_data/20221101/news_info.csv')
news_info_22_12 = pd.read_csv('/public/news_data/20221201/news_info.csv')
news_info_23_1 = pd.read_csv('/public/news_data/20230101/news_info.csv')
news_info_23_2 = pd.read_csv('/public/news_data/20230201/news_info.csv')
news_info_23_3 = pd.read_csv('/public/news_data/20230301/news_info.csv')
news_info_23_4 = pd.read_csv('/public/news_data/20230401/news_info.csv')
news_info = pd.concat([news_info_history
                       , news_info_22_8
                       , news_info_22_9
                       , news_info_22_10
                       , news_info_22_11
                       , news_info_22_12
                       , news_info_23_1
                       , news_info_23_2
                       , news_info_23_3
                       , news_info_23_4], axis=0)

# news_company_label
news_company_label_history = pd.read_csv('/public/news_data/history/news_company_label.csv')
news_company_label_22_8 = pd.read_csv('/public/news_data/20220801/news_company_label.csv')
news_company_label_22_9 = pd.read_csv('/public/news_data/20220901/news_company_label.csv')
news_company_label_22_10 = pd.read_csv('/public/news_data/20221001/news_company_label.csv')
news_company_label_22_11 = pd.read_csv('/public/news_data/20221101/news_company_label.csv')
news_company_label_22_12 = pd.read_csv('/public/news_data/20221201/news_company_label.csv')
news_company_label_23_1 = pd.read_csv('/public/news_data/20230101/news_company_label.csv')
news_company_label_23_2 = pd.read_csv('/public/news_data/20230201/news_company_label.csv')
news_company_label_23_3 = pd.read_csv('/public/news_data/20230301/news_company_label.csv')
news_company_label_23_4 = pd.read_csv('/public/news_data/20230401/news_company_label.csv')
news_company_label = pd.concat([news_company_label_history
                                , news_company_label_22_8
                                , news_company_label_22_9
                                , news_company_label_22_10
                                , news_company_label_22_11
                                , news_company_label_22_12
                                , news_company_label_23_1
                                , news_company_label_23_2
                                , news_company_label_23_3
                                , news_company_label_23_4], axis=0)

news_test = news_info[['newsId', 'newsTitle', 'newsTs', 'newsSummary', 'emotionIndicator']]
company_test = news_company_label[['stockCode', 'chineseName', 'newsId', 'relevance']]
data = pd.merge(news_test, company_test, on='newsId')
remain_list = ['SZ', 'SH']
def filter_Stkcd(Stkcd):
    return Stkcd[:6] if Stkcd[-5:-3] in remain_list else 0
def Stkcd_ajust(data):
    test = data[data['stockCode'] != 'csf']
    test.loc[:, 'Stkcd'] = test['stockCode'].apply(filter_Stkcd)
    return test
Stkcd_ajust_data = Stkcd_ajust(data)
Stkcd_ajust_data = Stkcd_ajust_data[Stkcd_ajust_data['Stkcd'] != 0]
Stkcd_ajust_data = Stkcd_ajust_data.dropna().reset_index(drop=True)
Stkcd_ajust_data['time'] = pd.to_datetime(Stkcd_ajust_data['newsTs'])
Stkcd_ajust_data['Date'] = Stkcd_ajust_data['time'].dt.date
Stkcd_ajust_data['year'] = Stkcd_ajust_data['time'].dt.year
Stkcd_ajust_data['half_hour'] = Stkcd_ajust_data['time'].dt.floor('30T')
only_one_news = Stkcd_ajust_data.newsTitle.drop_duplicates().index
Stkcd_ajust_data = Stkcd_ajust_data[Stkcd_ajust_data.index.isin(only_one_news)].reset_index(drop=True)
relevance_ajust_data = Stkcd_ajust_data[Stkcd_ajust_data['relevance'] == 1]
news_data = relevance_ajust_data[['Stkcd', 'time', 'Date', 'chineseName', 'newsTitle', 'newsSummary','newsId', 'emotionIndicator']].reset_index(drop=True)
news_data.time = pd.to_datetime(news_data.time)
news_data.Date = pd.to_datetime(news_data.Date)
def ajust_date(x):
    if x.time.hour == 9 & x.time.minute < 30:
        x.Date = x.Date - pd.offsets.Day(1)
    elif x.time.hour < 9:
        x.Date = x.Date - pd.offsets.Day(1)
    return x
news_data = news_data.apply(ajust_date, axis=1)
news_data = news_data[['Stkcd', 'Date', 'time', 'newsTitle', 'newsSummary', 'newsId', 'emotionIndicator']]
news_data['Stkcd'] = news_data['Stkcd'].astype(int)

################### stock_processing
print(f'[+] {time.strftime("%c")} stock_processing start')
def fetch_data(start_date, end_date, collection, time_query_key='TRADE_DT', factor_ls=None):
    """
    从数据库中读取需要指定日期范围的数据,包含startdate，包含enddate

    start_date:which date your need factors, str,'1991-01-01'
    end_date:str,'1991-01-01'
    collection: select collection after connecting to mongodb, such as: 'TRADE_DT'
    time_query_key: str, time key name for query database
    save_list: list with variable your need, make sure your variable is right,
                default= 'all',get all data


    """
    if end_date is not None:
        # 将end-date延后一天，以便形成闭区间
        end_date = (pd.to_datetime(end_date) + pd.Timedelta(1, unit='d')).strftime('%Y-%m-%d')
        query = {time_query_key: {"$gte": start_date, "$lte": end_date}}
    else:
        query = {time_query_key: {'$gte': start_date}}

    print('Querying......')

    if factor_ls is not None:
        fields = dict.fromkeys(factor_ls, 1)
        cursor = collection.find(query, fields)
    else:
        cursor = collection.find(query)
    data = pd.DataFrame.from_records(cursor)

    if len(data) != 0:
        data[time_query_key] = pd.to_datetime(data[time_query_key])
    return data
client = pymongo.MongoClient()
collection = client['basic_data']['Daily_return_with_cap']
data_return_raw = fetch_data('2013-12-30', '2023-05-05', collection, time_query_key='TRADE_DT')
stock_data = data_return_raw[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_OPEN', 'S_DQ_ADJFACTOR', 'S_DQ_VOLUME', 'S_DQ_CLOSE', 'FLOAT_A_SHR']]
stock_data['Mktvalue'] = stock_data['S_DQ_CLOSE'] * stock_data['FLOAT_A_SHR']
stock_data.rename(columns={'S_INFO_WINDCODE': 'Stkcd', 'TRADE_DT': 'Date'}, inplace=True)
stock_data.Stkcd = stock_data.Stkcd.apply(lambda x:x[:6]).astype(int)
stock_data.Date = pd.to_datetime(stock_data.Date)
stock_data['price'] = stock_data['S_DQ_ADJFACTOR'] * stock_data['S_DQ_OPEN']
pct_change = stock_data[['Date', 'price', 'Stkcd']].set_index(['Date', 'Stkcd']).groupby(level=1).pct_change().reset_index().rename(columns={'price': 'return_yesterday'})
pct_change['return'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(-1)
pct_change['return_tomorrow'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(-2)
pct_change['return_2day'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(-3)
pct_change['return_3day'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(-4)
pct_change['return_4day'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(-5)
pct_change['return_5day'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(-6)
pct_change['return_2day_ago'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(1)
pct_change['return_3day_ago'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(2)
pct_change['return_4day_ago'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(3)
pct_change['return_5day_ago'] = pct_change.groupby('Stkcd')['return_yesterday'].shift(4)
stock_data = stock_data.merge(pct_change, on=['Date', 'Stkcd'], how='outer')
stock_data = stock_data.set_index('Date').groupby('Stkcd').resample('d').bfill().droplevel(0).reset_index()
stock_data = stock_data[['Date', 'Stkcd', 'Mktvalue', 'return',	'return_yesterday',	'return_tomorrow',	'return_2day',	'return_3day',	'return_4day',	'return_5day',	'return_2day_ago',	'return_3day_ago',	'return_4day_ago',	'return_5day_ago']]
total_data = pd.merge(stock_data, news_data, on=['Stkcd', 'Date'], how='outer', indicator=True, validate='1:m')
total_data = total_data[total_data._merge == 'both'].dropna().reset_index(drop=True)

################### mark_score
print(f'[+] {time.strftime("%c")} mark_score start')
total_data.drop('_merge', axis=1, inplace=True)
total_data['label_1'] = total_data['return_tomorrow'].apply(lambda x: 1 if x > 0 else 0)
threshold_1 = 0.02 
threshold_2 = 0.06
def label_2(x):
        if x > threshold_2:
            return "very positive"
        elif x > threshold_1:
            return "positive"
        elif x < -threshold_2:
            return "very negative"
        elif x < -threshold_1:
            return "negative"
        else:
            return "neutral"
total_data['label_2'] = total_data['return_tomorrow'].apply(label_2)
mark_score = total_data
mark_score.to_csv('mark_score.csv', index=False)