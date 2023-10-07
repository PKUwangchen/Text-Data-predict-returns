import pandas as pd
import json
mark_score = pd.read_csv("/public/news_data/mark_score.csv").dropna()


prompt_2 = """
文本分类任务：忘记你之前的指示。假装你是个金融专家，一个有股票推荐经验的财务专家。请对下面的新闻基于其对公司影响的好坏进行判断，新闻从好消息到坏消息，分成"very positive"或者"positive"或者"neutral"或者"negative"或者"very negative"或者"not sure"。
下面是一些范例:
这家公司发展的非常好 -> "very positive"
这家公司发展的较好 -> "positive"
这家公司发展的一般 -> "neutral"
这家公司发展的较差 -> "negative"
这家公司发展的非常差  -> "very negative"
已有信息不足以判断 -> "not sure"
请对下述新闻进行分类。返回"very positive"或者"positive"或者"neutral"或者"negative"或者"very negative"或者"not sure"，无需其它说明和解释。

"""

def train_test_split(mark_score):
    train_data = mark_score[mark_score['Date'] <= '2020-12-31'].reset_index(drop=True)
    validation_data = mark_score[(mark_score['Date'] >= '2021-01-01') & (mark_score['Date'] <= '2021-12-31')].reset_index(drop=True)
    test_data = mark_score[mark_score['Date'] >= '2022-01-01'].reset_index(drop=True)
    return train_data, validation_data, test_data
train_data, validation_data, test_data = train_test_split(mark_score)



train_json = train_data.loc[:,['newsSummary','label_2']]

all_list = []
for i in range(len(train_json)):
    temp_dict = {}
    temp_dict["instruction"] = prompt_2
    temp_dict["input"] = train_json.iloc[i,0]
    temp_dict["output"] = train_json.iloc[i,1]
    all_list.append(temp_dict)

with  open('/public/news_data/ChatGLM-Efficient-Tuning/data/new_data_train.json', 'w',encoding='utf-8') as file:
    file.write(json.dumps(all_list, ensure_ascii=False,indent=4))   
    

train_json =  validation_data.loc[:,['newsSummary','label_2']]

all_list = []
for i in range(len(train_json)):
    temp_dict = {}
    temp_dict["instruction"] = prompt_2
    temp_dict["input"] = train_json.iloc[i,0]
    temp_dict["output"] = train_json.iloc[i,1]
    all_list.append(temp_dict)

with  open('/public/news_data/ChatGLM-Efficient-Tuning/data/new_data_val.json', 'w',encoding='utf-8') as file:
    file.write(json.dumps(all_list, ensure_ascii=False,indent=4))   
    
train_json =  test_data.loc[:,['newsSummary','label_2']]

all_list = []
for i in range(len(train_json)):
# for i in range(10):
    temp_dict = {}
    temp_dict["instruction"] = prompt_2
    temp_dict["input"] = train_json.iloc[i,0]
    temp_dict["output"] = train_json.iloc[i,1]
    all_list.append(temp_dict)


with  open('/public/news_data/ChatGLM-Efficient-Tuning/data/new_data_test.json', 'w',encoding='utf-8') as file:
    file.write(json.dumps(all_list, ensure_ascii=False,indent=4))   
    

