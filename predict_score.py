import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
model = model.eval()
mark_score = pd.read_csv("mark_score.csv").dropna()

def train_test_split(mark_score):
    train_data = mark_score[mark_score['Date'] <= '2020-12-31'].reset_index(drop=True)
    validation_data = mark_score[(mark_score['Date'] >= '2021-01-01') & (mark_score['Date'] <= '2021-12-31')].reset_index(drop=True)
    test_data = mark_score[mark_score['Date'] >= '2022-01-01'].reset_index(drop=True)
    return train_data, validation_data, test_data
train_data, validation_data, test_data = train_test_split(mark_score)

prompt_1 = """

文本分类任务：忘记你之前的指示。假装你是个金融专家，一个有股票推荐经验的财务专家。将一段新闻进行分类，分成'好消息'或者'坏消息'或者'不确定'。

下面是一些范例:

这家公司净利润增长很快 -> 好消息
这家公司快不行了  -> 坏消息
已有信息不足以判断 -> 不确定

请对下述新闻进行分类。返回'好消息'或者'坏消息'或者'不确定'，无需其它说明和解释。

xxxxxx ->

"""
prompt_2 = """

文本分类任务：忘记你之前的指示。假装你是个金融专家，一个有股票推荐经验的财务专家。将一段新闻进行分类，新闻从好消息到坏消息，分成'5'或者'4'或者'3'或者'2'或者'1'或者不确定。

下面是一些范例:

这家公司发展的非常好 -> 5
这家公司发展的较好 -> 4 
这家公司发展的一般 -> 3
这家公司发展的较差 -> 2
这家公司发展的非常差  -> 1
已有信息不足以判断 -> 不确定

请对下述新闻进行分类。返回'5'或者'4'或者'3'或者'2'或者'1'或者不确定，无需其它说明和解释。

xxxxxx ->

"""

def get_prompt(prompt,text):
    return prompt.replace('xxxxxx',text)
            
def get_predict_1(test_data):
    history = []
    for response_1, history in model.stream_chat(tokenizer, get_prompt(prompt_1, test_data.newsSummary), history=history, temperature=1):
        if len(response_1) > 1:
            break
    if test_data.name % 10000 == 0:
        print(f'{time.strftime("%c")} predict_1 ->', test_data.name)
    return  response_1
def get_predict_2(test_data):
    history = []
    for response_2, history in model.stream_chat(tokenizer, get_prompt(prompt_2, test_data.newsSummary), history=history, temperature=1):
        if len(response_2) > 1:
            break
    if test_data.name % 10000 == 0:
        print(f'{time.strftime("%c")} predict_2 ->', test_data.name)
    return  response_2
def false_output_1(test_data):
    false_part = test_data[test_data['predict_1'].apply(lambda x: x not in ['好消息', '坏消息', '不确定'])]
    adjust_result = false_part.apply(get_predict_1, axis=1)
    return adjust_result
def false_output_2(test_data):
    false_part = test_data[test_data['predict_2'].apply(lambda x: x not in ['5', '4', '3', '2', '1', '不确定'])]
    adjust_result = false_part.apply(get_predict_2, axis=1)
    return adjust_result

test_data['predict_1'] = test_data.apply(get_predict_1, axis=1)
test_data['predict_2'] = test_data.apply(get_predict_2, axis=1)

for _ in range(10):
    adjust_result_1 = false_output_1(test_data)
    test_data.loc[adjust_result_1.index, 'predict_1'] = adjust_result_1
    adjust_result_2 = false_output_2(test_data)
    test_data.loc[adjust_result_2.index, 'predict_2'] = adjust_result_2
    print(f'[+] {time.strftime("%c")}第{_}次调整')

test_data.to_csv('test_data_predict.csv', index=False)
