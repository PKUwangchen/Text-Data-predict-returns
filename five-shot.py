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
label_good = [
    '平安银行：上半年实现净利润220.88亿元 同比增长26%#&#8月17日电，平安银行公告，上半年实现净利润220.88亿元，同比增长25.6%。'
    , '万科A：一季度净利润14.29亿元，同比增长10.58%#&#证券时报网讯，4月28日晚，万科A发布一季度财务报告，公司2022年一季度净利润14.29亿元，同比增长10.58%。'
    , '佰奥智能(300836.SZ)发布2022年三季报，前三季度实现营收为2.81亿元，同比增加34.02%，实现归母净利润为1200.18万元，同比增加11.08%。其中单三季度实现营收为5570.86万元，同比减少5.44%，实现归母净利润为-46.97万元，同比减少154.81%。'
]
label_bad = [
    '阿尔特公告，公司股东嘉兴珺文银宝投资合伙企业（有限合伙）计划以集中竞价交易或大宗交易方式减持公司股份不超过14,678,609股，减持比例不超过公司当前总股本剔除回购专户账户中的股份数量后的3%。'
    , '信濠光电(301051.SZ)发布2022年半年度报告，报告期内，公司实现营业收入7.54亿元，同比下降20.94%，归属于上市公司股东净亏损4764万元，同比由盈转亏，归属于上市公司股东的扣除非经常性损益后亏损7066万元，同比由盈转亏，基本每股收益为-0.60元。'
]
label_very_positive = [
    '指南针(300803.SZ)发布2022年第一季度报告，公司实现营业收入为6.23亿元，同比增长89.51%。归属于上市公司股东的净利润为2.67亿元，同比增长176.50%。归属于上市公司股东的扣除非经常性损益的净利润为2.64亿元，同比增长179.67%。基本每股收益为0.66元/股。'
]
label_positive = [
    '德石股份(301158.SZ)披露2021年年度报告，报告期内，公司实现营业收入4.45亿元，同比增长5.50%。归属于上市公司股东的净利润6240.54万元，同比增长3.62%。归属于上市公司股东的扣除非经常性损益的净利润5696.59万元，同比增长0.81%。经营活动现金净流入3819.56万元，同比增长38.01%。基本每股收益0.55元，年报推每10股派发现金红利1.20元(含税)。'
]
label_neutral = [
     '广和通(300638):公司及全资子公司向银行申请授信事宜 广和通:关于公司及全资子公司向银行申请授信事宜的公告证券代码：300638 证券简称：广和通 公告编号：2022-147深圳市广和通无线股份有限公司关于公司及全资子公司向银行申请授信事宜的公告本公司及董事会全体成员保证信息披露的内容真实、准确、完整，没有虚假 记载、误导性陈述或重大遗漏。深圳市广和通无线股份有限公司（以下简称“公司”）于2022年11月18日召开的第三届董事会第二十六次会议审议通过《关于公司及全资子公司向银行申请授信事宜的议案》，具体内容公告如下：为保证资金流动性，支持公司战略发展规划，公司董事会同意公司及全资子公司向商业银行申请不超过人民币25亿元的银行授信额度，具体融资币种、金额、期限、授信方式及用途等以公司及全资子公司与国内外商业银行签署的合同约定为准。授信期限为自公司股东大会审议通过之日起两年，上述授信额度可循环使用。授信额度不等于公司融资金额，实际融资金额以公司在授信额度内与银行实际发生的融资金额为准，具体授信明细以实际发生为准。董事会授权公司法定代表人签署上述授信额度内的一切有关合同、协议、凭证、抵押等各类法律文件。本次授信事项已经公司2022年11月18日召开的第三届董事会第二十六次会议审议通过，尚需提交公司股东大会审议。'
]
label_negative = [
    '科信技术(300565.SZ)公告，公司持股5%以上股东曾宪琦于2021年6月26日至2022年7月22日期间以集中竞价方式减持合计数量达到208万股(占公司总股本比例1.00%)。'
]
label_very_negative = [
'吉艾科技(300309.SZ)公布2022年第一季度报告，公司实现营业收入183.77万元，同比下降92.58%。归属于上市公司股东的净利润-1.12亿元。归属于上市公司股东的扣除非经常性损益的净利润-1.08亿元。基本每股收益-0.13元。'
]
def get_prompt(prompt,text):
    return prompt.replace('xxxxxx',text)

def get_few_shot_1(test_data):
    history = []
    history.append((label_good[0]+' -> ', '好消息'))
    history.append((label_good[1]+' -> ', '好消息'))
    history.append((label_good[2]+' -> ', '好消息'))
    history.append((label_bad[0]+' -> ', '坏消息'))
    history.append((label_bad[1]+' -> ', '坏消息'))
    for response_1, history in model.stream_chat(tokenizer, get_prompt(prompt_1, test_data.newsSummary), history=history, temperature=1):
        if len(response_1) > 1:
            break
    if test_data.name % 10000 == 0:
        print(f'{time.strftime("%c")} predict_1 ->', test_data.name)
    return  response_1
def get_few_shot_2(test_data):
    history = []
    history.append((label_very_positive[0]+' -> ', '5'))
    history.append((label_positive[0]+' -> ', '4'))
    history.append((label_neutral[0]+' -> ', '3'))
    history.append((label_negative[0]+' -> ', '2'))
    history.append((label_very_negative[0]+' -> ', '1'))
    for response_2, history in model.stream_chat(tokenizer, get_prompt(prompt_2, test_data.newsSummary), history=history, temperature=1):
        if len(response_2) > 1:
            break
    if test_data.name % 10000 == 0:
        print(f'{time.strftime("%c")} predict_2 ->', test_data.name)
    return  response_2
def false_output_1(test_data):
    false_part = test_data[test_data['predict_1'].apply(lambda x: x not in ['好消息', '坏消息', '不确定'])]
    adjust_result = false_part.apply(get_few_shot_1, axis=1)
    return adjust_result
def false_output_2(test_data):
    false_part = test_data[test_data['predict_2'].apply(lambda x: x not in ['5', '4', '3', '2', '1', '不确定'])]
    adjust_result = false_part.apply(get_few_shot_1, axis=1)
    return adjust_result

test_data['predict_1'] = test_data.apply(get_few_shot_1, axis=1)
test_data['predict_2'] = test_data.apply(get_few_shot_2, axis=1)

for _ in range(10):
    print(f'[+] {time.strftime("%c")}第{_+1}次调整')
    adjust_result_1 = false_output_1(test_data)
    test_data.loc[adjust_result_1.index, 'predict_1'] = adjust_result_1
    adjust_result_2 = false_output_2(test_data)
    test_data.loc[adjust_result_2.index, 'predict_2'] = adjust_result_2
    

test_data.to_csv('test_data_predict_fiveshot.csv', index=False)