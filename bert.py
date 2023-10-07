import pandas as pd
import numpy as np
import time
mark_score = pd.read_csv('mark_score.csv')
def train_test_split(mark_score):
    train_data = mark_score[mark_score['Date'] <= '2021-12-31'].reset_index(drop=True)
    test_data = mark_score[mark_score['Date'] >= '2022-01-01'].reset_index(drop=True)
    return train_data, test_data
train_data, test_data = train_test_split(mark_score)
from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') # 
model = BertModel.from_pretrained("bert-large-uncased")
train_data['newsSummary'] = train_data['newsSummary'].apply(lambda x: x[:250])
test_data['newsSummary'] = test_data['newsSummary'].apply(lambda x: x[:250])
bert_train_matrix = [[] for _ in range(len(train_data))]
bert_test_matrix = [[] for _ in range(len(test_data))]
for _ in range(len(train_data)):
    encoded_input = tokenizer(train_data.newsSummary[_], return_tensors='pt')
    output = model(**encoded_input)
    bert_train_matrix[_] = output.last_hidden_state[0][1:-1].mean(axis=0).tolist()
    bert_train_matrix[_].append(train_data['newsId'][_])
    if _ % 1000 == 0:
        jindu = _ / len(train_data)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'train', jindu)
for _ in range(len(test_data)):
    encoded_input = tokenizer(test_data.newsSummary[_], return_tensors='pt')
    output = model(**encoded_input)
    bert_test_matrix[_] = output.last_hidden_state[0][1:-1].mean(axis=0).tolist()
    bert_test_matrix[_].append(test_data['newsId'][_])
    if _ % 1000 == 0:
        jindu = _ / len(test_data)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'test', jindu)
bert_train_matrix = pd.DataFrame(bert_train_matrix)
bert_test_matrix = pd.DataFrame(bert_test_matrix)
bert_train_matrix.to_csv('bert_train_matrix.csv', index=False)
bert_test_matrix.to_csv('bert_test_matrix.csv', index=False)