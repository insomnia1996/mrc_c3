# coding=utf-8
import json,re,os
import numpy as np

f_macbert = open(r'./check_points/c3/macbert_large/logits_test.txt', 'r', encoding='utf-8')
f_roberta = open(r'./check_points/c3/roberta_wwm_ext_large/logits_test.txt', 'r', encoding='utf-8')
f_nezha = open(r'./check_points/c3/nezha_wwm_large/logits_test.txt', 'r', encoding='utf-8')


test_len = 3892
logits_all=[]
for i in range(test_len):
    logits_macbert = f_macbert.readline().strip().split(' ')
    logits_nezha = f_nezha.readline().strip().split(' ')
    logits_roberta = f_roberta.readline().strip().split(' ')
    assert len(logits_macbert)==len(logits_nezha)
    assert len(logits_macbert)==len(logits_roberta)
    total_logits=[]
    for j in range(len(logits_macbert)):
        total_logits.append(np.sum([0.5*float(logits_macbert[j]),
                                     0.2*float(logits_roberta[j]),
                                     0.3*float(logits_nezha[j])]
                                    ))
    logits_all.append(total_logits)
print(len(logits_all))


submission_test = r'./check_points/c3/ensemble/c3_predict.json'
g = json.load(open(r'./mrc_data/c3/test.json', 'r', encoding='utf-8'))
idlst = []
for i in range(len(g)):
    for j in range(len(g[i][1])):
        idlst.append(g[i][1][j]['id'])

test_preds = [int(np.argmax(logits_)) for logits_ in logits_all]
assert len(idlst) == len(test_preds)

with open(submission_test, "w") as f:
    for l in range(len(idlst)):
        dic = {'id': idlst[l], 'label': test_preds[l]}
        f.write(json.dumps(dic) + '\n')
f.close()


