# coding=utf-8
import sys
sys.path.append('../')
import re
import json, pickle
import os
import argparse
from tools import official_tokenization as tokenization



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

tokenizer = tokenization.BertTokenizer(vocab_file=r'./check_points/prev_trained_model/roberta_wwm_ext_large/vocab.txt', do_lower_case=True)
key_ratio = 0.3#关键词占整句的比例，如果有关键句，则只用它做input，否则用全句做input

def feature_to_label(feature_dir,n_class):#convert input features to labels
    in_f = pickle.load(open(feature_dir, 'rb'))
    input_ids = []#[n,4,seq_len]
    input_ids_cut = []#[n, 4, cutnum, len]
    input_mask = []#[n,4,seq_len],还要mask掉不关键信息
    segment_ids = []#[n,4,seq_len]
    label_id = []

    features = [[]]
    print("#features", len(in_f))
    #要从input_ids中context部分选择与问题相关的关键部分。先对examples分句。
    # 输入一句句子的id列表，根据标点断句。
    cut_tokens = [511, 136, 106, 8039, 8043, 8013, 132]
    # 511 for 。, 136 for ?, 8043 for ？, 106 for !, 8013 for ！, 8039 for ；,132 for ; .
    for f in in_f:#InputFeatures, 102 for [SEP], 101 for [CLS], 102 for [MASK].
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        label = f[0].label_id# 正确答案label
        if 102 in f[label].input_ids:
            qa_ind = f[label].input_ids.index(102) + 1  # SEP index + 1
        else:
            qa_ind = -1  # SEP index + 1
        qa = f[label].input_ids[qa_ind:]  # 问题与真实答案
        #print("Answer: ", tokenizer.convert_ids_to_tokens(qa))
        #print(label)
        context = f[label].input_ids[:qa_ind-1]#去掉最后一个[SEP]
        # 在context中找到qa中出现的词，训练一个SEQ2SEQ分类器。

        key_cnt = 0
        new_context = cut_list(context, cut_tokens)
        #print("Rawinput is cutted into %d pieces." % len(new_context))
        cnt_piece = [0] * len(new_context)  # 记录每个句子的关键词数
        context_mask = []
        for ind in range(len(new_context)):
            for word in new_context[ind]:
                if word in qa:  # 句子中词在qa中出现，说明为要找的关键词
                    cnt_piece[ind] += 1  # 该句关键词数+1
            cnt_piece[ind] /= len(new_context[ind])  # 计算比例
            #print(tokenizer.convert_ids_to_tokens(new_context[ind]))
            if cnt_piece[ind] > key_ratio:  # 句中含关键词比例达到thres
                key_cnt += 1
                #print("true")
                context_mask.extend([1] * len(new_context[ind]))  # 转为全1 mask
            else:
                context_mask.extend([0] * len(new_context[ind]))  # 转为全0 mask
        if key_cnt<3:#至少抽取出两句关键句时再进行mask，否则关键句太少，原样输出
            context_mask=[1]*len(context_mask)
        #print(context_mask)

        for i in range(n_class):
            tmp_mask = f[i].segment_ids  # [0 for context, 1 for question and choice, 0 for padding]

            tmp_mask[:len(context_mask)] = context_mask
            #print(tmp_mask)

            in_fea=InputFeatures(
                input_ids=f[i].input_ids,
                input_mask=tmp_mask,
                segment_ids=f[i].segment_ids,
                label_id=f[0].label_id)
            features[-1].append(in_fea)
            if len(features[-1]) == n_class:
                features.append([])
    if len(features[-1]) == 0:
        features = features[:-1]
    return features#[n,4],InputFeature



def cut_list(lst,tokens):#按tokens将lst切分为子lst
    new_lst=[]
    last_token=0
    for i in range(len(lst)):
        if lst[i] in tokens:#切分token出现
            new_lst.append(lst[last_token:i+1])
            last_token = i+1
    if last_token<len(lst):
        new_lst.append(lst[last_token:])
    return new_lst



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir",
                        default='./mrc_data/c3/',
                        type=str)
    parser.add_argument("--pkl_name",
                        default='train_features512',
                        type=str)
    args = parser.parse_args()
    train_features = feature_to_label(os.path.join(args.feature_dir,args.pkl_name+'.pkl'), 4)

    print('input id: ',train_features[-1][2].input_ids,'\ninput mask: ',train_features[-1][2].input_mask,'\nlabel: ',train_features[0][2].label_id)
    print("feature shape: (",len(train_features),len(train_features[-1]),")")
    #save
    with open(os.path.join(args.feature_dir,args.pkl_name+'_key.pkl'), 'wb') as w:
        pickle.dump(train_features, w)