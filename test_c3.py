# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import logging
import os
import pickle
import random

import numpy as np
import torch
from google_albert_pytorch_modeling import AlbertConfig, AlbertForMultipleChoice
from pytorch_modeling import BertConfig, BertForMultipleChoice, ALBertConfig, ALBertForMultipleChoice
from tools import official_tokenization as tokenization
from tools import utils
from tools.pytorch_optimization import get_optimization, warmup_linear
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

n_class = 4
reverse_order = False
sa_step = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class c3Processor(DataProcessor):
    def __init__(self, data_dir):
        self.D = []
        self.data_dir = data_dir

        data = []
        with open(os.path.join(self.data_dir , "d-test.json"),"r", encoding="utf8") as f:
            data += json.load(f)

        for i in range(len(data)):
            for j in range(len(data[i][1])):
                    d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                    for k in range(len(data[i][1][j]["choice"])):
                        d += [data[i][1][j]["choice"][k].lower()]
                    for k in range(len(data[i][1][j]["choice"]), 4):
                        d += ['无效答案']  # 有些C3数据选项不足4个，添加[无效答案]能够有效增强模型收敛稳定性
                    self.D += [d]

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self.D, "test")


    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        cache_dir = os.path.join(self.data_dir, set_type + '_examples.pkl')
        if os.path.exists(cache_dir):
            examples = pickle.load(open(cache_dir, 'rb'))
        else:
            examples = []
            for (i, d) in enumerate(data):

                for k in range(4):
                    guid = "%s-%s-%s" % (set_type, i, k)
                    text_a = tokenization.convert_to_unicode(data[i][0])
                    text_b = tokenization.convert_to_unicode(data[i][k + 2])
                    text_c = tokenization.convert_to_unicode(data[i][1])
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c))

            with open(cache_dir, 'wb') as w:
                pickle.dump(examples, w)
        print("example length: ",len(examples))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    #features就是tokenid、mask等，尚未embedding


    features = [[]]#二维数组
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)

        tokens_c = tokenizer.tokenize(example.text_c)

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_c + ["[SEP]"] + tokens_b

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #TODO: 看features的结构，改bat
        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids))
        if len(features[-1]) == n_class:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--gpu_ids",
                        default='0',
                        type=str,
                        required=True)
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default='c3',
                        type=str,
                        required=True)
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default='check_points/pretrain_models/albert_xxlarge_google_zh_v1121/pytorch_model.pth',
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--test_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--schedule",
                        default='warmup_linear',
                        type=str,
                        help='schedule')
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--float16',
                        action='store_true',
                        default=False)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=422,
                        help="random seed for initialization")
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    args = parser.parse_args()
    args.setting_file = os.path.join(args.output_dir, args.setting_file)
    args.log_file = os.path.join(args.output_dir, args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))



    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    processor = c3Processor(args.data_dir)
    label_list = processor.get_labels()

    tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)


    if 'albert' in args.bert_config_file:
        if 'google' in args.bert_config_file:
            bert_config = AlbertConfig.from_json_file(args.bert_config_file)
            model = AlbertForMultipleChoice(bert_config, num_choices=n_class)
        else:
            bert_config = ALBertConfig.from_json_file(args.bert_config_file)
            model = ALBertForMultipleChoice(bert_config, num_choices=n_class)
    else:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        model = BertForMultipleChoice(bert_config, num_choices=n_class)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    if args.init_checkpoint is not None:
        utils.torch_show_all_params(model)
        utils.torch_init_model(model, args.init_checkpoint)
    if args.float16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    #elif n_gpu > 1:
    else:
        model = torch.nn.DataParallel(model)

    #load model params
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")),strict=False)

    test_dataloader = None
    if args.do_test:
        test_examples = processor.get_test_examples()
        feature_dir = os.path.join(args.data_dir, 'test_features{}.pkl'.format(args.max_seq_length))
        if os.path.exists(feature_dir):
            test_features = pickle.load(open(feature_dir, 'rb'))
        else:
            test_features = convert_examples_to_features(test_examples, args.max_seq_length, tokenizer)
            with open(feature_dir, 'wb') as w:
                pickle.dump(test_features, w)
        input_ids = []
        input_mask = []
        segment_ids = []


        for f in test_features:
            input_ids.append([])
            input_mask.append([])
            segment_ids.append([])
            for i in range(n_class):
                input_ids[-1].append(f[i].input_ids)
                input_mask[-1].append(f[i].input_mask)
                segment_ids[-1].append(f[i].segment_ids)


        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)


        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        if args.local_rank == -1:
            test_sampler = SequentialSampler(test_data)
        else:
            test_sampler = DistributedSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

        logger.info("***** Running test phase *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)

        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        logits_all = []
        for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                reshaped_logits = model(input_ids, segment_ids, input_mask, return_logits=True)

            logits = reshaped_logits.detach().cpu().numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]



            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1
        logger.info("***** Test results *****")
        output_test_file = os.path.join(args.output_dir, "results_test.txt")

        with open(output_test_file, "w") as f:
            for i in range(len(logits_all)):
                for j in range(len(logits_all[i])):
                    f.write(str(logits_all[i][j]))
                    if j == len(logits_all[i]) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")
        f.close()

        # the test submission order can't be changed
        submission_test = os.path.join(args.output_dir, "%s_predict.json" %(args.task_name))
        g=json.load(open(r'./mrc_data/c3/d-test.json','r',encoding='utf-8'))
        idlst=[]
        for i in range(len(g)):
            for j in range(len(g[i][1])):
                idlst.append(g[i][1][j]['id'])
        g.close()
        test_preds = [int(np.argmax(logits_)) for logits_ in logits_all]
        assert len(idlst)==len(test_preds)

        with open(submission_test, "w") as f:
            for l in range(len(idlst)):
                dic={'id':idlst[l], 'label':test_preds[l]}
                f.write(json.dumps(dic)+'\n')
        f.close()

if __name__ == "__main__":
    main()
