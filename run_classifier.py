# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']="2"
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
'''
自己导入
'''
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig, BertModel
import torch.nn as nn
from sklearn.metrics import confusion_matrix,classification_report,precision_recall_fscore_support,precision_score,recall_score,f1_score,accuracy_score
from myModel.modeling import BertPooler,ResNet

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            linefile = f.readlines()
            for line in linefile[:3]:
            # for line in linefile:
                lines.append(line.strip().split("\t"))

            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            # lines = []
            # for line in reader:
            #     if sys.version_info[0] == 2:
            #         line = list(cell for cell in line)
            #     lines.append(line)
            # print(len(lines))
            return lines

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return ['游戏', '散文', '孩子', '养生', '时尚', '历史', '军事', '世界', '社会', '旅游', '食物', '探索', '汽车', '娱乐', '金融', '故事', '科技', '运动']
        # return ['新闻', '旅游', '教育', '科技', '文化', '汽车', '健康', '财经', '女人', '体育', '娱乐', '房地产']
        # return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
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

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    # if task_name == "cola":
    #     return {"mcc": matthews_corrcoef(labels, preds)},confusion_matrix(labels,preds)
    return {"mcc": matthews_corrcoef(labels, preds)},classification_report(labels,preds),accuracy_score(labels,preds)
    # elif task_name == "sst-2":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "mrpc":
    #     return acc_and_f1(preds, labels)
    # elif task_name == "sts-b":
    #     return pearson_and_spearman(preds, labels)
    # elif task_name == "qqp":
    #     return acc_and_f1(preds, labels)
    # elif task_name == "mnli":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "mnli-mm":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "qnli":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "rte":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "wnli":
    #     return {"acc": simple_accuracy(preds, labels)}
    # else:
    #     raise KeyError(task_name)


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels,in_channels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels,in_channels,kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.maxpooling = nn.MaxPool1d(2,stride=2)
        self.pooler = BertPooler(config)
        self.avgpool3 = nn.AvgPool1d(3, stride=3)
        self.liner = nn.Linear(64*768, 768)
        self.lstm = nn.LSTM(768,384,batch_first=True,bias=True,dropout=0.1,bidirectional=True)
        self.lstmLiners = nn.Sequential(
            nn.Linear(64 * 768, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128,num_labels)
        )
        self.resnet = ResNet()



    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        # encoder = torch.zeros_like(encoded_layers[-1])
        # for i in range(len(encoded_layers)):
        #     encoder += encoded_layers[i]*0.1*(i+1)
        # encoder = encoder/7.8
        #
        # pooled_output = self.pooler(encoder)
        '''
            原始方法
        '''
        pooled_output = self.dropout(pooled_output)
        # # pooled_output = self.pooler(pooled_output)
        logits = self.classifier(pooled_output)
        '''
        conv  3  1   残差连接   3层
        '''
        # encoded_layers = self.dropout(encoded_layers)
        # conv1 = encoded_layers
        # for _ in range(3):
        #     conv2 = self.conv1(conv1)
        #     conv1 = torch.cat((conv1,conv2),2)
        #     conv1 = self.maxpooling(conv1)

        '''
        conv 1+3+5 拼接 做2次
        '''
        # encoded_layers = self.dropout(encoded_layers)
        # conv1 = encoded_layers
        # for _ in range(2):
        #     conv2 = self.conv1(conv1)
        #     conv3 = self.conv3(conv1)
        #     conv4 = self.conv5(conv1)
        #     conv1 = torch.cat((conv2,conv3,conv4),2)
        #     conv1 = self.avgpool3(conv1)
        # # pooled_output = self.pooler(conv1)
        # flatten = conv1.view(encoded_layers.size()[0],-1)
        # logits = self.liner(flatten)
        # logits = self.classifier(logits)

        '''
        lstm
        '''
        # lstmoutput = self.dropout(encoded_layers)
        # #for _ in range(3):
        # lstmoutput, _ = self.lstm(lstmoutput)
        # flatten = lstmoutput.reshape(lstmoutput.size()[0], -1)
        # logits = self.lstmLiners(flatten)
        '''
        resNet
        '''
        #encoded_layers = self.dropout(encoded_layers)
        #encoded_layers = encoded_layers.permute(0,2,1)
        #logits = self.resnet(encoded_layers)
        #logits = self.classifier(logits)



        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--data_dir",default="trainData/souhu",type=str,required=False,help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--data_dir",default="trainData/legal",type=str,required=False,help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir",default="trainData/NLPCC2017",type=str,required=False,help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-model-pretrain/bert-base-chinese", type=str, required=False,help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",default="COLA",type=str,required=False,help="The name of the task to train.")
    # parser.add_argument("--output_dir",default="output",type=str,required=False,help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--output_dir",default="output1",type=str,required=False,help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--output_dir",default="output11",type=str,required=False,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir",default="output111111112",type=str,required=False,help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--do_train",default=True,action='store_true',help="Whether to run training.")
    parser.add_argument("--do_eval",default=True,action='store_true',help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",default=True,action='store_true',help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length",default=64,type=int,help="The maximum total input sequence length after WordPiece tokenization. \nSequences longer than this will be truncated, and sequences shorter \nthan this will be padded.")
    parser.add_argument("--train_batch_size",default=64,type=int,help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",default=32,type=int,help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",default=5.0,type=float,help="Total number of training epochs to perform.")
    ## Other parameters
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",action='store_true',help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',type=int,default=42,help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',action='store_true',help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',type=float, default=0,help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n 0 (default value): dynamic loss scaling.\nPositive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    processors = {"cola": ColaProcessor,}
    output_modes = { "cola": "classification",}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
                        # level =logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,num_labels=num_labels,in_channels=args.max_seq_length)
    if args.fp16:
        model.half()
    model.to(device)

    # print(model)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        #训练数据集
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        #测试数据集
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids1 = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask1 = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids1 = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_label_ids1 = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1, all_label_ids1)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)



        model.train()
        # for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for epoch in range(int(args.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                # print('batch_loss: {:.3f} batch'.format(loss.data))
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()

                #测试
                if step % 500 ==0:
                    print('epoch:{:10} step {:10} batch_loss: {:.3f} '.format(epoch,step,loss.data))
                if (step+1) % 1000 == 0:

                    eval_loss = 0
                    nb_eval_steps = 0
                    preds = []
                    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        with torch.no_grad():
                            logits = model(input_ids, segment_ids, input_mask, labels=None)
                        # create eval loss and other metric required by the task
                        loss_fct = CrossEntropyLoss()
                        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_steps += 1
                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(
                                preds[0], logits.detach().cpu().numpy(), axis=0)

                    preds = preds[0]
                    preds = np.argmax(preds, axis=1)
                    # f1 = classification_report(all_label_ids1.numpy(),preds)
                    acc = accuracy_score(preds, all_label_ids1.numpy())
                    # print(f1)
                    print("===========================================")
                    print(acc)


                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)


    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                              cache_dir=cache_dir, num_labels=num_labels,
                                                              in_channels=args.max_seq_length)
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels,in_channels=args.max_seq_length)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels,in_channels=args.max_seq_length)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result,classification_report_true,acc = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss
        print(classification_report_true)
        print(acc)


        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            # writer.write(classification_report_true)
            # writer.write(acc)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
