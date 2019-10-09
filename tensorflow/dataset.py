# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import mmap
import logging
import linecache
import numpy as np
from collections import Counter


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len, vocab,
                 train_files='', dev_files='', test_files=''):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files

        self.train_set, self.dev_set, self.test_set = [], [], []
        self.vocab = vocab
        # if train_files:
        #     for train_file in train_files:
        #         self.train_set += self._load_dataset(train_file, train=True)
        #     self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
        #
        # if dev_files:
        #     for dev_file in dev_files:
        #         self.dev_set += self._load_dataset(dev_file)
        #     self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))
        #
        # if test_files:
        #     for test_file in test_files:
        #         self.test_set += self._load_dataset(test_file)
        #     self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, lists, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        # 多文档多段落
        # data_set = []
        # for lidx, line in enumerate(lists):   # 对每一个样本（多文档多段落）
        #     sample = json.loads(line.strip())
        #     if train:
        #         if len(sample['answer_spans']) == 0:
        #             continue
        #         if sample['answer_spans'][0][1] >= self.max_p_len:  # 对选出来的【【15, 65】】，过滤掉大于最大长度的文档
        #             continue
        #
        #     if 'answer_docs' in sample:
        #         sample['answer_passages'] = sample['answer_docs']
        #
        #     sample['question_tokens'] = sample['segmented_question']
        #
        #     sample['passages'] = []     # 存的是index
        #     sample['para_label'] = []   # 存的是1 selected 0 not selected
        #     sample['answer_label'] = []
        #     for d_idx, doc in enumerate(sample['documents']):   # 对每一篇文档处理,如果是训练，直接用最相关的段落；否则使用问题进行计算找到最相关的段落
        #         # if not doc['is_selected']:                      # fixme:作弊
        #         #     continue
        #
        #         # # 对每个段落进行处理，使其按照相关排序截断到400
        #         # # 对每一个段落先按照句子划分；对每一个句子和问题计算相关度，选出top-10
        #         # for paragraph in doc['paragraphs']:
        #         #     sentences = (doc['paragraphs'][0]).split('。')
        #         #     sentences = list(filter(None, sentences))
        #         para_infos = []     # 存的是段落，段落和问题的common在问题长度的占比，以及段落的长度
        #         para_update = []    # 按照得分排序后，存成一个新的update，然后在新的里面，还原顺序
        #         for p_id, para_tokens in enumerate(doc['segmented_paragraphs']):  # 对一篇文章里的每一段(有的只有一段)
        #             question_tokens = sample['segmented_question']
        #             common_with_question = Counter(para_tokens) & Counter(question_tokens)
        #             correct_preds = sum(common_with_question.values())
        #             if correct_preds == 0:
        #                 recall_wrt_question = 0
        #             else:
        #                 recall_wrt_question = float(correct_preds) / len(question_tokens)
        #             para_infos.append((para_tokens, recall_wrt_question, p_id)) # fixme: 有可能最相关的文档，和问题计算的得分也比较低
        #         para_infos.sort(key=lambda x: (-x[1], x[2]))    # 按照第一个维度的降序，第二个维度的升
        #         fake_passage_tokens = []
        #         most_related = 2
        #         if len(para_infos) == 1:
        #             fake_passage_tokens.append(para_infos[0][0])
        #             if para_infos[0][2] == doc['most_related_para']:
        #                 most_related = para_infos[0][2]
        #             else:
        #                 most_related = 2
        #         else:
        #             for i in range(2):                      # fixme:选择几个段落
        #                 para_update.append(para_infos[i])
        #             para_update.sort(key=lambda x: x[2])    # 按照文章的段落顺序排序
        #             for para_info in para_update[:2]:       # 取出段落
        #                 fake_passage_tokens.append(para_info[0])
        #             for index, para_info in enumerate(para_update[:2]):
        #                 if para_info[2] == doc['most_related_para']:
        #                     most_related = index
        #                     break
        #                 else:
        #                     most_related = 2
        #         sample['passages'].append(fake_passage_tokens)  # 把最高的那个段落加到sample['passages']
        #         # 增加标签信息
        #         # np.eye(class)[y.reshape(-1)].T
        #         sample['para_label'].append(most_related)   #
        #
        #         if doc['is_selected']:
        #             sample['answer_label'].append(1)
        #         else:
        #             sample['answer_label'].append(0)
        #
        #     data_set.append(sample)
        # return data_set

        # 多文档单段落
        data_set = []
        for lidx, line in enumerate(lists):  # 对每一个样本（多文档多段落）
            sample = json.loads(line.strip())
            if train:
                if len(sample['answer_spans']) == 0:
                    continue
                if sample['answer_spans'][0][1] >= self.max_p_len:  # 对选出来的【【15, 65】】，过滤掉大于最大长度的文档
                    continue

            if 'answer_docs' in sample:
                sample['answer_passages'] = sample['answer_docs']

            sample['question_tokens'] = sample['segmented_question']

            sample['passages'] = []
            sample['answer_label'] = []
            for d_idx, doc in enumerate(sample['documents']):  # 对每一篇文档处理,如果是训练，直接用最相关的段落；否则使用问题进行计算找到最相关的段落
                # if not doc['is_selected']:                      # fixme:prepare的时候不需要
                #     continue
                if train:  # 把被选择的和未被选择的最相关的段落都加到sample['passages']
                    most_related_para = doc['most_related_para']
                    sample['passages'].append(doc['segmented_paragraphs'][most_related_para])

                else:
                    para_infos = []  # 存的是段落，段落和问题的common在问题长度的占比，以及段落的长度
                    for para_tokens in doc['segmented_paragraphs']:  # 对一篇文章里的每一段
                        question_tokens = sample['segmented_question']
                        common_with_question = Counter(para_tokens) & Counter(question_tokens)
                        correct_preds = sum(common_with_question.values())
                        if correct_preds == 0:
                            recall_wrt_question = 0
                        else:
                            recall_wrt_question = float(correct_preds) / len(question_tokens)
                        para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                    para_infos.sort(key=lambda x: (-x[1], x[2]))
                    fake_passage_tokens = []
                    for para_info in para_infos[:1]:  # 只取第一个最高的
                        fake_passage_tokens += para_info[0]
                    sample['passages'].append(fake_passage_tokens)  # 把最高的那个段落加到sample['passages']

                if doc['is_selected']:
                    sample['answer_label'].append(1)
                else:
                    sample['answer_label'].append(0)
            data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id, training):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        # i 不能等0，没有第0行
        batch_data = {'raw_data': [linecache.getline(data, i) for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'answer_label': [],
                      'answer_index': [],
                      'answer_loss': [],
                      'question_type': [],
                      'start_id': [],
                      'end_id': []}
        batch_data['raw_data'] = self._load_dataset(batch_data['raw_data'], training)
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):      # 对于每一个样例
            docs = []
            length = []
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    docs.append(self.vocab.convert_to_ids(sample['passages'][pidx]))
                    max_passage_len = min(len(sample['passages'][pidx]), self.max_p_len)
                    length.append(max_passage_len)
                else:
                    docs.append([])
                    length.append(0)

            batch_data['passage_token_ids'].append(docs)
            # max_passage_len = max([len(i) for i in docs])
            # max_passage_len = min(max_passage_len, self.max_p_len)
            batch_data['passage_length'].append(length)

            sample['question_token_ids'] = self.vocab.convert_to_ids(sample['question_tokens'])
            batch_data['question_token_ids'].append(sample['question_token_ids'])
            batch_data['question_length'].append(len(sample['question_token_ids']))

            batch_data['question_type'].append(sample['question_type'])
            batch_data['answer_label'].append(sample['answer_label'])
            if training:
                # 用于tf.gather_nd
                batch_data['answer_index'].append([sidx, sample['answer_passages'][0]])
                # 用于计算doc loss
                batch_data['answer_loss'].extend(sample['answer_passages'])
            elif len(sample['answer_passages']) != 0:
                batch_data['answer_index'].append([sidx, sample['answer_passages'][0]])
                batch_data['answer_loss'].extend(sample['answer_passages'])
            else:
                batch_data['answer_index'].append([sidx, 0])
                batch_data['answer_loss'].extend([0])
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id, training)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                # gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                # batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                # batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
                batch_data['start_id'].append(sample['answer_spans'][0][0])
                batch_data['end_id'].append(sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id, training):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max([max(sample) for sample in batch_data['passage_length']]))  # fixme
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        # if training:
        pad_a_len = min(5, max([len(i) for i in batch_data['answer_label']]))
        for id, sample in enumerate(batch_data['passage_token_ids']):
            batch_data['passage_token_ids'][id] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                                   for ids in sample]

        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        # if training:
        batch_data['answer_label'] = [(ids + [pad_id] * (pad_a_len - len(ids)))[: pad_a_len]
                                          for ids in batch_data['answer_label']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def mapcount(self, filename):
        f = open(filename, "r+")
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
        f.close()
        return lines

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True, training=False):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_files
        elif set_name == 'dev':
            data = self.dev_files
        elif set_name == 'test':
            data = self.test_files
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        # data_size = self.mapcount(data)
        data_size = len(linecache.getlines(data))
        indices = np.arange(1, data_size+1)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id, training)
