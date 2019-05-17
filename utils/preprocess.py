###############################################################################
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
This module finds the most related paragraph of each document according to recall.
"""
import sys
import json
from collections import Counter


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched,一篇文档分好词的
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)  # 找两个集合里的重复部分，次数是较少的一边
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched，一个段落
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def find_best_question_match(doc, question, with_score=False):
    """
    For each document, find the paragraph that matches best to the question.
    Args:
        doc: The document object.
        question: The question tokens.
        with_score: If True then the match score will be returned,
            otherwise False.
    Returns:
        The index of the best match paragraph, if with_score=False,
        otherwise returns a tuple of the index of the best match paragraph
        and the match score of that paragraph.
    """
    most_related_para = -1
    max_related_score = 0
    most_related_para_len = 0
    for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
        if len(question) > 0:
            related_score = metric_max_over_ground_truths(recall,
                    para_tokens,
                    question)
        else:
            related_score = 0

        if related_score > max_related_score \
                or (related_score == max_related_score \
                and len(para_tokens) < most_related_para_len):
            most_related_para = p_idx
            max_related_score = related_score
            most_related_para_len = len(para_tokens)
    if most_related_para == -1:
        most_related_para = 0
    if with_score:
        return most_related_para, max_related_score
    return most_related_para


def find_fake_answer(sample):
    """
    For each document, finds the most related paragraph based on recall,
    then finds a span that maximize the f1_score compared with the gold answers
    and uses this span as a fake answer span
    Args:
        sample: a sample in the dataset
    Returns:
        None
    Raises:
        None
    """
    for doc in sample['documents']: # 对每一个文档
        most_related_para = -1
        most_related_para_len = 999999
        max_related_score = 0                                               # doc['segmented_paragraphs']二维的，是一篇文章的多个段落
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):   # para_tokens 对每一个段落
            if len(sample['segmented_answers']) > 0:                        # sample['segmented_answers'] 多个候选答案
                related_score = metric_max_over_ground_truths(recall,       # 对所有的答案都会计算一个得分，选最高的
                                                              para_tokens,
                                                              sample['segmented_answers'])
            else:
                continue
            if related_score > max_related_score \
                    or (related_score == max_related_score
                        and len(para_tokens) < most_related_para_len):
                most_related_para = p_idx
                most_related_para_len = len(para_tokens)
                max_related_score = related_score
        doc['most_related_para'] = most_related_para
    # 下面开始选答案来自那个文档，以及答案片段计算
    sample['answer_docs'] = []
    sample['answer_spans'] = []
    sample['fake_answers'] = []
    sample['match_scores'] = []

    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    answer_tokens = set()
    for segmented_answer in sample['segmented_answers']:    # 对于所有的候选答案生成一个set集合
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    for d_idx, doc in enumerate(sample['documents']):       # 对每一个文档
        if not doc['is_selected']:                          # 没有被选择的文档可以直接过滤掉
            continue
        if doc['most_related_para'] == -1:
            doc['most_related_para'] = 0
        most_related_para_tokens = doc['segmented_paragraphs'][doc['most_related_para']][:1000] # 选择文档里最相关的那个段落
        for start_tidx in range(len(most_related_para_tokens)): # 从前面开始，找到在答案集合里有的那个词。然后再从结尾开始，计算每一个子串的得分
            if most_related_para_tokens[start_tidx] not in answer_tokens:
                continue                                        # 计算很久，所有的子串都计算
            for end_tidx in range(len(most_related_para_tokens) - 1, start_tidx - 1, -1):
                span_tokens = most_related_para_tokens[start_tidx: end_tidx + 1]
                if len(sample['segmented_answers']) > 0:
                    match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                                sample['segmented_answers'])    # 计算的是最相关片段和答案F1值
                else:
                    match_score = 0
                if match_score == 0:
                    break
                if match_score > best_match_score:
                    best_match_d_idx = d_idx
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_fake_answer = ''.join(span_tokens)
    if best_match_score > 0:
        sample['answer_docs'].append(best_match_d_idx)
        sample['answer_spans'].append(best_match_span)
        sample['fake_answers'].append(best_fake_answer)
        sample['match_scores'].append(best_match_score)


if __name__ == '__main__':
    # for line in sys.stdin:
    line = './../data/demo/devset/search.dev.json'
    save_path = './save_path.json'
    with open(line, 'r', encoding='utf-8') as f:    # 加载未处理的数据
        for line in f:
            sample = json.loads(line)
            # sample = json.loads(line)
            find_fake_answer(sample)
            # print(json.dump(sample, encoding='utf8', ensure_ascii=False))
            # 存文件,追加
            with open(save_path, 'a', encoding='utf-8') as out:
                json.dump(sample, out, ensure_ascii=False)
                # out.write('\n')

    # ./../data/demo/devset/search.rawsmall.json

    # jsonStr = '{"name":"aspiring", "age": 17, "hobby": ["money","power", "read"],"parames":{"a":1,"b":2}}'
    #
    # # 将json格式的字符串转为python数据类型的对象
    # jsonData = json.loads(jsonStr)
    # print(jsonData)
    # print(type(jsonData))
    # print(jsonData['hobby'])