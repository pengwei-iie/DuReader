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
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
import tfu
import modules

class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.fully_hidden = args.fully_hidden
        self.layer = args.layer
        self.head = args.head
        self.batch_size = args.batch_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.dropout_keep_prob = args.dropout_keep_prob
        self.is_train = tf.cast(True, tf.bool)

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        # num_gpus = 4
        #
        # for i in range(num_gpus):
        #     with tf.device('/gpu:%d', i):
        self._embed()
        # self._fusion()
        self._encode()
        self._match()
        self._fuse()
        # self._self_attention()  # add self_attention
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        def _mapper(pl):
            mask = tf.cast(pl, tf.bool)
            length = tf.reduce_sum(tf.to_int32(mask), -1)
            return {'data': pl, 'mask': mask, 'length': length}

        self.p = _mapper(tf.placeholder(tf.int32, [None, None]))
        self.q = _mapper(tf.placeholder(tf.int32, [None, None]))
        self.p_length = self.p['length']
        self.q_length = self.q['length']
        self.p_pos = tf.placeholder(tf.int32, [None, None])
        self.q_pos = tf.placeholder(tf.int32, [None, None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        # self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p['data'])
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q['data'])

    # def _dialogue_pos_embedding(self, ids, emb_dim, scope='dialogue_pos_emb', reuse=False):
    #     with tf.variable_scope(scope, reuse=reuse):
    #         pos_emb = tf.get_variable('dia_pos_emb', dtype=tf.float32,
    #                                   shape=(self._config['dia_pos_num'], emb_dim))
    #         res = tf.nn.embedding_lookup(pos_emb, ids)
    #         return res

    # def fusion(self, old, new, name):
    #     # 连接特征
    #     tmp = tf.concat([old, new, old*new, old-new], axis=2)   # b, len, hidden*4
    #     # 激活
    #     new_sens_tanh = tf.nn.tanh(tfu.dense(tmp, self.hidden_size*2, scope=name))
    #     # gate
    #     gate = tf.nn.sigmoid(tfu.dense(tmp, 1, scope=name+"sigmoid"))
    #     outputs = gate*new_sens_tanh + (1-gate)*old
    #     return outputs

    def _single_encoder(self, reuse=False):
        with tf.variable_scope('paragraph_encoder', reuse=reuse):
            hidden, layers, heads, ffd_hidden = self.hidden_size, self.layer, self.head, self.fully_hidden
            sent = tfu.add_timing_signal(self.p_emb)
            # sent = tfu.dropout(sent, self._kprob, self._is_train)
            trans = tfu.TransformerEncoder(hidden=hidden, layers=layers, heads=heads, ffd_hidden=ffd_hidden,
                                           keep_prob=self.dropout_keep_prob, is_train=self.is_train, scope='paragraph_trans')
            sent_emb = trans(sent, self.p['mask'])      # batch5, num, 256
            return sent_emb

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately.
        """
        with tf.variable_scope("paragraph_encoder"):
            # word_num = tf.shape(self.p_emb)[1]
            # emb_h = self.p_emb.shape.as_list()[-1]
            # sent = tf.reshape(emb, [self._batch * sent_num, word_num, emb_h])
            # sent_mask = tf.reshape(mask, [self._batch * sent_num, word_num])
            self.para_emb = self._single_encoder()  # 得到了所有段落的表示     batch5, num, hidden
            self.sep_p_encodes, _ = rnn('bi-lstm', self.para_emb, self.p_length, self.hidden_size)  # 得到rnn的输出和状态
            # hidden = para_emb.shape.as_list()[-1]
            # # 下面得到段落级别的表示，batch5， hidden
            # para_level_emb, prob = tfu.summ(para_emb, hidden, self.p['mask'],
            #                                 self.dropout_keep_prob, self.is_train, 'summ2sent', True)

            # 得到每一个段落的表示
            # para_level_emb = tf.reshape(para_level_emb,
            #                             [self.batch_size, -1, para_level_emb.shape.as_list()[-1]])  # 经过了参数的attention

            # 对编码的段落中每个词的attention，用于后面PGnet
            # prob = tf.reshape(prob, [self.batch_size, -1, word_num])
            # # 对每一个句子在词级别做完attention之后，reshape，然后在开始分离第一句话，即ot
            # # order, _ = tf.split(sent_level_emb, [1, sent_num - 1], axis=1)
            # # _, prob = tf.split(prob, [1, sent_num - 1], axis=1)
            #
            # # 避免oom，选择top-10的word，主要是计算后面的层级pointer
            # prob, index = tf.nn.top_k(prob, self._config.get('top_word', 10))
            # select_ids = tfu.batch_gather(ids, index)
            # sent_emb = tf.reshape(para_emb, [self._batch, sent_num, word_num, hidden])
            # # _, sent_emb = tf.split(sent_emb, [1, sent_num - 1], axis=1)
            # select_emb = tfu.batch_gather(sent_emb, index)

            # 这里可以开始加门了sent_level_emb（b, diags, hidden_size）
            # Ws = tf.Variable(tf.random_normal(shape=[hidden, 1], mean=0, stddev=0.01), name='v1')
            # score = tf.sigmoid(tf.matmul(sent_level_emb, Ws))
            # score = tf.sigmoid(tfu.dense(sent_level_emb, 1))
            # sent_level_emb = score * sent_level_emb

            # # 2.用工单标题做sigomid，order先要扩围，先在最后一个维度上连接，然后过一个dense，再加一个激活
            # order = tf.tile(order, [1, sent_num, 1])
            # sent_level_emb = tf.concat(values=[sent_level_emb, order], axis=2)
            # score = tf.sigmoid((tf.tanh(tfu.dense(sent_level_emb, 1))))
            # sent_level_emb = score * sent_level_emb

            # 使用fusion
            # score = tf.matmul(a=tf.nn.relu(tfu.dense(sent_level_emb, 1, use_bias=False, scope="score_b_orderanddiag")),
            #                   b=tf.nn.relu(tfu.dense(order, 1, use_bias=False, scope="score_b_orderanddiag", reuse=True)),
            #                   transpose_b=True)
            # # 计算attention，得到融合标题的句子表达
            # order_aware_sents = sent_level_emb * score
            # # 连接向量
            # new_sens = tf.concat([order_aware_sents, sent_level_emb, order_aware_sents * sent_level_emb], axis=2)
            # # 激活
            # new_sens_tanh = tf.nn.tanh(tfu.dense(new_sens, hidden, scope="new_sens_dense"))
            # # 利用这个new_sens_dense计算sigmoid
            # gate = tf.nn.sigmoid(tfu.dense(new_sens, 1, scope="gate"))
            # # 得到最后的表示
            # sent_level_emb = gate * new_sens_tanh + (1 - gate) * sent_level_emb
            #
            # sent_level_emb = self._pos_embedding_encoder(
            #     self._config['dialogue'], sent_level_emb, dia_mask, self._dialogue['pos'], 'dialogue', reuse)

            # return sent_level_emb, prob, select_ids, select_emb
        # with tf.variable_scope('passage_encoding'):
        #     self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size) # 得到rnn的输出和状态

        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            # self.sep_p_encodes += modules.positional_encoding(self.p['data'], self.hidden_size,
            #                                                   zero_pad=False, scale=False, scope='encoder_position')

            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

        # p_pos_emb = self._dialogue_pos_embedding(self.p_pos, emb.shape.as_list()[-1])

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, self.match_q_encodes = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p, self.q)                   # 连接了四个向量，最后得到b*len(pa)*1200
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)
            self.match_q_encodes = tf.nn.dropout(self.match_q_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion_p'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)  # 经过双向RNN,变成前向+后向，150+150
        with tf.variable_scope('fusion_q'):
            self.fuse_q_encodes, _ = rnn('bi-lstm', self.match_q_encodes, self.q_length,
                                         self.hidden_size, layer_num=1)  # 经过双向RNN,变成前向+后向，150+150
        if self.use_dropout:
            self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)
            self.fuse_q_encodes = tf.nn.dropout(self.fuse_q_encodes, self.dropout_keep_prob)

    def _self_attention(self):
        # 双线性softmax
        with tf.variable_scope('bi_linear'):
            # 经过双向lstm，最后一维变成300
            batch_add5 = tf.shape(self.fuse_p_encodes)[0]

            # use xavier initialization
            W_bi = tf.get_variable("W_bi", [self.hidden_size * 2, self.hidden_size * 2],
                                   initializer=tf.contrib.layers.xavier_initializer())
            # W_bi = tf.get_variable("W_bi", [self.hidden_size * 2, self.hidden_size * 2],
            #                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32))
            tmp = tf.reshape(self.fuse_p_encodes, [-1, self.hidden_size*2])
            tmp = tf.matmul(tmp, W_bi)
            tmp = tf.reshape(tmp, [batch_add5, -1, self.hidden_size*2])
            # 以上就是通过reshape的方式进行双线性变化
            before_softmax = tf.matmul(tmp, self.fuse_p_encodes, transpose_b=True)      # b, n, n
            L = tfu.mask_softmax(before_softmax, self.p['mask'])
            # L = tf.nn.softmax(tf.matmul(tmp, self.fuse_p_encodes, transpose_b=True))
            self.binear_passage = tf.matmul(L, self.fuse_p_encodes)
            self.binear_passage = tfu.fusion(self.fuse_p_encodes, self.binear_passage, self.hidden_size, name="binear")

            # 将最后一维变成self.hidden_size
            # self.binear_passage = tfu.dense(self.binear_passage, 1, "to_hidden_size")
            # 需要再经过一个双向LISTM
            with tf.variable_scope('self_attention'):
                self.fina_passage, _ = rnn('bi-lstm', self.binear_passage, self.p_length,
                                             self.hidden_size, layer_num=1)  # 经过双向RNN,变成前向+后向，150+150

            # 对问题操作,自对其，后续
            W_q = tf.get_variable("W_q", [self.hidden_size * 2, self.hidden_size * 2],
                                   initializer=tf.contrib.layers.xavier_initializer())
            tmp = tf.reshape(self.fuse_q_encodes, [-1, self.hidden_size * 2])
            tmp = tf.matmul(tmp, W_q)
            tmp = tf.reshape(tmp, [batch_add5, -1, self.hidden_size * 2])   # b, q-len, hidden
            alpha = tf.nn.softmax(tmp)      # b, n_q, 300
            self.self_ques = alpha*self.fuse_q_encodes

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )       # fina_passage: b*5, len_p, hidden --> b, 5*len_p, hidden

            # b, 5, len_q, hidden       [0:, 0, 0:, 0:] 表示只用第一个问题，因为5个问题都是一样的
            no_dup_question_encodes = tf.reshape(
                self.fuse_q_encodes,
                [batch_size, -1, tf.shape(self.fuse_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]

        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p['data']: batch['passage_token_ids'],
                         self.q['data']: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id']}
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix, rand_seed,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix, rand_seed)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p['data']: batch['passage_token_ids'],
                         self.q['data']: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id']}
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix, rand_seed):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        model_prefix = model_prefix + str(rand_seed)
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
