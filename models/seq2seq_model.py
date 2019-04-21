#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-16 上午10:01
# @作者   : Lin lifang
# @文件   : seq2seq_model.py
import copy
import os

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, num_layer=1, hidden_size=100, dropout=0):
        super(Encoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 4), stride=(3, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(4, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.gru = nn.GRU(128, hidden_size, num_layer, batch_first=True, dropout=dropout)

    def forward(self, x, hidden):
        h0 = hidden
        out = self.layer1(x)  # batch * 32 * 16 * 63
        out = self.layer2(out)  # batch * 64 * 4 * 31
        out = self.layer3(out)  # batch * 128 * 1 * 30
        out = out.squeeze(2)
        out = out.transpose(1, 2)
        out, hidden = self.gru(out, h0)
        return out, hidden

    def init_hidden(self, batch_size, use_cuda=False):
        h0 = Variable(torch.zeros(self.num_layer, batch_size, self.hidden_size))
        if use_cuda:
            return h0.cuda()
        else:
            return h0


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.attn_linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: decode hidden state, (batch_size , N)
        :param encoder_outputs: encoder's all states, (batch_size,T,N)
        :return: weithed_context :(batch_size,N), alpha:(batch_size,T)
        """
        hidden_expanded = hidden.unsqueeze(2)  # (batch_size,N,1)
        if self.method == 'dot':
            energy = torch.bmm(encoder_outputs, hidden_expanded).squeeze(2)
        elif self.method == 'general':
            energy = self.attn(encoder_outputs)
            energy = torch.bmm(energy, hidden_expanded).squeeze(2)
        elif self.method == 'concat':
            hidden_expanded = hidden.unsqueeze(1).expand_as(encoder_outputs)
            energy = self.attn(torch.cat((hidden_expanded, encoder_outputs), 2))
            energy = self.attn_linear(self.tanh(energy)).squeeze(2)
        alpha = nn.functional.softmax(energy, dim=-1)
        weighted_context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)

        return weighted_context, alpha


class RNNAttentionDecoder(nn.Module):
    def __init__(self, attn_model, vocab_size, hidden_size, output_size, num_layer=1, dropout=0.):
        super(RNNAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.attention = Attention(attn_model, hidden_size)
        self.gru = nn.GRU(vocab_size + hidden_size, hidden_size, num_layer, batch_first=True, dropout=dropout)
        self.wc = nn.Linear(2 * hidden_size, hidden_size)  # ,bias=False)
        self.ws = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        fix_embedding = torch.from_numpy(np.eye(vocab_size, vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, inputs, last_ht, last_hidden, encoder_outputs):
        """
        :param inputs: (batch_size,)
        :param last_ht: (obatch_size,hidden_size)
        :param last_hidden: (batch_size,hidden_size)
        :param encoder_outputs: (batch_size,T,hidden_size)
        """
        embed_input = self.embedding(inputs)  # one-hot batch_size vocab_size
        rnn_input = torch.cat((embed_input, last_ht), 1)  # batch_size vocab_size+hidden_size
        output, hidden = self.gru(rnn_input.unsqueeze(1), last_hidden)
        output = output.squeeze(1)
        weighted_context, alpha = self.attention(output, encoder_outputs)
        ht = self.tanh(self.wc(torch.cat((output, weighted_context), 1)))
        output = self.ws(ht)
        return output, ht, hidden, alpha


class AttentionDecoder(nn.Module):
    def __init__(self, attn_model, vocab_size, hidden_size, output_size, num_layers=1, dropout=0.0, use_cuda=False):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.use_cuda = use_cuda
        self.attention = Attention(attn_model, hidden_size)
        self.gru = nn.GRU(vocab_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        fix_embedding = torch.from_numpy(np.eye(vocab_size, vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad = False
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden, context, max_len, encoder_outputs):
        """
        解码层 训练阶段
        :param inputs: batch T
        :param hidden: batch D
        :param context: batch D
        :param max_len: 解码层最大长度
        :param encoder_outputs: 编码层输出 batch T D
        :return:
        """
        batch_size, _ = inputs.size()
        embedded = self.embedding(inputs)  # batch T D
        scores = []
        for k in range(max_len - 1):
            output, hidden = self.gru(torch.cat((embedded[:, k], context), dim=1).unsqueeze(1), hidden)
            output = output.squeeze(1)
            score = self.linear(output)
            scores.append(score)
            context, alpha = self.attention(output, encoder_outputs)
        return scores

    def greedy_search_decode(self, inputs, hidden, context, max_len, encoder_outputs):
        """
        解码层 贪心算法 测试阶段
        :param inputs: batch 1 start symbol
        :param hidden: batch D
        :param context: batch 1 D encoder hidden
        :param max_len: 解码层最大长度
        :param encoder_outputs: 编码层输出 batch T D
        :return:
        """
        embedded = self.embedding(inputs)  # batch 1 d
        scores = []
        for k in range(max_len - 1):
            output, hidden = self.gru(torch.cat((embedded, context), dim=1).unsqueeze(1), hidden)
            output = output.squeeze(1)
            # output = torch.cat((_output, context), dim=1)
            score = self.linear(output)
            scores.append(score)
            decoded = score.max(1)[1]  # 最大值索引 batch
            embedded = self.embedding(decoded)  # batch 1 d
            context, alpha = self.attention(output, encoder_outputs)
        return scores

    def beam_search_decode(self, inputs, hidden, context, max_len, encoder_outputs, topk=1):
        """
        beam search 算法 每次计算topk个值 topk=1 是贪心算法
        :param inputs:
        :param hidden:
        :param context:
        :param max_len:
        :param encoder_outputs:
        :param topk: beam宽度
        :return:
        """
        embedded = self.embedding(inputs)  # batch 1 d
        output, hidden = self.gru(torch.cat((embedded, context), dim=1).unsqueeze(1), hidden)
        candidate = output.squeeze(1)  # batch * D
        score = self.linear(candidate)
        beam = Beam([score, hidden], topk, max_len)
        nodes = beam.get_next_nodes()
        t = 1
        while t < max_len:
            siblings = []
            for inputs, hidden in nodes:
                inputs = inputs.to(encoder_outputs.device)
                embedded = self.embedding(inputs)
                context, alpha = self.attention(hidden.squeeze(1), encoder_outputs)
                output, hidden = self.gru(torch.cat((embedded, context), dim=1).unsqueeze(1), hidden)
                candidate = output.squeeze(1)
                score = self.linear(candidate)
                siblings.append([score, hidden])
            nodes = beam.select_k(siblings, t)
            t += 1
        return beam.get_best_seq()


class Beam:
    def __init__(self, root, num_beam, max_len):
        """
        root : (score, hidden)
        batch * vocab_size
        """
        score, hidden = root
        self.num_beam = num_beam
        self.max_len = max_len
        self.batch_size = score.size()[0]
        self.hidden = torch.zeros_like(hidden)
        score = F.log_softmax(score, 1)
        s, i = score.topk(num_beam)
        s = s.data
        i = i.data
        self.beams = []
        for ii in range(num_beam):
            path = torch.zeros(self.batch_size, max_len)
            path[:, 0] = i[:, ii]
            beam = [s[:, ii], path, hidden]
            self.beams.append(beam)

    def select_k(self, siblings, t):
        """
        siblings : [score,hidden]
        """
        candidate = []
        for p_index, sibling in enumerate(siblings):
            score, hidden = sibling
            parents = self.beams[p_index]  # (cummulated score, list of sequence)
            score = F.log_softmax(score, 1)
            s, i = score.topk(self.num_beam)
            s = s.data
            i = i.data
            for kk in range(self.num_beam):
                vocab_id = copy.deepcopy(parents[1])
                vocab_id[:, t] = i[:, kk]
                current_score = parents[0] + s[:, kk]
                candidate.append([current_score, vocab_id, hidden])
        # 候选集排序
        beams = [[torch.zeros(self.batch_size), torch.zeros(self.batch_size, self.max_len, dtype=torch.int),
                  torch.zeros_like(self.hidden)] for _ in range(self.num_beam)]
        for ii in range(self.batch_size):
            beam = [[cand[0][ii], cand[1][ii, :], cand[2][ii]] for cand in candidate]
            beam = sorted(beam, key=lambda x: x[0], reverse=True)[:self.num_beam]

            for kk in range(self.num_beam):
                beams[kk][0][ii] = beam[kk][0]
                beams[kk][1][ii, :] = beam[kk][1]
                beams[kk][2][ii] = beam[kk][2]
        self.beams = beams
        # last_input, hidden
        return [[b[1][:, t], b[2]] for b in self.beams]

    def get_best_seq(self):
        return self.beams[0][1]

    def get_next_nodes(self):
        return [[b[1][:, 0], b[2]] for b in self.beams]


class Seq2seqModel(nn.Module):
    def __init__(self, attn_model, vocab_size, hidden_size, output_size, num_layer=1, dropout=0, use_cuda=False):
        super(Seq2seqModel, self).__init__()
        self.use_cuda = use_cuda
        self.encoder = Encoder(num_layer, hidden_size, dropout)
        # self.decoder = RNNAttentionDecoder(attn_model, vocab_size, hidden_size, output_size, num_layer, dropout)
        self.decoder = AttentionDecoder(attn_model, vocab_size, hidden_size, output_size, num_layer, dropout, use_cuda)

    def forward(self, inputs, targets):
        """
        前向算法 编码层-解码层
        :param inputs:
        :param targets:
        :return:
        """
        batch_size, max_len = targets.size()
        hidden = self.encoder.init_hidden(batch_size, self.use_cuda)
        encoder_outputs, last_hidden = self.encoder(inputs, hidden)
        context = encoder_outputs.mean(dim=1)
        scores = self.decoder(targets, last_hidden, context, max_len, encoder_outputs)
        loss = self.loss_layer(scores, targets, max_len)
        acc = self.accuracy(scores, targets, max_len)
        return scores, loss, acc

    @staticmethod
    def loss_layer(scores, targets, max_len):
        """
        损失函数
        :param scores: batch max_len -1 vocab_size
        :param targets: batch max_len
        :param max_len:
        :return:
        """
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0
        for kk in range(max_len - 1):
            score = scores[kk]
            target = targets[:, kk + 1]
            loss += criterion(score, target)
        return loss

    @staticmethod
    def accuracy(scores, targets, max_len):
        """
        评估准确率
        :param scores:
        :param targets:
        :param max_len:
        :return:
        """
        predict = numpy.array([score.cpu().max(1)[1].data.tolist() for score in scores])
        num_eq = numpy.sum(targets[:, 1:].cpu().numpy() == predict.transpose(), axis=1)
        accuracy = (num_eq == max_len - 1).sum().item() / targets.size()[0]
        return accuracy

    def best_path(self, inputs, max_len, start, topk):
        batch_size = 1
        hidden = self.encoder.init_hidden(batch_size, self.use_cuda)
        encoder_outputs, last_hidden = self.encoder(inputs, hidden)
        context = encoder_outputs.mean(dim=1)
        start_target = torch.ones(batch_size).long() * start
        if self.use_cuda:
            start_target = start_target.cuda()
        best_path = self.decoder.beam_search_decode(start_target, last_hidden, context, max_len, encoder_outputs, topk)
        return best_path

    def evaluate(self, inputs, targets, start):

        batch_size, max_len = targets.size()
        hidden = self.encoder.init_hidden(batch_size, self.use_cuda)
        encoder_outputs, last_hidden = self.encoder(inputs, hidden)
        context = encoder_outputs.mean(dim=1)
        start_target = torch.ones(batch_size).to(targets.device).long() * start

        scores = self.decoder.greedy_search_decode(start_target, last_hidden, context, max_len, encoder_outputs)
        loss = self.loss_layer(scores, targets, max_len)
        acc = self.accuracy(scores, targets, max_len)
        return scores, loss, acc

    def forward_encoder(self, inputs, batch_size, use_cuda):
        """编码层前向算法"""
        init_hidden = self.encoder.init_hidden(batch_size, use_cuda)
        encoder_outputs, last_hidden = self.encoder(inputs, init_hidden)
        return encoder_outputs, last_hidden

    def forward_decoder(self, inputs, last_ht, last_hidden, encoder_outputs):
        output, last_ht, last_hidden, alpha = self.decoder(inputs, last_ht, last_hidden, encoder_outputs)
        return output, last_ht, last_hidden, alpha

    def save(self):
        name2 = "./data/seq2seq_model.pth"
        torch.save(self.state_dict(), name2)

    def load_model(self):
        file_list = os.listdir("./data")
        if "seq2seq_model.pth" in file_list:
            name = "./data/seq2seq_model.pth"
            self.load_state_dict(torch.load(name))
            print("the latest model has been load")
