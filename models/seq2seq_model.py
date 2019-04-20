#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-16 上午10:01
# @作者   : Lin lifang
# @文件   : seq2seq_model.py
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

	def beam_search_decode(self, inputs, hidden, context, max_length, encoder_outputs, topk=1):
		"""
		beam search 算法 每次计算topk个值 topk=1 是贪心算法
		:param inputs:
		:param hidden:
		:param context:
		:param max_length:
		:param encoder_outputs:
		:param topk: beam宽度
		:return:
		"""
		embedded = self.embedding(inputs)  # batch 1 d
		output, hidden = self.gru(torch.cat((embedded, context), dim=1).unsqueeze(1), hidden)
		candidate = output.squeeze(1)
		score = self.linear(candidate)
		beam = Beam([score, hidden], num_beam=topk)
		nodes = beam.get_next_nodes()
		t = 1
		while t < max_length:
			siblings = []
			for inputs, hidden in nodes:
				if self.use_cuda:
					inputs = inputs.cuda()
				embedded = self.embedding(inputs)
				context, alpha = self.attention(hidden.squeeze(1), encoder_outputs)
				output, hidden = self.gru(torch.cat((embedded.squeeze(1), context), dim=1).unsqueeze(1), hidden)
				candidate = output.squeeze(1)
				score = self.linear(candidate)
				siblings.append([score, hidden])
			nodes = beam.select_k(siblings)
			t += 1
		return beam.get_best_seq()


class Beam:
	def __init__(self, root, num_beam):
		"""
		root : (score, hidden)
		batch * vocab_size
		"""
		self.num_beam = num_beam
		score = F.log_softmax(root[0], 1)
		s, i = score.topk(num_beam)
		s = s.data.tolist()[0]
		i = i.data.tolist()[0]
		i = [[ii] for ii in i]
		hidden = [root[1] for _ in range(num_beam)]
		self.beams = list(zip(s, i, hidden))
		self.beams = sorted(self.beams, key=lambda x: x[0], reverse=True)

	def select_k(self, siblings):
		"""
		siblings : [score,hidden]
		"""
		candidate = []
		for p_index, sibling in enumerate(siblings):
			parents = self.beams[p_index]  # (cummulated score, list of sequence)
			score = F.log_softmax(sibling[0], 1)
			s, i = score.topk(self.num_beam)
			scores = s.data.tolist()[0]
			indices = i.data.tolist()[0]
			candidate.extend(
				[(parents[0] + scores[i], parents[1] + [indices[i]], sibling[1]) for i in range(len(scores))])
		candidate = sorted(candidate, key=lambda x: x[0], reverse=True)
		self.beams = candidate[:self.num_beam]
		# last_input, hidden
		return [[Variable(torch.LongTensor([b[1][-1]])).view(1, -1), b[2]] for b in self.beams]

	def get_best_seq(self):
		return self.beams[0][1]

	def get_next_nodes(self):
		return [[Variable(torch.LongTensor([b[1][-1]])).view(1, -1), b[2]] for b in self.beams]


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
