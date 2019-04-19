#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-18 上午11:40
# @作者   : Lin lifang
# @文件   : transformer_model.py
import copy
import math
import os

import numpy as np
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, num_layer=1, hidden_size=512, dropout=0):
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
		self.gru = nn.GRU(128, hidden_size // 2, num_layer, batch_first=True, bidirectional=True, dropout=dropout)

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
		h0 = Variable(torch.zeros(self.num_layer * 2, batch_size, self.hidden_size // 2))
		if use_cuda:
			return h0.cuda()
		else:
			return h0


class Embedding(nn.Module):
	def __init__(self, vocab_size, hidden_size, dropout=0.0):
		super(Embedding, self).__init__()
		self.hidden_size = hidden_size
		self.token_embedding = nn.Embedding(vocab_size, hidden_size)
		self.layer_norm = LayerNorm(hidden_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, inputs):
		"""
		嵌入层
		:param inputs: batch * T
		:return:
		"""
		batch_size, seq_len = inputs.size()
		token_embedding = self.token_embedding(inputs)
		position_embedding = self.position_embedding(inputs, seq_len)

		embedding = token_embedding + position_embedding
		embedding = self.layer_norm(embedding)
		embedding = self.dropout(embedding)
		return embedding

	def position_embedding(self, inputs, seq_len):
		position_enc = torch.zeros((seq_len, self.hidden_size))
		for pos in range(seq_len):
			for kk in range(self.hidden_size):
				position_enc[pos, kk] = float(pos / np.power(10000, 2 * kk / self.hidden_size))
		position_enc[0::2] = torch.sin(position_enc[::2])
		position_enc[1::2] = torch.cos(position_enc[1::2])
		position_enc.unsqueeze(0).repeat((inputs.size()[0], 1, 1))
		return position_enc.to(inputs.device)


class LayerNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-12):
		"""Construct a layernorm module in the TF style (epsilon inside the square root).
		"""
		super(LayerNorm, self).__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.bias = nn.Parameter(torch.zeros(hidden_size))
		self.variance_epsilon = eps

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.weight * x + self.bias


class SelfAttention(nn.Module):
	def __init__(self, num_heads, hidden_size, dropout=0.0):
		super(SelfAttention, self).__init__()
		if hidden_size % num_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, num_heads))
		self.num_heads = num_heads
		self.attention_head_size = int(hidden_size / num_heads)
		self.all_head_size = self.num_heads * self.attention_head_size

		self.query = nn.Linear(hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(dropout)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, query_states, key_states, attention_mask, mask_future):
		"""
		注意力层
		:param query_states: B * T * D
		:param key_states: B * T * D
		:param attention_mask: B * T [1,1,1,1,1,1,0,0,0,0,0]
		:param mask_future: True or False
		:return:
		"""
		mixed_query_layer = self.query(query_states)
		mixed_key_layer = self.key(key_states)
		mixed_value_layer = self.value(key_states)
		# 切分 B * T * D => B*H * T * D/H
		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # q dot k => B*H * T * T_k
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask
		if mask_future:
			diag_vals = torch.ones_like(attention_scores[0, :, :])
			triu = (1.0 - torch.tril(diag_vals)) * -10000.0
			mask = triu.unsqueeze(0).expand_as(attention_scores)
			attention_scores = attention_scores + mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer


class Output(nn.Module):
	def __init__(self, hidden_size, dropout=0.0):
		super(Output, self).__init__()
		self.dense = nn.Linear(hidden_size, hidden_size)
		self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class Attention(nn.Module):
	def __init__(self, num_heads, hidden_size, dropout=0.0):
		super(Attention, self).__init__()
		self.self = SelfAttention(num_heads, hidden_size, dropout=0.0)
		self.output = Output(hidden_size, dropout)

	def forward(self, query_states, key_states, attention_mask, mask_future):
		self_output = self.self(query_states, key_states, attention_mask, mask_future)
		attention_output = self.output(self_output, query_states)
		return attention_output


class BertLayer(nn.Module):
	def __init__(self, num_heads, hidden_size, dropout=0.0):
		super(BertLayer, self).__init__()
		self.encoder_attention = Attention(num_heads, hidden_size, dropout)
		self.decoder_attention = Attention(num_heads, hidden_size, dropout)
		self.output = Output(hidden_size, dropout)

	def forward(self, query_states, key_states, attention_mask):
		query_states = self.encoder_attention(query_states, query_states, attention_mask, False)
		query_states = self.decoder_attention(query_states, key_states, attention_mask, True)
		# intermediate_output = self.intermediate(attention_output)
		# layer_output = self.output(attention_output, query_states)
		return query_states


class BertEncoder(nn.Module):
	def __init__(self, num_layers, num_heads, hidden_size, dropout):
		super(BertEncoder, self).__init__()
		layer = BertLayer(num_heads, hidden_size, dropout)
		self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

	def forward(self, query_states, key_states, attention_mask):
		for layer_module in self.layer:
			query_states = layer_module(query_states, key_states, attention_mask)
		return query_states


class BertModel(nn.Module):
	def __init__(self, num_layers, num_heads, vocab_size, hidden_size, dropout=0.0):
		super(BertModel, self).__init__()
		self.embeddings = Embedding(vocab_size, hidden_size, dropout)
		# self.encoder = BertEncoder(1, num_heads, hidden_size, dropout)
		self.decoder = BertEncoder(num_layers, num_heads, hidden_size, dropout)
		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, input_ids, attention_mask, key_states):
		"""
		transformer
		:param input_ids: batch * T
		:param attention_mask: batch * T [[1,1,1,1,1,0,0,0,0]]
		:param key_states: batch * T * D
		:return:
		"""
		# mask padding
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # batch 1 1 T
		attention_mask = (1.0 - attention_mask) * -10000.0
		embedding_output = self.embeddings(input_ids)
		# query_states = self.encoder(embedding_output, embedding_output, attention_mask, True)
		decoder_layers = self.decoder(embedding_output, key_states, attention_mask)
		scores = self.linear(decoder_layers)
		return scores

	def greedy_search_decode(self, input_ids, max_len, key_states):
		"""
		解码层 贪心算法 测试阶段
		:param input_ids: batch 1 start symbol
		:param max_len: 解码层最大长度
		:param key_states: 编码层输出 batch T D
		:return:
		"""
		embedded = self.embedding(input_ids)  # batch 1 d

		scores = []
		for k in range(max_len - 1):
			output, hidden = self.gru(torch.cat((embedded, context), dim=1).unsqueeze(1), hidden)
			output = output.squeeze(1)
			# output = torch.cat((_output, context), dim=1)
			score = self.linear(output)
			scores.append(score)
			decoded = score.max(1)[1]  # 最大值索引 batch
			embedded = self.embedding(decoded)  # batch 1 d
			context, alpha = self.attention(output, key_states)
		return scores

	def beam_search_decode(self, input_ids, max_len, key_states, topk=1):
		"""
		beam search 算法 每次计算topk个值 topk=1 是贪心算法
		:param input_ids:
		:param max_len: 最大长度
		:param key_states:
		:param topk: beam宽度
		:return:
		"""
		embedded = self.embeddings(input_ids)  # batch 1 d
		mask = torch.zeros_like(input_ids, dtype=torch.float, device=input_ids.device).unsqueeze(1).unsqueeze(2)
		mask[:, :, :, 0] = 1
		# query_states = self.encoder(embedded, embedded, mask)
		decoder_layers = self.decoder(embedded, key_states, mask, )
		scores = self.linear(decoder_layers)[:, 0, :]
		beam = Beam(scores, topk)
		nodes = beam.get_next_nodes()
		for t in range(1, max_len):
			mask[:, :, :, t] = 1
			siblings = []
			for node in nodes:
				input_ids[:, t] = node
				embedded = self.embeddings(input_ids)
				# query_states = self.encoder(embedded, embedded, mask)
				decoder_layers = self.decoder(embedded, key_states, mask, )
				scores = self.linear(decoder_layers)[:, t, :]
				siblings.append(scores)
			nodes = beam.select_k(siblings)
		return beam.get_best_seq()


class Beam:
	def __init__(self, score, num_beam):
		"""
		root : (score, hidden)
		batch * vocab_size
		"""
		self.num_beam = num_beam
		score = F.log_softmax(score, 1)
		s, i = score.topk(num_beam)
		s = s.data.tolist()[0]
		i = i.data.tolist()[0]
		i = [[ii] for ii in i]
		self.beams = list(zip(s, i))
		self.beams = sorted(self.beams, key=lambda x: x[0], reverse=True)

	def select_k(self, siblings):
		"""
		siblings : [score,hidden]
		"""
		candidate = []
		for p_index, score in enumerate(siblings):
			parents = self.beams[p_index]  # (cummulated score, list of sequence)
			score = F.log_softmax(score, 1)
			s, i = score.topk(self.num_beam)
			scores = s.data.tolist()[0]
			indices = i.data.tolist()[0]
			candidate.extend([(parents[0] + scores[i], parents[1] + [indices[i]]) for i in range(len(scores))])
		candidate = sorted(candidate, key=lambda x: x[0], reverse=True)
		self.beams = candidate[:self.num_beam]
		# last_input, hidden
		return [b[1][-1] for b in self.beams]

	def get_best_seq(self):
		return self.beams[0][1]

	def get_next_nodes(self):
		return [b[1][-1] for b in self.beams]


class TransformerModel(nn.Module):
	def __init__(self, num_layers, num_heads, vocab_size, hidden_size, dropout, use_cuda):
		super(TransformerModel, self).__init__()
		self.use_cuda = use_cuda
		self.encoder = Encoder(1, hidden_size, dropout)
		self.bert_model = BertModel(num_layers, num_heads, vocab_size, hidden_size, dropout)
		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, inputs, targets, seq_len):

		batch_size, MAX_LEN = targets.size()

		hidden = self.encoder.init_hidden(batch_size, self.use_cuda)
		encoder_outputs, last_hidden = self.encoder(inputs, hidden)
		max_len = encoder_outputs.size()[1]
		pad_data = Variable(torch.zeros(batch_size, max_len - MAX_LEN).to(targets.device)).long()
		pad_target = torch.cat((targets, pad_data), dim=1)
		mask = torch.zeros_like(pad_target).float()
		for kk in range(batch_size):
			b_len = seq_len[kk] + 2
			mask[kk, :b_len] = 1

		scores = self.bert_model(pad_target, mask, encoder_outputs)
		loss = self.loss_layer(scores, targets, MAX_LEN)
		acc = self.accuracy(scores, targets, MAX_LEN)
		return loss, acc

	@staticmethod
	def loss_layer(scores, targets, max_len):
		criterion = torch.nn.CrossEntropyLoss()
		loss = 0
		for kk in range(max_len - 1):
			score = scores[:, kk]
			target = targets[:, kk + 1]
			loss += criterion(score, target)
		return loss

	@staticmethod
	def accuracy(scores, targets, max_len):
		predict = scores.max(2)[1]
		num_eq = (targets[:, 1:].cpu().data == predict[:, :max_len - 1].cpu()).sum(dim=1)
		accuracy = (num_eq == max_len - 1).sum() / targets.size()[0]
		return accuracy.item()

	def best_path(self, inputs, MAX_LEN, start, topk):
		batch_size = 1
		hidden = self.encoder.init_hidden(batch_size, self.use_cuda)
		encoder_outputs, last_hidden = self.encoder(inputs, hidden)
		max_len = encoder_outputs.size()[1]
		start_target = torch.zeros((batch_size, max_len), dtype=torch.long, device=inputs.device)
		start_target[:, 0] = start
		if self.use_cuda:
			start_target = start_target.cuda()
		best_path = self.bert_model.beam_search_decode(start_target, max_len, encoder_outputs, topk)
		return best_path

	def save(self):
		name2 = "./data/transformer_model.pth"
		torch.save(self.state_dict(), name2)

	def load_model(self):
		file_list = os.listdir("./data")
		if "transformer_model.pth" in file_list:
			name = "./data/transformer_model.pth"
			self.load_state_dict(torch.load(name))
			print("the latest model has been load")
