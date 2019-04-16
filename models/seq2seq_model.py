#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-16 上午10:01
# @作者   : Lin lifang
# @文件   : seq2seq_model.py
import os

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
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		batch_size, d, _, t = out.size()
		out = out.view(batch_size, d, t)
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


class RNNAttentionDecoder2(nn.Module):
	def __init__(self, attn_model, vocab_size, hidden_size, output_size, num_layers=1, dropout=0.0):
		super(RNNAttentionDecoder2, self).__init__()
		self.hidden_size = hidden_size
		self.input_vocab_size = vocab_size
		self.output_size = output_size
		self.attn = Attention(attn_model, hidden_size)
		self.gru = nn.GRU(vocab_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
		self.embedding = nn.Embedding(vocab_size, vocab_size)
		fix_embedding = torch.from_numpy(np.eye(vocab_size, vocab_size).astype(np.float32))
		self.embedding.weight = nn.Parameter(fix_embedding)
		self.embedding.weight.requires_grad = False

		self.wc = nn.Linear(hidden_size + vocab_size, hidden_size)
		self.ws = nn.Linear(hidden_size * 2, output_size)

	def forward(self, inputs, last_ht, last_hidden, encoder_outputs):
		embed_input = self.embedding(inputs)
		attn_input = self.wc(torch.cat((embed_input, last_hidden[-1]), 1))
		weighted_context, alpha = self.attn(attn_input, encoder_outputs)
		rnn_input = torch.cat((embed_input, weighted_context), 1)
		output, hidden = self.gru(rnn_input.unsqueeze(1), last_hidden)
		output = output.squeeze()
		output = self.ws(torch.cat((output, nn.functional.tanh(weighted_context)), 1))
		return output, last_ht, hidden, alpha


class Seq2seqModel(nn.Module):
	def __init__(self, attn_model, vocab_size, hidden_size, output_size, num_layer=1, dropout=0):
		super(Seq2seqModel, self).__init__()
		self.encoder = Encoder(num_layer, hidden_size, dropout)
		self.decoder = RNNAttentionDecoder(attn_model, vocab_size, hidden_size, output_size, num_layer, dropout)

	def forward(self):
		pass

	def forward_encoder(self, inputs, batch_size, use_cuda):
		"""编码层前向算法"""
		init_hidden = self.encoder.init_hidden(batch_size, use_cuda)
		encoder_outputs, last_hidden = self.encoder(inputs, init_hidden)
		return encoder_outputs, last_hidden

	def forward_decoder(self, inputs, last_ht, last_hidden, encoder_outputs):
		output, last_ht, last_hidden, alpha = self.decoder(inputs, last_ht, last_hidden, encoder_outputs)
		return output, last_ht, last_hidden, alpha

	def save(self, circle):
		name = "./data/temp_seq2seq_model" + str(circle) + ".pth"
		torch.save(self.state_dict(), name)
		name2 = "./data/seq2seq_model.pth"
		torch.save(self.state_dict(), name2)

	def load_model(self):
		file_list = os.listdir("./data")
		if "seq2seq_model.pth" in file_list:
			name = "./data/seq2seq_model.pth"
			self.load_state_dict(torch.load(name))
			print("the latest model has been load")


def train(inputs, targets, model, optimizer, criterion, hidden_size, clip, use_cuda=False):
	loss = 0
	optimizer.zero_grad()
	batch_size, max_len = targets.size()

	encoder_outputs, last_hidden = model.forward_encoder(inputs, batch_size, use_cuda)

	last_ht = Variable(torch.zeros(batch_size, hidden_size))
	outputs = torch.zeros((batch_size, max_len - 1)).long()
	if use_cuda:
		last_ht = last_ht.cuda()
		outputs = outputs.cuda()

	for di in range(max_len - 1):
		de_inputs = targets[:, di]
		target = targets[:, di + 1]
		output, last_ht, last_hidden, alpha = model.forward_decoder(de_inputs, last_ht, last_hidden, encoder_outputs)
		outputs[:, di] = output.max(1)[1].data
		loss += criterion(output, target)

	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
	optimizer.step()

	num_eq = (targets[:, 1:].cpu().data == outputs.cpu()).sum(dim=1)
	accuracy = (num_eq == max_len - 1).sum().item() / batch_size
	return loss.item(), accuracy


def evaluate(inputs, targets, model, criterion, hidden_size, use_cuda=False):
	model.train(False)
	loss = 0
	batch_size, max_len = targets.size()
	encoder_outputs, last_hidden = model.forward_encoder(inputs, batch_size, use_cuda)
	last_ht = Variable(torch.zeros(batch_size, hidden_size))
	outputs = torch.zeros((batch_size, max_len - 1)).long()
	if use_cuda:
		last_ht = last_ht.cuda()
		outputs = outputs.cuda()
	de_inputs = targets[:, 0]
	for di in range(max_len - 1):
		# 解码 上一时刻输出 作为此时刻输入
		output, last_ht, last_hidden, alpha = model.forward_decoder(de_inputs, last_ht, last_hidden, encoder_outputs)
		de_inputs = output.max(1)[1]
		outputs[:, di] = de_inputs.data
		loss += criterion(output, targets[:, di + 1])
	num_eq = (targets[:, 1:].data == outputs).sum(dim=1)
	accuracy = (num_eq == max_len - 1).sum().item() / batch_size
	model.train(True)
	return loss.item(), accuracy, outputs


def seq2seq_test(inputs, targets, max_len, model, hidden_size, use_cuda=False):
	model.train(False)
	batch_size = inputs.size()[0]
	encoder_outputs, last_hidden = model.forward_encoder(inputs, batch_size, use_cuda)
	last_ht = Variable(torch.zeros(batch_size, hidden_size))
	outputs = torch.zeros((batch_size, max_len - 1)).long()
	if use_cuda:
		last_ht = last_ht.cuda()
		outputs = outputs.cuda()

	de_inputs = targets[:, 0]
	for di in range(max_len - 1):
		# 解码 上一时刻输出 作为此时刻输入
		output, last_ht, last_hidden, alpha = model.forward_decoder(de_inputs, last_ht, last_hidden, encoder_outputs)
		de_inputs = output.max(1)[1]
		outputs[:, di] = de_inputs.data
	return outputs.data.tolist()[0]
