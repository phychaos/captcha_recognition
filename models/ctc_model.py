#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午11:22
# @作者   : Lin lifang
# @文件   : ctc_model.py
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class ResBlock(nn.Module):
	def __init__(self, in_c, out_c, stride=1):
		super(ResBlock, self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_c, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(out_c, track_running_stats=True)
		)
		self.down_sample = nn.Sequential()
		if in_c != out_c:
			self.down_sample = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_c, track_running_stats=True)
			)
		self.in_c = in_c
		self.out_c = out_c

	def forward(self, x):
		out = self.block(x)
		out += self.down_sample(x)
		return out


class EncoderModel(nn.Module):
	def __init__(self, hidden_size, num_layers=1, dropout=0, use_cuda=False):
		super(EncoderModel, self).__init__()
		self.use_cuda = use_cuda
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.layer1 = nn.Sequential(
			# ResBlock(3,32),
			nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=(3, 4), stride=(3, 2)),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			# nn.Dropout2d(dropout)
		)
		self.layer2 = nn.Sequential(
			# ResBlock(32,32),
			nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=(4, 3), stride=(4, 2)),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			# nn.Dropout2d(dropout)
		)
		self.layer3 = nn.Sequential(
			# ResBlock(32,32),
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=(4, 2), stride=(1, 1)),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.gru = nn.GRU(128, hidden_size // 2, num_layers, batch_first=True, bidirectional=True)

	def forward(self, x, hidden):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = out.squeeze(2)
		out = out.transpose(1, 2)
		out, hidden = self.gru(out, hidden)
		return out

	def init_hidden(self, batch_size):
		h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size // 2))
		if self.use_cuda:
			return h0.cuda()
		else:
			return h0


class CTCModel(nn.Module):
	def __init__(self, output_size, num_layers, hidden_size, dropout=0, use_cuda=False):
		super(CTCModel, self).__init__()
		self.encoder = EncoderModel(hidden_size, num_layers, dropout, use_cuda)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, inputs):
		batch_size = inputs.size()[0]
		init_hidden = self.encoder.init_hidden(batch_size)
		outputs = self.encoder(inputs, init_hidden)
		outputs = self.linear(outputs)
		return outputs

	def accuracy(self, outputs, targets, seq_lens, blank=0):
		predict = self.decode_ctc_outputs(outputs, blank)
		targets = np.split(targets.data.cpu().numpy(), seq_lens.data.cpu().numpy().cumsum())[:-1]
		accuracy = np.array([np.array_equal(targets[i], predict[i]) for i in range(len(predict))]).mean()
		return accuracy, predict

	@staticmethod
	def decode_ctc_outputs(outputs, blank=0):
		"""
		ctc 解码层
		:param outputs: batch × T * D
		:param blank:
		:return:
		"""
		outputs = outputs.max(dim=-1)[1].cpu().data.numpy()  # B * T
		seq_len = outputs.shape[1]
		predict = [np.array([sample[i] for i in range(seq_len - 1) if sample[i] != sample[i - 1]] + [sample[-1]],
							dtype=np.int32) for sample in outputs]
		return [sample[sample != blank] for sample in predict]

	def save(self):
		name = "./data/ctc_model.pth"
		torch.save(self.state_dict(), name)

	def load_model(self):
		file_list = os.listdir("./data")
		if "ctc_model.pth" in file_list:
			name = "./data/ctc_model.pth"
			self.load_state_dict(torch.load(name))
			print("the latest model has been load")
