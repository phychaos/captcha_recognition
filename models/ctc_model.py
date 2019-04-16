#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午11:22
# @作者   : Lin lifang
# @文件   : ctc_model.py
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
from core.utils import decode_ctc_outputs


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


class CTCModel(nn.Module):
	def __init__(self, output_size, num_units=128, num_layers=1, dropout=0):
		super(CTCModel, self).__init__()
		self.num_layers = num_layers
		self.num_units = num_units
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
			nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=(4, 3), stride=(4, 2)),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			# nn.Dropout2d(dropout)
		)
		self.layer3 = nn.Sequential(
			# ResBlock(32,32),
			nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=(4, 2), stride=(1, 1)),
			nn.BatchNorm2d(32),
			nn.ReLU()
		)
		self.gru = nn.GRU(32, num_units, num_layers, batch_first=True, bidirectional=True)
		self.linear = nn.Linear(num_units * 2, output_size)

	def forward(self, x, hidden):
		h0 = hidden
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		batch_size, d, _, t = out.size()
		out = out.view(batch_size, d, t)
		out = out.transpose(1, 2)
		out, hidden = self.gru(out, h0)
		out = self.linear(out)
		return out

	def init_hidden(self, batch_size, use_cuda=False):
		h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.num_units))
		if use_cuda:
			return h0.cuda()
		else:
			return h0

	def save(self, circle):
		name = "./data/temp_ctc_model" + str(circle) + ".pth"
		torch.save(self.state_dict(), name)
		name2 = "./data/ctc_model.pth"
		torch.save(self.state_dict(), name2)

	def load_model(self):
		file_list = os.listdir("./data")
		if "ctc_model.pth" in file_list:
			name = "./data/ctc_model.pth"
			self.load_state_dict(torch.load(name))
			print("the latest model has been load")


def ctc_train(inputs, targets, lens, ctc, optimizer, criterion, clip, use_cuda=False):
	if use_cuda:
		inputs = inputs.cuda()
	optimizer.zero_grad()
	batch_size = inputs.size()[0]
	init_hidden = ctc.init_hidden(batch_size, use_cuda=use_cuda)
	ctc_outputs = ctc(inputs, init_hidden)  # seqLen * BatchSize * Hidden
	batch_size, seq_len, _ = ctc_outputs.size()
	act_lens = Variable(torch.IntTensor(batch_size * [seq_len]), requires_grad=False)
	loss = criterion(ctc_outputs.transpose(0, 1), targets, act_lens, lens)
	loss.backward()
	torch.nn.utils.clip_grad_norm_(ctc.parameters(), clip)
	optimizer.step()

	# TODO
	decoded_outputs = decode_ctc_outputs(ctc_outputs)
	decoded_targets = np.split(targets.data.cpu().numpy(), lens.data.cpu().numpy().cumsum())[:-1]
	accuracy = np.array([np.array_equal(decoded_targets[i], decoded_outputs[i]) for i in range(batch_size)]).mean()

	return loss.item(), accuracy


def ctc_evaluate(inputs, targets, lens, ctc, criterion, clip, use_cuda=False):
	if use_cuda:
		inputs = inputs.cuda()
	ctc.train(False)
	batch_size = inputs.size()[0]
	init_hidden = ctc.init_hidden(batch_size, use_cuda=use_cuda)
	ctc_outputs = ctc(inputs, init_hidden)  # seqLen * BatchSize * Hidden
	batch_size, seq_len, _ = ctc_outputs.size()
	act_lens = Variable(torch.IntTensor(batch_size * [seq_len]), requires_grad=False)
	loss = criterion(ctc_outputs.transpose(0, 1), targets, act_lens, lens)

	# TODO
	decoded_outputs = decode_ctc_outputs(ctc_outputs)
	decoded_targets = np.split(targets.data.cpu().numpy(), lens.data.cpu().numpy().cumsum())[:-1]
	accuracy = np.array([np.array_equal(decoded_targets[i], decoded_outputs[i]) for i in range(batch_size)]).mean()

	ctc.train(True)
	return loss.item(), accuracy, decoded_outputs


def ctc_test(inputs, ctc, use_cuda=False):
	if use_cuda:
		inputs = inputs.cuda()
	ctc.train(False)
	batch_size = inputs.size()[0]
	init_hidden = ctc.init_hidden(batch_size, use_cuda=use_cuda)
	ctc_outputs = ctc(inputs, init_hidden)

	# TODO
	decoded_outputs = decode_ctc_outputs(ctc_outputs)
	return decoded_outputs[0]
