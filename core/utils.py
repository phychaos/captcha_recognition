#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午11:23
# @作者   : Lin lifang
# @文件   : utils.py
import numpy as np
from torch.utils import data
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from config.config import TRAIN_DATA, TEST_DATA
from config.parameter import ImageHeight, ImageWidth, MAX_LEN, token2id
import re

p = re.compile('_|\.')

class Captcha(data.Dataset):
	def __init__(self, root='/', model='ctc'):
		self.model = model
		self.images_path = [os.path.join(root, img) for img in os.listdir(root)]
		self.transform = transforms.Compose([
			transforms.Resize((ImageHeight, ImageWidth)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])

	def __getitem__(self, index):
		img_path = self.images_path[index]
		label = p.split(img_path.split("/")[-1])[0]
		y_int, y_out, label_len = str2label(label, self.model)
		y_int = torch.LongTensor(y_int)
		y_out = torch.LongTensor(y_out)
		image = Image.open(img_path)
		image = self.transform(image)
		return image, y_int, y_out, label_len

	def gen_image(self, filename):
		label = p.split(filename.split("/")[-1])[0]
		y, _, label_len = str2label(label)
		y = torch.Tensor(y).type(torch.IntTensor)
		image = Image.open(filename)
		image = self.transform(image)
		return image, y, label_len, label

	def __len__(self):
		return len(self.images_path)


def str2label(str_label, model='ctc'):
	"""字符串转ID"""
	label_len = len(str_label)
	if model == 'ctc':
		target_output = str_label.ljust(MAX_LEN, "_")
		target_input = target_output
	else:
		target_input = '^' + str_label
		target_output = str_label + '$'
		target_input = target_input.ljust(MAX_LEN + 1, "_")
		target_output = target_output.ljust(MAX_LEN + 1, "_")
		label_len += 1
	target_input_id = []
	target_output_id = []
	for ints, out in zip(target_input, target_output):
		int_idx = token2id.get(ints, 0)
		out_idx = token2id.get(out, 0)
		target_input_id.append(int_idx)
		target_output_id.append(out_idx)
	return target_input_id, target_output_id, label_len


def load_dataset(batch_size, model="ctc"):
	"""
	加载数据
	:param batch_size:
	:param model:
	:return:
	"""
	train_data_set = Captcha(TRAIN_DATA, model)
	# train_data_set = Captcha(TEST_DATA, model)
	test_data_set = Captcha(TEST_DATA, model)
	train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=4)
	test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False, num_workers=4)
	return train_data_loader, test_data_loader


def generate_batch_data(x, y, z, batch_size=100):
	num_sample = len(z)
	total_batch = num_sample // batch_size
	if num_sample % batch_size > 0:
		total_batch += 1
	data_load = []
	for ii in range(total_batch):
		start, end = ii * batch_size, (ii + 1) * batch_size
		x_data = x[start:end, :, :, :]
		y_data = y[start:end]
		z_data = z[start:end]
		data_load.append([x_data, y_data, z_data])
	return data_load


def load_image(filename):
	data_captcha = Captcha()
	images, y, label_len, label = data_captcha.gen_image(filename)
	return images, y, label_len, label
