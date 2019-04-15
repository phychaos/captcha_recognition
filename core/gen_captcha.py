#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午10:52
# @作者   : Lin lifang
# @文件   : gen_captcha.py
import random
from config.parameter import *
from captcha.image import ImageCaptcha
import numpy as np
from tqdm import tqdm

from core.utils import Captcha

token_list = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F',
	'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p',
	'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z']

TOKEN_NUM = len(token_list)


def get_width():
	return int(MIN_WIDTH + 40 * random.random())


def get_height():
	return int(MIN_HEIGHT + 20 * random.random())


def get_string():
	string = ""
	char_num = random.randint(MIN_LEN, MAX_LEN)
	for kk in range(char_num):
		token_index = random.randint(0, TOKEN_NUM - 1)
		string += token_list[token_index]
	return string


def get_captcha(num, path):
	font_sizes = [x for x in range(40, 45)]
	data_name = {}
	for _ in tqdm(range(num), desc="迭代步数"):
		width, height = get_width(), get_height()
		images = ImageCaptcha(width, height, font_sizes=font_sizes)
		name = get_string()
		num = len(data_name.get(name, []))
		data_name.setdefault(name, []).append(0)
		image = images.generate_image(name)
		image.save(path + name + '_' + str(num) + '.jpg')


def convert_to_npz(image_file, npz_file):
	data = Captcha(image_file)
	text = []
	image_len = []
	images = []
	for x, y, label_len in tqdm(data.gen_image(), desc='step'):
		images.append(x)
		text.append(y)
		image_len.append(label_len)
	np.savez(npz_file, img=np.array(images), text=text, image_len=image_len)
