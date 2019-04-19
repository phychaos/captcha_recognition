#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-16 下午5:15
# @作者   : Lin lifang
# @文件   : gen_image.py
from config.config import TRAIN_DATA, TEST_DATA, IMAGE_DATA
from core.gen_captcha import get_captcha


def gen_image():
	get_captcha(num=200000, path=TRAIN_DATA)
	# get_captcha(num=3000, path=TEST_DATA)
	# get_captcha(num=19, path=IMAGE_DATA)


if __name__ == '__main__':
	gen_image()
