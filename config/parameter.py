#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午11:05
# @作者   : Lin lifang
# @文件   : parameter.py

MIN_HEIGHT = 40
MIN_WIDTH = 100

MIN_LEN = 4
MAX_LEN = 7

ImageHeight = 48
ImageWidth = 128

VOCAB_SIZE = 37


class CTCParam:
	BATCH_SIZE = 200
	num_epoch = 100
	num_layer = 1
	num_units = 100
	dropout = 0
	clip = 5
	lr = 5e-2


class Seq2seqParam:
	BATCH_SIZE = 1000
	num_epoch = 100
	num_layer = 1
	hidden_size = 100
	embed_size = 100
	attn_model = 'dot'
	dropout = 0
	clip = 5
	lr = 5e-4


class TransformerParam:
	BATCH_SIZE = 500
	num_epoch = 100
	num_layer = 3
	hidden_size = 512
	num_heads = 8
	attn_model = 'dot'
	dropout = 0
	clip = 5
	lr = 5e-4


token2id = {
	"_": 0,
	"0": 1,
	"1": 2,
	"2": 3,
	"3": 4,
	"4": 5,
	"5": 6,
	"6": 7,
	"7": 8,
	"8": 9,
	"9": 10,
	"A": 11,
	"a": 11,
	"B": 12,
	"b": 12,
	"C": 13,
	"c": 13,
	"D": 14,
	"d": 14,
	"E": 15,
	"e": 15,
	"F": 16,
	"f": 16,
	"G": 17,
	"g": 17,
	"H": 18,
	"h": 18,
	"I": 19,
	"i": 19,
	"J": 20,
	"j": 20,
	"K": 21,
	"k": 21,
	"L": 22,
	"l": 22,
	"M": 23,
	"m": 23,
	"N": 24,
	"n": 24,
	"O": 25,
	"o": 25,
	"P": 26,
	"p": 26,
	"Q": 27,
	"q": 27,
	"R": 28,
	"r": 28,
	"S": 29,
	"s": 29,
	"T": 30,
	"t": 30,
	"U": 31,
	"u": 31,
	"V": 32,
	"v": 32,
	"W": 33,
	"w": 33,
	"X": 34,
	"x": 34,
	"Y": 35,
	"y": 35,
	"Z": 36,
	"z": 36,
	"^": 37,
	"$": 38
}
