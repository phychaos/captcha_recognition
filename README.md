# captcha_recognition
pytorch  验证码识别  
算法：
>CTC
>seq2seq + beam search/greedy search
>transformer + beam search/greedy search
>

#### 生成验证码
```Python
def gen_image():
	get_captcha(num=200000, path=TRAIN_DATA)
	get_captcha(num=1000, path=TEST_DATA)
	get_captcha(num=10, path=IMAGE_DATA)


if __name__ == '__main__':
	gen_image()
```
>num为验证码数量
>TRAIN_DATA, TEST_DATA,IMAGE_DATA 为验证码存放路径, 在config.config.py中配置

#### 训练&测试模型 
```python
if __name__ == '__main__':
	run() # 训练
	test() # 测试
```

## Result
>|                   | num_beam=1 |  num_beam=3  |
>|-------------------|------------|--------------|
>| CTC               |  0.9274    |              |
>| seq2seq           |  0.9613    |   0.9404     |
>| transformer       |  0.9644    |   0.9366     |