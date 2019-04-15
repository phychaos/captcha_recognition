# captcha_recognition
pytorch ctc 验证码识别

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

#### 训练模型
```python
if __name__ == '__main__':
	run()
```

#### 测试
>将测试文件存放在images
```python
if __name__ == '__main__':
	test()
```