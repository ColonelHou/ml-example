
# _*_ coding: utf-8 _*_

from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据  ‘MNIST_data’ 是我保存数据的文件夹的名称
mnist = input_data.read_data_sets('/Users/houningning/Documents/opensource/tensorflow/hw/mnist/', one_hot=True)

# 各种图片数据以及标签 images是图像数据  labels 是正确的结果
trainimg = mnist.train.images
trainlabels = mnist.train.labels
testimg = mnist.test.images
testlabels = mnist.test.labels

print("Type of training is %s" % (type(trainimg)))
print("Type of trainlabel is %s" % (type(trainlabels)))
print("Type of testing is %s" % (type(testimg)))
print("Type of testing is %s" % (type(testlabels)))

# 随机取一条数据, 查看图片显示
nsmaple = 1
randidx = np.random.randint(trainimg.shape[0], size=nsmaple)
print(randidx)
curr_img = np.reshape(trainimg[randidx[0],:], (28, 28))  # 数据中保存的是 1*784 先reshape 成 28*28
curr_label = np.argmax(trainlabels[randidx[0], :])
plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
# plt.show()



