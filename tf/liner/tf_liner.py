#coding=utf-8
import tensorflow as tf
import numpy as np

print('===============')
# 使用numpy随机生成100个点
x = np.random.rand(100)
y = x * 0.1 + 0.2

# 构建一个线性模型
k = tf.Variable(0.)
b = tf.Variable(0.)
y_predict = k * x + b

# 定义代价函数
loss = tf.reduce_mean(tf.square(y_predict - y))

# 定义一个梯度下降法进行训练的优化器，学习率为0.1
optimizer = tf.train.GradientDescentOptimizer(0.1)

# 最小化代价函数
train = optimizer.minimize(loss)

# 初始化全局变量
init = tf.global_variables_initializer()

# 定义会话执行
with tf.Session() as sess:
    sess.run(init)

    # 迭代次数
    for step in range(201):
        # 训练201次
        sess.run(train)

        # 每训练20次打印K，b
        if step%20 == 0:
            print(step, sess.run([k, b]))
