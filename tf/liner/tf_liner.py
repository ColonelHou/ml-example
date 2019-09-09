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
# y = wx + b y:标签 = 权重*特征 + 偏差（y 轴截距）
y_predict = k * x + b

# 定义代价函数 损失:单个样本 模型预测准确程度, 完全正确(损失为0), 否则损失会较大
#  L2 损失(损失函数) (y - y')2; 均方误差 (MSE):所有平方损失之和 除以样本数量
loss = tf.reduce_mean(tf.square(y_predict - y))

# 定义一个梯度下降法进行训练的优化器，学习率为0.1
optimizer = tf.train.GradientDescentOptimizer(0.1)

# 最小化代价函数
# 训练模型目标: 样本上找到一组平均损失较小的 "权重" 和 "偏差"
# 收敛:直到总体损失不再变化或至少变化极其缓慢为止
train = optimizer.minimize(loss)

# 初始化全局变量
init = tf.global_variables_initializer()

# 保存模型, 保存的模型是checkpoint格式的
# servering 模型目录中是一个pb格式文件和一个名为variables的目录
# 将已经训练好的checkpoint转换为servable format
saver = tf.train.Saver()

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
    saver.save(sess, './tf/liner/model/liner_sec')
