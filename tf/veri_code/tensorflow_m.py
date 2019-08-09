# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captcha.image import ImageCaptcha
import random

# number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
number = ['0', "1", "2", "3", "4", '5', '6', '7', '8', '9']


# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 's', 'y', 'z']
# ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#             'V', 'W', 'X', 'Y', 'Z']

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        # ord返回一个对应ASCII值
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        # i代表的是循环的次数，c代表的是验证码的内容
        # 如果没有错误char2pos返回的是每位数字的验证码数值，比如是3，5，9等
        # vector[i]的0+验证码number,10+验证码number,20+验证码number,30+验证码number
        idx = i * CHAR_SET_LEN + char2pos(c)
        # print i,CHAR_SET_LEN,char2pos(c),idx
        vector[idx] = 1
    return vector


def random_captcha_text(char_test=number, chaptcha_size=2):
    chaptcha_text = []
    for i in range(chaptcha_size):
        c = random.choice(char_test)
        chaptcha_text.append(c)
    return chaptcha_text


def gen_captcha_text_and_image():
    """
    随机生成验证码，把验证码生成矩阵
    :return:
    """
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    chaptcha_image = Image.open(captcha)
    # chaptcha_image.show()
    # 另外一种图片转矩阵的方式
    # image = tf.read_file("test.jpg", 'r')
    # image_tensor = tf.image.decode_jpeg(image)
    chaptcha_image = np.array(chaptcha_image)

    return captcha_text, chaptcha_image


def train_crack_captcha_cnn():
    # 经过卷积、池化、欠拟合、全连接后的输出
    output = crack_captcha_cnn()
    # 输出是40个一维向量
    # loss计算损失值
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    # tf.nn.sigmoid_cross_entropy_with_logits()函数计算交叉熵,输出的是一个向量而不是数;
    # 交叉熵刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近
    # tf.reduce_mean()函数求矩阵的均值
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    # Adam算法优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    # dimension=0的按行找最大值的下标，dimension=1按列找最大值的下标，dimension=2是一个三维结构，找二维里面的最大值
    max_idx_p = tf.argmax(predict, 2)
    max_idx_1 = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    # 判断max_idx_p和max_idx_l是否相等，返回一个布尔类型的向量
    correct_pred = tf.equal(max_idx_p, max_idx_1)
    # 计算真实值和预估值误差的平均值,既可以看出预测值和真实值的差距
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 第一步：训练好的模型参数都给保存起来，以便以后进行验证或测试，保存训练模型首先声明一个类tf.train.Saver()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    step = 0
    while True:
        # 生成训练一个batch,batch_x返回的是64行的图片的矩阵，batch_y返回的是一个验证码位置的向量
        batch_x, batch_y = get_next_batch(64)
        _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
        # print(step, loss_, _)

        if step % 100 == 0:
            batch_x_test, batch_y_test = get_next_batch(100)
            acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0})
            print(step, acc)

            if acc > 0.60:
                # 第二步：保存模型，第一个参数固定，第二个参数是设置保存的路径和名称，第三个参数是把  global_step=step
                saver.save(sess, "./crack_capcha.model")
                break

        step += 1


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    """
    卷积神经网络我们要定义好输入、输出、参数矩阵
    w_alpha和b_alpha的目的是增加噪音，随机生成偏置项
    :return:
    """
    # 把X变成一个另一个维度的张量，这是一个样本的输入， 这个shape四维是默认的，[每次给网络数据的个数，高度，宽度，通道]
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 定义一个shape为[3, 3, 1, 32]，正态分布均值为0，标准方差为1.0的张量，表达意思是做一个卷积窗口是3*3大小，通道为1，输出多少到多少的特征图
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    # 特征参数和特征图的参数是一致的，也一定为和上面的一样的为32
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # tf.cnn.conv2d()是执行卷积操作，具体参数的含义网上查询，strides一般都如此设置，他是为步长
    # 卷积核一般为1*1，3*3，5*5，卷积核越大看到数据越多，但是计算的数据就越多，一般会选择卷积核3*3
    # 此处卷积核的卷积核数量是下层的通道数
    """
    因为随着网络的加深，feature map的长宽尺寸缩小，本卷积层的每个map提取的特征越具有代表性（精华部分），
    所以后一层卷积层需要增加feature map的数量，才能更充分的提取出前一层的特征，一般是成倍增加
    """
    # tf.nn.relu是将每行非最大值置0
    # 第一层卷积
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    # 对卷积层的输出的数据池化处理, 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # 和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 防止或减轻过拟合而使用的函数, L2正则化
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # 第二层卷积
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 第三层卷积
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 全连接层
    # 每次池化过后，图片的高度和宽度均缩小为为原来的一半，经过上面三次的池化，高度和宽度均缩小8倍
    # 8*20*64都是第三层卷积的输出神经元，1024是自己定义的代表可以输出多少个特征
    image_height = int(conv3.shape[1])
    image_width = int(conv3.shape[2])
    w_d = tf.Variable(w_alpha * tf.random_normal([image_height * image_width * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    # 所以tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])的作用是把最后一层隐藏层的输出转换成一维的形式
    # dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.reshape(conv3, [-1, image_height * image_width * 64])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    # 经过全连接层输出为一维，1024个向量

    # w_out定义成一个形状为 [1024, 4 * 10] = [1024, 40]
    # out的输出为 8*10 的向量， 8代表识别结果的位数，10是每一位上可能的结果（0到9）
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out


# 生成一个训练的batch
def get_next_batch(batch_size=60):
    # 声明一个160*80张量
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    # 声明一个10*4的张量
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 判断生成的图片是否为60*160
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            # 是一张水平像素60，垂直像素160，彩色的图片
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        # 彩色图片转换为灰度图片
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./crack_capcha.model")

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1.0})
        text = text_list[0].tolist()
        return text


if "__main__" == __name__:

    train = 0
    if train == 0:
        text, image = gen_captcha_text_and_image()

        # 图像大小
        IMAGE_HEIGHT, IMAGE_WIDTH = 60, 160

        #  验证码的长度
        MAX_CAPTCHA = len(text)
        char_set = number
        #  数字的长度
        CHAR_SET_LEN = len(char_set)
        # 一个列为长*宽，行数未定的占位符,图片验证码尺寸
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        # 一个行数未定，列为为验证码长度*验证码列表的占位符，可能验证码内容
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        # 神经网络中可以随机抛弃一些点
        keep_prob = tf.placeholder(tf.float32)

        train_crack_captcha_cnn()

    if train == 1:
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        char_set = number
        CHAR_SET_LEN = len(char_set)

        text, image = gen_captcha_text_and_image()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        # plt.imshow(image)

        MAX_CAPTCHA = len(text)
        image = convert2gray(image)
        image = image.flatten() / 255

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32)

        predict_text = crack_captcha(image)
        print("正确:{} 预测:{}".format(text, predict_text))
    '''
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    '''
