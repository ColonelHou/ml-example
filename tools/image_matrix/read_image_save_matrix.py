
# coding=utf8
from PIL import Image
import numpy as np
from scipy import misc
import matplotlib.pyplot as pyplot

def loadImage():
    im = Image.open('/Users/houningning/Documents/opensource/tensorflow/hw/mnist/9.jpg')
    # im.show()
    print(im.size)
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data)          #Image类返回矩阵的操作
    data = np.reshape(data,(379, 560))   #变换成304*720
    new_im = Image.fromarray(data)     #调用Image库，数组归一化
    # new_im.show()                        #显示新图片
    misc.imsave('new_img.jpg', new_im)   #保存新图片到本地
    return data

def Writedata(data):
    filename = './egative.txt'  #数据文件保存位置
    row = np.array(data).shape[0]   #获取行数n
    with open(filename,'w') as f: # 若filename不存在会自动创建，写之前会清空文件
        for i in range(0,row):
            f.write(str(data[i][0:]))
            f.write("\n")

loadImage()
Writedata()

