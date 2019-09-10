

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# 4个学生的数据, 前三列[学习时长, 每天问题数, 考试分数, 预测是否优秀]
# 模型: y = θ1*X1 + θ2 * X2 + θ3 * X3 + θ4
data = np.array([[10, 3, 9, 1], [9, 1, 7, 1], [4, 0, 5.5, 0], [6, 1, 8, 1]])
print(data)
# 取前三列
#print(data[:,:3])
# 预测第四列
# print(data[:, 3:])

# 第一列
#print(data[:, 0])
# 第二列
#print(data[:, 1])
#plt.show()

st = StandardScaler()
# 归一化并训练
data_std = st.fit_transform(data[:, :3])
# 定义逻辑回归
lr = linear_model.LogisticRegression()

lr.fit(data_std, data[:, 3])
# θ1,θ2,θ3的值
print(lr.coef_)
# 截距
print(lr.intercept_)
print(data_std)
plt.scatter(data_std[:, 0], data_std[:, 1])
plt.plot(data_std[:, 0], 0.47980098 * data_std[:, 0] + 0.56706637)
plt.show()
