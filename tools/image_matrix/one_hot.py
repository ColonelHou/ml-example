

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#原始数据集
data = [['自有房',40,50000],
        ['无自有房',22,13000],
        ['自有房',30,30000]]
data = pd.DataFrame(data,columns=['house','age','income'])
print(data)

#把带中文的标称属性转换为数值型，因为one-hot编码也需要先转换成数值型，用简单整数代替即可
listUniq = data.loc[:,'house'].unique()
for j in range(len(listUniq)):
    data.loc[:,'house'] = data.loc[:,'house'].apply(lambda x:j if x==listUniq[j] else x)
print(data)

#进行one-hot编码
tempdata = data[['house']]
print(tempdata)
enc = OneHotEncoder()
enc.fit(tempdata)

#one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
tempdata = enc.transform(tempdata).toarray()
print(tempdata)
print('取值范围整数个数：',enc.n_values_)

#再将二维数组转换为DataFrame，记得这里会变成多列
tempdata = pd.DataFrame(tempdata,columns=['house']*len(tempdata[0]))
print(tempdata)
