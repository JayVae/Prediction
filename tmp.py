import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_excel("beijing.xlsx")
# dataset_bak = pd.read_excel("beijing.xlsx")
dataset_bak = dataset["sunshine_duration"]
values = dataset_bak.values
# 保证所有数据都是float32类型
values = values.astype('float32').reshape(-1, 1)

scaler = MinMaxScaler()
data_all = scaler.fit_transform(values)
dataset_bak = pd.DataFrame(data_all)

dataset2 = pd.read_excel("beijing.xlsx", usecols=[0,4])
# print(dataset2)
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.set_index("date", inplace=True)
values = dataset.values
#保证所有数据都是float32类型
values = values.astype('float32')
df = pd.DataFrame(values)
cols, names = [], []
# i: n_in, n_in-1, ..., 1，为滞后期数
# 分别代表t-n_in, ... ,t-1期
a = df.shift(0)
# print(a)
# print(df)
# print(a)
print("-----------------------")
colum_name = dataset.columns
for i in range(3, 0, -1):
    cols.append(df.shift(i))
    names += [('%s(t-%d)' % (j, i)) for j in colum_name]

for i in range(0, 2):
    cols.append(dataset_bak.shift(-i))
    if i == 0:
        names += [('sunshine_duration(t)')]
    else:
        names += [('sunshine_duration(t+%d)' % i)]
# print(cols)
print(names)
print("-----------------------")
agg = pd.concat(cols, axis=1)
agg.columns = names

print(agg)
df = pd.DataFrame(agg)
df.to_csv("nan.csv")
agg.dropna(inplace=True)
print(agg)
df = pd.DataFrame(agg)
df.to_csv("dropnan.csv")

