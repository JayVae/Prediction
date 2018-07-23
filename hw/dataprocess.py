#coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator

file_name = u'E:\HW\练习数据\初赛文档\练习数据\csv\data_2015_2.csv'
df = pd.read_csv(file_name, header=None,names=['type','day'],sep='\t',usecols=[1,2])
colors=['gray','gold','green','blue','cyan','deeppink','brown','yellow','lightsalmon','lime','maroon','olive','sandybrown','mediumpurple','navy']
flavors=['flavor1','flavor2','flavor3','flavor4','flavor5','flavor6','flavor7','flavor8','flavor9','flavor10','flavor11','flavor12','flavor13','flavor14','flavor15']
# typegroup=df.groupby(by='type')
# a=typegroup.get_group('flavor11')['day'].value_counts().sort_index()
end_date=df['day'].max()
start_date=df['day'].min()
date_full=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=start_date, end=end_date))]
plt.figure(figsize=(16.39,9.35))
i=0
for key,group in df.groupby(by='type'):
    if key in flavors:
        key_value_list=group['day'].value_counts().sort_index()
        value= key_value_list.values
        date_time=key_value_list.index
        value_full = []
        for index in range(0, len(date_full)):
            if date_full[index] in date_time:
                tmp = key_value_list[date_full[index]]
            else:
                tmp = 0
            value_full.append(tmp)
        data_time_translation = [datetime.strptime(d, '%Y-%m-%d').date() for d in date_full]
        plt.plot(data_time_translation, value_full, colors[i],lw=1.5,label=key)
        i=i+1
        print("绘制第%d个图像 对应类型为%s"%(i,key))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 显示时间坐标的格式
autodates = AutoDateLocator()                # 时间间隔自动选取
plt.gcf().autofmt_xdate()  # 自动旋转日期标记
plt.grid(True)
plt.axis("tight")
plt.xlabel('Time')
plt.ylabel('count',size=20)
plt.title('flavor',size=20)
plt.gca().xaxis.set_major_locator(autodates)
plt.legend(loc=0)
plt.savefig(file_name[0:len(file_name)-3]+'png')
# plt.show()
plt.close('all')



