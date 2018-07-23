#coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator
import os
import sys
import os.path

# 将同一类型的图像汇总
base_path=r'E:\HW\练习数据\初赛文档\练习数据\csv'
end_date='2016-01-31'
start_date='2015-01-01'
flavors=['flavor1','flavor2','flavor3','flavor4','flavor5','flavor6','flavor7','flavor8','flavor9','flavor10','flavor11','flavor12','flavor13','flavor14','flavor15']
fileList=[]
date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=start_date, end=end_date))]

def list_file(path):
    count=0
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.csv':
            count = count+1
            print(filename)
            fileList.append(filename)
            #fp = open(dirname+os.sep+filename,'r')
            #print(len(fp.readlines())-1)
            #fp.close()
    print('total csv file: ',count)

def flavor_month(flavor_type):
    data_ori = pd.Series()
    for file in fileList:
        df = pd.read_csv(base_path+'\\'+file, header=None, names=['type', 'day'], sep='\t', usecols=[1, 2])
        if(flavor_type in df['type'].values):
            a = df.groupby(by='type').get_group(flavor_type)['day'].value_counts().sort_index()
            data_ori=data_ori.append(a)

    date_time = list(data_ori.index)
    value_l = []
    for ind in range(0, len(date_l)):
        if date_l[ind] in date_time:
            val_tmp=date_l[ind]
            tmp = data_ori[val_tmp]
        else:
            tmp = 0
        value_l.append(tmp)
    plt.figure(figsize=(16.39, 9.35))
    data_time_translation = [datetime.strptime(d, '%Y-%m-%d').date() for d in date_l]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 显示时间坐标的格式
    # autodates = AutoDateLocator()  # 时间间隔自动选取
    plt.xticks(pd.date_range(start_date, end_date, freq='M'))  # 时间间隔
    df_tmp=pd.DataFrame(data=value_l,index=data_time_translation)
    df_tmp.to_csv(base_path+r'\全部2'+'\\'+flavor_type + '.csv')

    plt.plot(data_time_translation, value_l, 'b', lw=2.5)
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.grid(True)
    plt.axis("tight")
    plt.xlabel('Time')
    plt.ylabel('count')
    plt.title('flavor')
    # plt.gca().xaxis.set_major_locator(autodates)
    plt.legend(loc=0)
    # plt.show()
    plt.savefig(base_path+r'\全部2'+'\\'+flavor_type + '.png')
    plt.close('all')


if __name__ == '__main__':
    list_file(u'E:\HW\练习数据\初赛文档\练习数据\csv')
    for flavor in flavors:
        flavor_month(flavor)

    # plt.show()


