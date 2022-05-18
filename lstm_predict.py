#coding=utf-8
import pandas as pd
import numpy as np
from matplotlib import pyplot
from numpy import sqrt
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


def load_data(file_name, input_sequence_length=12, output_sequence_length=3, split=0.8, single_flag=False):
    if single_flag:
        dataset = pd.read_excel("beijing.xlsx", usecols=[0,4])
        dataset_bak = dataset["sunshine_duration"]
    else:
        dataset = pd.read_excel(file_name)
        dataset_bak = dataset["sunshine_duration"]
    # 归一化
    values = dataset_bak.values
    values = values.astype('float32').reshape(-1, 1)
    print(type(values))
    scaler2 = MinMaxScaler()
    data_all = scaler2.fit_transform(values)
    dataset_bak = pd.DataFrame(data_all)

    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset.set_index("date", inplace=True)
    colum_name = dataset.columns
    values = dataset.values
    # 保证所有数据都是float32类型
    values = values.astype('float32')

    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(values)
    df = pd.DataFrame(data_all)

    cols, names = [], []
    for i in range(input_sequence_length, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in colum_name]

    for i in range(0, output_sequence_length):
        cols.append(dataset_bak.shift(-i))
        if i == 0:
            names += [('sunshine_duration(t)')]
        else:
            names += [('sunshine_duration(t+%d)' % i)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    df = pd.DataFrame(agg)
    df.to_csv("agg.csv")
    # reshaped_data = np.array(agg).astype('float32')
    values = agg.values
    split_boundary = int(agg.shape[0] * split)
    train = values[:split_boundary, :]
    df = pd.DataFrame(train)
    df.to_csv("train.csv")
    test = values[split_boundary:, :]
    train_x = train[:, :input_sequence_length*len(colum_name)]
    test_x = test[:, :input_sequence_length*len(colum_name)]
    train_y = train[:, input_sequence_length*len(colum_name): ]
    df = pd.DataFrame(train_y)
    df.to_csv("train_y.csv")
    test_y = test[:, input_sequence_length*len(colum_name): ]

    #将输入X改造为LSTM的输入格式，即[samples,timesteps,features]
    # train_X = train_X.reshape((train_X.shape[0], n_in, n_vars))
    train_x = train_x.reshape(train_x.shape[0], input_sequence_length, len(colum_name))
    test_x = test_x.reshape(test_x.shape[0], input_sequence_length, len(colum_name))
    return train_x, train_y, test_x, test_y, scaler,split_boundary, scaler2, dataset


def build_model(train_x, train_y):
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    print(model.layers)
    # model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(train_y.shape[1]))
    # model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model



def train_model(train_x, train_y, test_x):
    model = build_model(train_x, train_y)

    try:
        history = model.fit(train_x, train_y, batch_size=100, nb_epoch=50, validation_split=0.1)
        # 画出学习过程
        p1 = pyplot.plot(history.history['loss'], color='blue', label='train')
        p2 = pyplot.plot(history.history['val_loss'], color='yellow', label='test')
        pyplot.legend(["train", "test"])
        pyplot.show()
        # 保存model
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
    except KeyboardInterrupt:
        print(predict)
    return predict

def evaluate_forecasts(test, forecasts, output_sequence_length):
    rmse_dic = {}
    for i in range(output_sequence_length):
        actual = [float(row[i]) for row in test]
        predicted = [float(forecast[i]) for forecast in forecasts]
        rmse = sqrt(mean_squared_error(np.array(actual), np.array(predicted)))
        rmse_dic['t+' + str(i+1) + ' RMSE'] = rmse
    return rmse_dic

#以原始数据为背景画出预测数据
def plot_forecasts(series, forecasts):
    #用蓝色画出原始数据集
    pyplot.plot(series.values)
    n_seq = len(forecasts[0])
    if len(forecasts[0])==1:
        pyplot.plot(forecasts[:,0])
    else:
        #用红色画出预测值
        for i in range(1,len(forecasts)+1):
            xaxis = [x for x in range(i, i+n_seq+1)]
            yaxis = [float(series.iloc[i-1,0])] + list(forecasts[i-1])
            pyplot.plot(xaxis, yaxis, color='red')
    #展示图像
    pyplot.show()

if __name__ == '__main__':
    output_sequence_length, input_sequence_length=12, 14
    train_x, train_y, test_x, test_y, scaler, split_boundary, scaler2, dataset = load_data('beijing.xlsx', output_sequence_length = output_sequence_length, single_flag=False)
    # shape=(samples, time_steps, input_dim)
    predict_y = train_model(train_x, train_y, test_x)
    sample_length = int((len(predict_y) / output_sequence_length))
    predict_y = predict_y.reshape((sample_length,output_sequence_length))
    print(type(predict_y))
    for i in range(output_sequence_length):
        reshaped_y = predict_y[:, i].reshape(-1, 1)
        reshaped_y = scaler2.inverse_transform(reshaped_y)
        predict_y[:, i] = reshaped_y[:, 0]
    test_y = scaler2.inverse_transform(test_y)
    # rmse
    rmse_dic_list = []
    rmse_dic_list.append(evaluate_forecasts(test_y, predict_y, output_sequence_length))
    df_dic = {}
    for i in range(len(rmse_dic_list)):
        df_dic['第' + str(i + 1) + '次'] = pd.Series(rmse_dic_list[i])
    rmse_df = DataFrame(df_dic)
    print(rmse_df)
    # 平均预测错误率MAE
    s = predict_y[0].shape
    erro_rate = np.zeros(s)
    for i in range(len(test_y)):
        erro_rate += abs(predict_y[i] / test_y[i] - 1)
    erro_rate_ave = erro_rate / len(test_y)
    err_df = DataFrame(pd.Series(erro_rate_ave))
    err_df.columns = ['平均预测错误率']
    err_df.index = ['超前%d步预测' % (i + 1) for i in range(output_sequence_length)]
    print(err_df)
    # 测试集前十个结果
    n_real = len(dataset) - len(test_x) - len(predict_y[0])
    y_real = DataFrame(dataset['sunshine_duration'][n_real:n_real + 10 + output_sequence_length])
    plot_forecasts(y_real, predict_y[0:10])
    # 整个测试集
    n_real = len(dataset) - len(test_x) - len(predict_y[0])
    y_real = DataFrame(dataset['sunshine_duration'][n_real:])
    plot_forecasts(y_real, predict_y)

    print(len(test_x))
    # 结果导出
    pre_df = DataFrame(predict_y)
    # 时间戳处理，让它只显示到日
    date_index = dataset.index[input_sequence_length-1+ len(train_x)-1:input_sequence_length-1+ len(train_x) + len(test_x)-1]
    pydate_array = date_index.to_pydatetime()
    date_only_array = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array)
    date_only_series = pd.Series(date_only_array)
    pre_df = pre_df.set_index(date_only_series)
    names_columns = ['sunshine_durationt+%d' % i  for i in range(output_sequence_length)]
    pre_df.columns = names_columns
    pre_df = pre_df.round(decimals=2)  # 小数点
    actual_df = DataFrame(test_y)
    names_columns = ['sunshine_durationt%d' % i for i in range(output_sequence_length)]
    actual_df.columns = names_columns
    actual_df = actual_df.set_index(date_only_series)
    actual_df = actual_df.round(decimals=2)
    writer = pd.ExcelWriter('Y-结果导出.xlsx')
    pre_df.to_excel(writer, "predict", startrow=len(test_x)+1, header=False)
    actual_df.to_excel(writer, "actual", startrow=len(test_x)+1)
    writer.save()