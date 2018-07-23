#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import csv

inputNum=10
train_batch_size=512

def load_data(file_name, sequence_length=inputNum, split=0.8):
    df = pd.read_csv(file_name, sep=',', usecols=[0])#使用第1列数据
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    # np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]
    test_x = x[:]
    train_y = y[: split_boundary]
    # test_y = y[split_boundary:]
    test_y = y[:]

    return train_x, train_y, test_x, test_y, scaler,split_boundary


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(activation='tanh',input_dim=1, output_dim=50, return_sequences=True))
    print(model.layers)
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size=train_batch_size, nb_epoch=800, validation_split=0.1)
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print(predict)
    print(test_y)
    try:
        fig = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    return predict, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, scaler,split_boundary = load_data('week_hi_fre2.csv')
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
    predict_y = scaler.inverse_transform([[i] for i in predict_y])
    test_y = scaler.inverse_transform(test_y)

    all_sum_err=0
    all_length=predict_y.size
    for i in range(predict_y.size):
        all_sum_err+=(abs((predict_y[i,0]-test_y[i,0])/test_y[i,0]))
    print all_length
    print "总体误差率："+str(all_sum_err/all_length)

    train_sum_err = 0
    train_length = split_boundary
    for i in range(split_boundary):
        train_sum_err += (abs((predict_y[i, 0] - test_y[i, 0]) / test_y[i, 0]))
    print train_length
    print "训练集误差率：" + str(train_sum_err / train_length)

    test_sum_err=0
    for i in range(split_boundary,predict_y.size):
        test_sum_err += (abs((predict_y[i, 0] - test_y[i, 0]) / test_y[i, 0]))
    test_length=predict_y.size-split_boundary
    print test_length
    print "测试集误差率"+str(test_sum_err/test_length)

    # plty=predict_y[split_boundary:predict_y.size,0]
    # pltt=test_y[split_boundary:predict_y.size,0]

    # fig2 = plt.figure(2)
    # plt.plot(predict_y, 'g:')
    # plt.plot(test_y, 'r-')
    # plt.savefig("总体_"+str(train_sum_err/train_length)+".jpg")
    # fig3=plt.figure(3)
    # plt.plot(plty,'g:')
    # plt.plot(pltt,'r-')
    # plt.savefig("测试_"+str(test_sum_err/test_length)+".jpg")
    # plt.show()

    csvfile = file('result.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(predict_y[:,0])
    csvfile.close()