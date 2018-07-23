from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import numpy as np
import statsmodels.tsa.stattools as st
import pandas as pd

def test_stationarity(timeseries):
    dftest = adfuller(timeseries,autolag='AIC')
    return dftest

def proper_model(data_ts, maxLag):
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
                return init_bic, init_p, init_q, init_properModel

arr_01=[1,1,0,0,1,0,1,1,0,0,0,0,4,0,2,3,0,0,0,4,0,0,0,0,0,0,0,1,3,2,0]
alltime=pd.read_csv(r"E:\HW\练习数据\初赛文档\练习数据\csv\全部2\flavor8.csv",index_col=0)
timeseries=alltime.iloc[0:31]
res=test_stationarity(timeseries['num'])
print(res)

# order = st.arma_order_select_ic(timeseries,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
# paras=order.bic_min_order
paras=proper_model(pd.Series(arr_01), 5)
print(paras)
