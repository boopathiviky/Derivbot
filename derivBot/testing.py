import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import datetime
import time
from iq import higher,lower,login,get_data,checkprofit
from training import train_data
import tensorflow as tf
import sys
import warnings
warnings.filterwarnings('ignore')
import random
import threading
import asyncio
try:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
  # Memory growth must be set before GPUs have been initialized
  print(e)

# def background(f):
#     '''
#     a threading decorator
#     use @background above the function you want to run in the background
#     '''
#     def bg_f(*a, **kw):
#         threading.Thread(target=f, args=a, kwargs=kw).start()
#     return bg_f
# bet_money = 1
# @background
# def updatebet(iq,start_bal):
#     global bet_money
#     time.sleep(63)
#     end_bal = iq.get_balance()
#     if start_bal > end_bal:
#         bet_money = int(bet_money) * int(martingale)
#         if bet_money>4:
#             bet_money=4    
#     else:
#         bet_money = 1
# def get_bet():
#     return bet_money
def preprocess_prediciton():
    df = get_data()
    
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['day'] = pd.DatetimeIndex(df['Date']).day
    df['hour'] = pd.DatetimeIndex(df['Date']).hour
    df['minute'] = pd.DatetimeIndex(df['Date']).minute

    # df.columns=['Open', 'Close', 'Low', 'High', 'volume', 'close_GBPUSD',
    #    'volume_GBPUSD', 'close_EURJPY', 'volume_EURJPY', 'close_AUDUSD',
    #    'volume_AUDUSD']
    """
    graphical analysis components
    """
    def bionic_pullback(row):
        '''
        0-5%: excellent strong candle close.
        5-10%: very strong candle close.
        10-20%: strong candle close.
        20-25%: good candle close.
        25-30%: ok candle close.
        30-35%: doubtful candle close.
        35-50% indecisive candle close.
        50-67%: weak candle close.
        67-100% very weak / reversal candle close.
        
        
        for wicks and pullback:-
        The best breakouts have candles which close near the top 20% of the candle size.
        They work ok with wicks between 20 and 30%.
        Risky between 30 and 40%.
        Turn easier into false breaks out above 40%.
        Can lead into reversals more often above 67%.
        '''
        candles_size_ratio=0
        if row.Candles == 1:
            body=(row.High-row.Close)
            pullback=(row.Close-row.Low)
            candles_size_ratio=(row.Close-row.Low)/(row.High-row.Low)
        elif row.Candles == 2:
            body=(row.Close-row.Low)
            pullback=(row.High-row.Close)
            candles_size_ratio=(row.Close-row.Low)/(row.High-row.Low)
        else:
            body=0
            pullback=0
        return (round(body,5)*100000,round(pullback,5)*100000,candles_size_ratio*100)
    def finding_candles_patterns(df):
        conditions=[
            df['Close']>df['Open'],
            df['Close']<df['Open'],
            df['Close']==df['Open']
        ]
        values=[2,1,0]#2 for up and 1 for down and 0 for doji pattern
        df['Candles'] = np.select(conditions, values)
        # candles_lst=df['Candles'].values.tolist()
        # result=containsPattern(candles_lst,3)
        # display(df)
        # print(candles_lst)
        return df
    df=finding_candles_patterns(df)
    df[["Body","Pullback","Candle_Ratio"]] = df.apply(lambda x: bionic_pullback(x), axis=1, result_type="expand")
    df=df[['Open', 'High', 'Low', 'Close','Body', 'Pullback', 'Candle_Ratio','year', 'month',
       'day', 'hour', 'minute']]
    
    
    n = 3
    m = 5
    j =15
    from scipy.signal import argrelextrema
    df['min'] = df.iloc[argrelextrema(df.Close.values, np.less_equal,
                        order=n)[0]]['Close']
    df['max'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal,
                        order=n)[0]]['Close']

    df['mmin'] = df.iloc[argrelextrema(df.Close.values, np.less_equal,
                        order=m)[0]]['Close']
    df['mmax'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal,
                        order=m)[0]]['Close']

    df['jmin'] = df.iloc[argrelextrema(df.Close.values, np.less_equal,
                        order=j)[0]]['Close']
    df['jmax'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal,
                        order=j)[0]]['Close']
    
    df['min'].loc[~df['min'].isnull()] = 1  # not nan
    df['min'].loc[df['min'].isnull()] = 0  # nan
    df['max'].loc[~df['max'].isnull()] = 1  # not nan
    df['max'].loc[df['max'].isnull()] = 0  # nan

    df['mmin'].loc[~df['mmin'].isnull()] = 1  # not nan
    df['mmin'].loc[df['mmin'].isnull()] = 0  # nan
    df['mmax'].loc[~df['mmax'].isnull()] = 1  # not nan
    df['mmax'].loc[df['mmax'].isnull()] = 0  # nan

    df['jmin'].loc[~df['jmin'].isnull()] = 1  # not nan
    df['jmin'].loc[df['jmin'].isnull()] = 0  # nan
    df['jmax'].loc[~df['jmax'].isnull()] = 1  # not nan
    df['jmax'].loc[df['jmax'].isnull()] = 0  # nan

    one_hot_data=np.array(df[['min','max']].values.tolist())
    df['label_snr']=np.argmax(one_hot_data, axis=1)

    # one_hot_data=np.array(df[['PUmin','PUmax']].values.tolist())
    # df['PUlabel_snr']=np.argmax(one_hot_data, axis=1)

    one_hot_data=np.array(df[['mmin','mmax']].values.tolist())
    df['m_snr']=np.argmax(one_hot_data, axis=1)

    one_hot_data=np.array(df[['jmin','jmax']].values.tolist())
    df['j_snr']=np.argmax(one_hot_data, axis=1)

    df['fut1'] = df["Close"].shift(3)
    df["imbalance"]=(df["Close"]-df["fut1"])*10000
    # df = df.drop(columns = {'min','max','mmin','mmax','jmin','jmax','omin','omax','fut1'})
    df = df.drop("fut1", axis=1) 

    df = df.loc[~df.index.duplicated(keep = 'first')]

    SEQ_LEN = 5
    
    def sma(price, period):
      sma = price.rolling(period).mean()
      return sma

    def ao(price, period1, period2):
        median = price.rolling(2).median()
        short = sma(median, period1)
        long = sma(median, period2)
        ao = short - long
        ao_df = pd.DataFrame(ao).rename(columns = {'Close':'ao'})
        return ao_df

    df['ao'] = ao(df['Close'], 5, 34)

    # df['MA_20'] = df['Close'].rolling(window = 20).mean() #moving average 20
    # df['MA_50'] = df['Close'].rolling(window = 50).mean() #moving average 50

    df['MA_8'] = df['Close'].rolling(window = 8).mean() #moving average 8
    df['MA_21'] = df['Close'].rolling(window = 21).mean() #moving average 21
    
    df['L14'] = df['Low'].rolling(window=14).min()
    df['H14'] = df['High'].rolling(window=14).max()
    df['%K'] = 100*((df['Close'] - df['L14']) / (df['H14'] - df['L14']) ) #stochastic oscilator
    df['%D'] = df['%K'].rolling(window=3).mean()

    df['EMA_20'] = df['Close'].ewm(span = 20, adjust = False).mean() #exponential moving average
    df['EMA_50'] = df['Close'].ewm(span = 50, adjust = False).mean()

    rsi_period = 14 
    chg = df['Close'].diff(1)
    gain = chg.mask(chg<0,0)
    df['gain'] = gain
    loss = chg.mask(chg>0,0)
    df['loss'] = loss
    avg_gain = gain.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
    avg_loss = loss.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()

    df['avg_gain'] = avg_gain
    df['avg_loss'] = avg_loss
    rs = abs(avg_gain/avg_loss)
    df['rsi'] = 100-(100/(1+rs)) #rsi index
    
    """
    Finishing preprocessing
    """
    df = df.drop(columns = {'Open','Low','High','avg_gain','avg_loss','L14','H14','gain','loss'}) #drop columns that are too correlated or are in somehow inside others

    df = df.dropna()
    df = df.fillna(method="ffill")
    df = df.dropna()
    
    df.sort_index(inplace = True)

    main_df = df

    main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    main_df.dropna(inplace=True)

    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)
    
    pred = pd.DataFrame(df_scaled,index = indexes)

    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)            
    
    for i in pred.iloc[len(pred) -SEQ_LEN :len(pred)   , :].values:
        prev_days.append([n for n in i[:]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days)])

    X = []

    for seq in sequential_data:
        X.append(seq)
    
    
    return np.array(X)

if(len(sys.argv) == 1):
    martingale = 2
    bet_money = 1
    ratio = 'EURUSD'
elif(len(sys.argv) != 4):
    print("The correct pattern is: python testing.py EURUSD (or other currency) INITIAL_BET(value starting in 1$ MIN) MARTINGALE (your martingale ratio default = 2)")
    print("\n\nEXAMPLE:\npython testing.py EURUSD 1 3")
    exit(-1)
else:
    bet_money = sys.argv[2] #QUANTITY YOU WANT TO BET EACH TIME
    ratio = sys.argv[1]
    martingale = sys.argv[3]
    
SEQ_LEN = 5  # how long of a preceeding sequence to collect for RNN, if you modify here, remember to modify in the other files too
FUTURE_PERIOD_PREDICT = 2  # how far into the future are we trying to predict , if you modify here, remember to modify in the other files too




# model = tf.keras.models.load_model('models/LSTM-best5withdate.model')
NAME = train_data() + '.model'
model = tf.keras.models.load_model(f'models/{NAME}')


i = 0
bid = True

MONEY = 10000 
trade = True

while(1):
    if i >= 10 and i % 2 == 0:
        NAME = train_data() + '.model'
        model = tf.keras.models.load_model(f'models/{NAME}')
        # model = tf.keras.models.load_model('models/LSTM-best5withdate.model')
        i = 0
        
    if datetime.datetime.now().second < 30 and i % 2 == 0: #GARANTE QUE ELE VAI APOSTAR NA SEGUNDA, POIS AQUI ELE JÁ PEGA OS DADOS DE UMA NA FRENTE,
        time_taker = time.time()
        pred_ready = preprocess_prediciton()             #LOGO, ELE PRECISA DE TEMPO PRA ELABORAR A PREVISÃO ANTES DE ATINGIR OS 59 SEGUNDOS PRA ELE
        pred_ready = pred_ready.reshape(1,SEQ_LEN,pred_ready.shape[3])      #FAZER A APOSTA, ENÃO ELE VAI TENTAR PREVER O VALOR DA TERCEIRA NA FRENTE
        result = model.predict(pred_ready)
        print('probability of PUT: ',result[0][0])
        print('probability of CALL: ',result[0][1])
        print(f'Time taken : {int(time.time()-time_taker)} seconds')
        i = i + 1  

    if datetime.datetime.now().second == 59 and i%2 == 1:
        if result[0][0] > 0.65 :
            print('PUT')
            id = asyncio.run(lower(bet_money))
            i = i + 1   
            trade = True
            print(id)
        elif result[0][1] > 0.65 :
            print('CALL')
            id = asyncio.run(higher(bet_money)) 
            i = i + 1
            trade = True
            print(id)
        else:
            trade = False
            i = i + 1
        print("trade sucesfully completed")
        if trade == True:
            time.sleep(2)
            print("checking win or lose")
            time.sleep(60)
            print(datetime.datetime.now().second)
            tempo = datetime.datetime.now().second
            while(tempo != 1): #wait till 1 to see if win or lose
                tempo = datetime.datetime.now().second
            
            print(datetime.datetime.now().second)
            profit = asyncio.run(checkprofit())

            if int(profit)>0:
                bet_money = 3
                    
            else:
                # print(f'Balance : {get_balance(iq)}')
                bet_money = int(bet_money) * int(martingale)


            