import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
from iq import get_data_needed ,login
import time
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

SEQ_LEN = 8 # how long
FUTURE_PERIOD_PREDICT = 2  # how far into the future are we trying to predict

def classify(current,future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop("future", axis=1) 
    
    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    # from sklearn.preprocessing import PowerTransformer
    indexes = df.index
    df_scaled = scaler.fit_transform(df)
    
    df = pd.DataFrame(df_scaled,index = indexes)
    
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences
            sequential_data.append([np.array(prev_days), i[-1]]) 

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if  put
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # if call
            buys.append([seq, target]) 

    random.shuffle(buys)  
    random.shuffle(sells)  # shuffle 

    
    lower = min(len(buys), len(sells))  

    buys = buys[:lower]  
    sells = sells[:lower]  
    
    
    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle

    X = []
    y = []

    for seq, target in sequential_data:  
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets

    return np.array(X), y



def train_data():
    print("i am in training...")
    #actives = ['EURUSD','GBPUSD','EURJPY','AUDUSD']
    
    df = get_data_needed()
    # df.columns=['Open','Close','Low','High', 'volume','close_AUDUSD', 'volume_AUDUSD']
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
    indexes = df.index
    df=df[['Open', 'High', 'Low', 'Close','Body', 'Pullback', 'Candle_Ratio']]

    from scipy.signal import argrelextrema

    n = 3
    
    df['min'] = df.iloc[argrelextrema(df.Close.values, np.less_equal,
                        order=n)[0]]['Close']
    df['max'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal,
                        order=n)[0]]['Close']
    
    m = 6
    
    df['mmin'] = df.iloc[argrelextrema(df.Close.values, np.less_equal,
                        order=m)[0]]['Close']
    df['mmax'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal,
                        order=m)[0]]['Close']
    
    j = 12
    
    df['jmin'] = df.iloc[argrelextrema(df.Close.values, np.less_equal,
                        order=j)[0]]['Close']
    df['jmax'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal,
                        order=j)[0]]['Close']
    
    o = 18
    
    df['omin'] = df.iloc[argrelextrema(df.Close.values, np.less_equal,
                        order=o)[0]]['Close']
    df['omax'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal,
                        order=o)[0]]['Close']
    
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

    df['omin'].loc[~df['omin'].isnull()] = 1  # not nan
    df['omin'].loc[df['omin'].isnull()] = 0  # nan
    df['omax'].loc[~df['omax'].isnull()] = 1  # not nan
    df['omax'].loc[df['omax'].isnull()] = 0  # nan

    one_hot_data=np.array(df[['min','max']].values.tolist())
    df['label_snr']=np.argmax(one_hot_data, axis=1)

    one_hot_data=np.array(df[['mmin','mmax']].values.tolist())
    df['m_snr']=np.argmax(one_hot_data, axis=1)

    one_hot_data=np.array(df[['jmin','jmax']].values.tolist())
    df['j_snr']=np.argmax(one_hot_data, axis=1)

    one_hot_data=np.array(df[['omin','omax']].values.tolist())
    df['o_snr']=np.argmax(one_hot_data, axis=1)

    df = df.drop(columns = {'min','max','mmin','mmax','jmin','jmax','omin','omax'})

    FUTURE_PERIOD_PREDICT = 2
    SEQ_LEN = 8


    df = df.loc[~df.index.duplicated(keep = 'first')]
    
    df['future'] = df["Close"].shift(-FUTURE_PERIOD_PREDICT) # future prediction
    

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

    df['MA_20'] = df['Close'].rolling(window = 20).mean() #moving average 20
    df['MA_50'] = df['Close'].rolling(window = 50).mean() #moving average 50

    df['MA_8'] = df['Close'].rolling(window = 8).mean() #moving average 20
    df['MA_21'] = df['Close'].rolling(window = 21).mean() #moving average 50

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

    df = df.drop(columns = {'Open','Low','High','avg_gain','avg_loss','L14','H14','gain','loss'}) #drop columns that are too correlated or are in somehow inside others

    df = df.dropna()
    dataset = df.fillna(method="ffill")
    dataset = dataset.dropna()

    dataset.sort_index(inplace = True)

    main_df = dataset

    main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    main_df.dropna(inplace=True)
    print(main_df.columns)
    main_df['target'] = list(map(classify, main_df['Close'], main_df['future']))

    main_df.dropna(inplace=True)

    main_df['target'].value_counts()

    main_df.dropna(inplace=True)

    main_df = main_df.astype('float32')

    times = sorted(main_df.index.values)
    last_5pct = sorted(main_df.index.values)[-int(0.1*len(times))]

    validation_main_df = main_df[(main_df.index >= last_5pct)]
    main_df = main_df[(main_df.index < last_5pct)]
    
    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)
    
    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"sells: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION sells: {validation_y.count(0)}, buys : {validation_y.count(1)}")
    
    train_y = np.asarray(train_y)
    validation_y = np.asarray(validation_y)
    
    
    
    
    LEARNING_RATE = 0.001 #isso mesmo
    EPOCHS = 100  # how many passes through our data #20 was good
    BATCH_SIZE = 16  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    NAME = f"{LEARNING_RATE}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-{EPOCHS}-{BATCH_SIZE}-PRED-{int(time.time())}"  # a unique name for the model
    print(NAME)
    
    
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
    
    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model = Sequential()
    model.add(LSTM(128, kernel_regularizer=l2(0.01), input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.16670
    
    model.add(LSTM(128, kernel_regularizer=l2(0.01), activation='tanh', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(2, activation='softmax'))
    
    
    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE, decay=5e-5)
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    
    filepath = 'LSTM-best'
    checkpoint_filepath = "models/{}.model".format(filepath) # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') # saves only the best ones
    
    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data =(validation_x, validation_y),
        callbacks=[tensorboard, checkpoint, earlyStoppingCallback],
    )
    
    """
    
    THIS CODE PURPOSE IS FOR ACCURACY TEST ONLY
    
    
    prediction = pd.DataFrame(model.predict(validation_x))
    
    m = np.zeros_like(prediction.values)
    m[np.arange(len(prediction)), prediction.values.argmax(1)] = 1
    
    prediction = pd.DataFrame(m, columns = prediction.columns).astype(int)
    prediction = prediction.drop(columns = {1})
    validation_y = pd.DataFrame(validation_y)
    
    high_acurate = prediction.loc[prediction[0] > 0.55] #VALORES QUE ELE PREVEU 0 COM PROB MAIOR QUE 0.55
    
    high_index = high_acurate.index     #PEGA OS INDEX DOS QUE TIVERAM PROB ACIMA DA ESPECIFICADA
    
    validation_y_used = pd.DataFrame(validation_y) #TRANSFORMA NUMPY PRA DATAFRAM
    prediction_compare  = validation_y_used.loc[high_index] #LOCALIZA OS INDEX QUE FORAM SEPARADOS
    prediction_compare[0].value_counts() #MOSTRA OS VALORES. COMO A GENTE ESCOLHEU 0 NO OUTRO O 0 TEM QUE TER UMA PROB MAIOR
    len(prediction)
    
    #acc = accuracy_score(validation_y,prediction)
    """
    
    return filepath

