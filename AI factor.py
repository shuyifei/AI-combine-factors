import re
import pymysql
import os
import numpy as np
import pandas as pd
import pickle as pkl
import time

path_read = r'W:\tonglian_data\ohlc_fea'

def readpkl(filename):
    return pd.read_pickle(path_read+"\\"+filename+".pkl")

close = readpkl('CLOSE_PRICE_2')
open_ = readpkl('OPEN_PRICE_2')
volume = readpkl('TURNOVER_VOL')
high = readpkl('HIGHEST_PRICE_2')
low = readpkl('LOWEST_PRICE_2')
amt = readpkl("TURNOVER_VALUE")
deal = readpkl('DEAL_AMOUNT')
turn = readpkl('TURNOVER_RATE')

close.columns.name = 'ID'
liangjia = {'close':close,'open':open_,'volume':volume,'high':high,'low':low,'amt':amt,'deal':deal, \
            'turn':turn}

def get_train(liangjia):
    liangjia_dict = {}
    for name,data in liangjia.items():
        data.columns.name = 'ID'
        liangjia_dict[name] = data[967:2187]
    return liangjia_dict

def get_valid(liangjia):
    liangjia_dict = {}
    for name,data in liangjia.items():
        data.columns.name = 'ID'
        liangjia_dict[name] = data[2187:2431]
    return liangjia_dict

def get_test(liangjia):
    liangjia_dict = {}
    for name, data in liangjia.items():
        data.columns.name = 'ID'
        liangjia_dict[name] = data[2431:]
    return liangjia_dict

train_dict = get_train(liangjia)
valid_dict = get_valid(liangjia)
test_dict = get_test(liangjia)

future_ret20_train = train_dict['open'].pct_change(20,fill_method=None).shift(-21).dropna(how='all')
future_ret20_valid = valid_dict['open'].pct_change(20,fill_method=None).shift(-21).dropna(how='all')
future_ret20_test = test_dict['open'].pct_change(20,fill_method=None).shift(-21).dropna(how='all')

def get_X(liangjia_dict):
    names = []
    container = []
    for name,data in liangjia_dict.items():
        names.append(name)
        data_unstack = data.unstack().swaplevel()
        container.append(data_unstack)
    nparray_container = []
    for i in container:
        array = i.values[...,np.newaxis]
        nparray_container.append(array)
    raw_frame = pd.DataFrame(np.concatenate(nparray_container,axis=1))
    raw_frame.columns = names
    raw_frame.index = liangjia_dict['close'].unstack().swaplevel().index
    final_frame = raw_frame.sort_index().swaplevel().dropna(how='all')
    return final_frame

train = get_X(liangjia)0
train_fill = train.groupby('TRADE_DATE').fillna(train.groupby("TRADE_DATE").mean())
train_fill_ts = train_fill.sort_index()

def get_Y(return_20):
    return_20_final = return_20.unstack().swaplevel()
    Y_20 = return_20_final.sort_index()
    return Y_20

train_Y = get_Y(future_ret20_train)
train_Y_clean = train_Y.dropna()
train_Y_clean_ = train_Y_clean.reindex(train_fill_ts.index).dropna()
train_Y_clean__ = train_Y_clean_.replace(np.inf,np.nan).dropna()

valid_Y = get_Y(future_ret20_valid)
valid_Y_clean = valid_Y.dropna()
valid_Y_clean_ = valid_Y_clean.reindex(train_fill_ts.index).dropna()
valid_Y_clean__ = valid_Y_clean_.replace(np.inf,np.nan).dropna()

train_fill_ts_clean = train_fill_ts.reindex(index = train_Y_clean__.index)
valid_fill_ts_clean = train_fill_ts.reindex(index = valid_Y_clean__.index)

def get_np_data(train_fill_ts_clean):
    count = 0
    index_container = []
    data_container = []
    ID_len = train_fill_ts_clean.groupby("ID").count().iloc[:,0]
    for i in range(len(ID_len)):
        individual = train_fill_ts_clean.iloc[count:count+ID_len[i],:]
        count += ID_len[i]
    for i in range(40,len(individual)+1):
        index = individual.index[i-1]
        data = individual.index[i-1]
        index_container.append(index)
        data_container.append(data)
    train_data = np.array(data_container,dtype = np.float32)
    train_index = np.array(index_container)
    return train_data,train_index

train_data,train_index = get_np_data(train_fill_ts_clean)
valid_data,valid_index = get_np_data(valid_fill_ts_clean)

train_indexs = pd.Index(train_index)
train_Y_final = train_Y_clean_.reindex(train_indexs)
train_Y_final = train_Y_final.values.reshape(-1,1).astype(np.float32)

valid_indexs = pd.Index(valid_index)
valid_Y_final = valid_Y_clean_.reindex(valid_indexs)
valid_Y_final = valid_Y_final.values.reshape(-1,1).astype(np.float32)

np.save('train_X.npy',train_data)
np.save('train_Y.npy',train_Y_final)
np.save('train_index.npy',train_index)

np.save('valid_X.npy',valid_data)
np.save('valid_Y.npy',valid_Y_final)
np.save('valid_index.npy',valid_index)

from keras import Sequential
from keras.layers import LSTM
from keras import models
from keras.layers import Dense
from keras import callbacks

train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
valid_X = np.load('valid_X.npy')
valid_Y = np.load('valid_Y.npy')

print(train_X.shape)
print(train_Y.shape)
print(valid_X.shape)
print(valid_Y.shape)

def standard(X):
    X_sd = (X - X.mean(axis=1).reshape(X.shape[0],-1,X.shape[2])) / (X.std(axis=1).reshape(X.shape[0],-1,X.shape[2])+1e-7)
    return X_sd

train_X_sd = standard(train_X)
valid_X_sd = standard(valid_X)

callbacks_list = [
    callbacks.EarlyStopping(monitor='mse',patience=20),
    callbacks.ModelCheckpoint(filepath='my_model2.h5',monitor='val_loss',save_best_only=True),0
    callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5)
]
model = models.Sequential()
model.add(LSTM(4,input_shape=(40,8)))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])
history = model.fit(train_X_sd,train_Y,batch_size=512,epochs=40,verbose=1,validation_data=(valid_X_sd,valid_Y),callbacks=callbacks_list)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.figure()
plt.plot(epochs,loss,'bo',label='Training_loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title("Training and validation loss")
plt.show()

from keras.models import load_model
model_load = load_model('initial_model.h5')

valid_Y_pred = model_load.predict(valid_X_sd)
valid_index_tuple = pd.MultiIndex.from_tuples([tuple(X) for x in valid_index.tolist()])
valid_Y_factor = pd.DataFrame(valid_Y_pred,index=valid_index_tuple).unstack().T.droplevel(0)
ha.analyse(valid_Y_factor,periods = [1,5,10,20,40,60])

from xgboost import XGBRegressor
xgb = XGBRegressor(
    learning_rate = 0.2,
    max_depth = 3,
    gamma = 0.1,
    subsample = 0.8,
    nthread = -1
).fit(train_fill_ts_clean,train_Y_clean__)

train_Y_predict = xgb.predict(train_fill_ts_clean)
valid_Y_predict = xgb.predict(valid_fill_ts_clean)

final = pd.DataFrame(train_Y_clean__)
final['predict'] = train_Y_predict
final.drop([0],axis=1,inplace=True)

xgboost_valid = final.unstack().T.droplevel(0)
xgboost_train = final.unstack().T.droplevel(0)

ha.analyse(xgboost_train, periods = [1,5,10,20,40,60])

def get_taogedata(train_X):
    train_X_md = train_X.copy()
    train_X_md = (train_X_md[:,:,[0,1,3,4]]/train_X_md[:,0,[0,0,0,0]].reshape(-1,1,4))
    train_X_big = np.log(train_X[:,:,[2,5,6,7]])
    #处理volume
    train_X_big[:,:,0][train_X_big[:,:,0]==-np.inf] = 0
    #处理amt
    train_X_big[:,:,1][train_X_big[:,:,1]==-np.inf] = 0
    #处理deal
    train_X_big[:,:,2][train_X_big[:,:,2]==-np.inf] = 0
    #处理turn
    train_X_big[:,:,3][train_X_big[:,:,3]==-np.inf] = -10
    train_X_taoge = np.concatenate([train_X_md,train_X_big],axis=2)
    return train_X_taoge

train_X_taoge = get_taogedata(train_X)
valid_X_taoge = get_taogedata(valid_X)

def lstm_model(train_X,train_Y,valid_X,valid_Y):
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss',patience=10),
        callbacks.ModelCheckpoint(filepath='my_lstmtaogemodel.h5',monitor='val_loss')
    ]
    model = models.Sequential()
    model.add(LSTM(1,input_shape=(5,8)))
    optimizer = keras.optimizers.RMSprop(learning_rate = 0.001)
    model.compile(loss='mse',optimizer = optimizer,metrics = ['mse','mae'])
    history = model.fit(train_X,train_Y,batch_size=512,epochs=20,verbose=1,validation_data=(valid_X,valid_Y))
    return model.history

history = lstm_model(train_X_taoge,train_Y,valid_X_taoge,valid_Y)
model.save('my_lstmtaogemodel.h5')

valid_Y_taogepred = model.predict(valid_X_taoge)
valid_index_tuple = pd.MultiIndex.from_tuples([tuple(x) for x in valid_index.tolist()])
valid_Y_taogefactor = pd.DataFrame(valid_Y_taogepred,index=valid_index_tuple).unstack().T.droplevel(0)
ha.analyse(train_Y_taogefactor,periods=[1,5,10,20,40,60])00

train_X5 = train_X[:,15:20,:]
valid_X5 = valid_X[:,15:20,:]

train_X5_taoge = get_taogedata(train_X5)
valid_X5_taoge = get_taogedata(valid_X5)
history = lstm_model(train_X5_taoge,train_Y,valid_X5_taoge,valid_Y)

#自定义huber loss function
def create_huber(threshold):
    def huber_fn(y_true,y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error)<threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error,squared_loss,linear_loss)
    return huber_fn

#自定义IC loss
from keras import backend as kb
def ic(y_true,y_pred):
    cov = kb.sum((y_true - kb.mean(y_true))*(y_pred-kb.mean(y_pred)))/tf.cast(len(y_true),tf.float32)
    stds = kb.std(y_true) * kb.std(y_pred)
    return -cov / stds

#神经网络训练
model = models.Sequential()
model.add(Dense(4,activation='relu',kernel_initializer='glorot_normal'))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(1))
model.compile(loss=ic,optimizer='adam',metrics = ic)
history = model.fit(train_X,train_Y,batch_size=512,epochs =3,verbose=1)

train_Y_pred = model.predict(train_X)
train_index_tuple = pd.MultiIndex.from_tuples([tuple(x) for x in train.index.tolist()])
train_Y_factor = pd.DataFrame(train_Y_pred,index = train_index_tuple).unstack().T.droplevel(0)
ha.analyse(train_Y_factor,periods=[1,5,10,20,40,60])