#!/usr/bin/env python
# coding: utf-8

# In[4]:



### load Data 
import tensorflow as tf
import gzip
from time import time
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import keras as ks
import keras
import numpy as np
import keras.backend as K
from random import random
from random import randint
from numpy import array
from numpy import zeros
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
import multiprocessing
#from eli5.sklearn import PermutationImportance
#from numba import jit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.callbacks import TensorBoard
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice
import numpy as np
import pickle 
import os
from keras.preprocessing.sequence import TimeseriesGenerator


# In[5]:



import multiprocessing
#import dask.dataframe as dk
import pandas as pd
import numpy as np
import datetime as dt

#import matplotlib.pyplot as plt
idx=pd.IndexSlice
from sklearn.metrics import make_scorer, r2_score,accuracy_score,precision_score
from sklearn.externals import joblib
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from tqdm import tqdm
import inspect 


# In[6]:



multiprocessing.cpu_count()


# In[ ]:



def data():
    readConfigForLoading=pd.read_csv('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/modelConfigInceptionLSTM100.csv')
    Year=readConfigForLoading['Year'][0]
    lookBackYear=readConfigForLoading['lookBackYear'][0]
    LSTMWindow = readConfigForLoading['LSTMWindow'][0]
    NumberOfFeatures = readConfigForLoading['NumberOfFeatures'][0]
    
    
    with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/CNNFeatureYear' + str(Year) +'lookBackYear' +str(lookBackYear) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
        X_train=pickle.load( handle)
    
    with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/CNNTargetYear' + str(Year) +'lookBackYear' +str(lookBackYear) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
        y_train=pickle.load( handle)

        
    #y_train=np.load('/beegfs/sr4376/Finance Data/hyperopt/hyperas/tempOpt/tempYtrainHyper5.pkl.npy')
    #X_test=np.load('/beegfs/sr4376/Finance Data/hyperopt/hyperas/tempOpt/tempXtestHyper5.pkl.npy')
    #y_test=np.load('/beegfs/sr4376/Finance Data/hyperopt/hyperas/tempOpt/tempYtestHyper5.pkl.npy')
    print(1)
    #, X_test, y_test
    return X_train, y_train


# In[ ]:


def create_model(X_train, y_train):
    
    def inception_module(layer_in, f1, f2, f3):

        conv1 =TimeDistributed( Conv1D(f1, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
        
        conv3 =TimeDistributed( Conv1D(f2, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
        conv3 = TimeDistributed(Conv1D(f2, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal'))(conv3)
        
        conv5 =TimeDistributed( Conv1D(f3, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
        conv5 = TimeDistributed(Conv1D(f3, kernel_size=5, padding='same', activation='relu', kernel_initializer='glorot_normal'))(conv5)
        
        pool = TimeDistributed(AveragePooling1D(pool_size=3, strides=1, padding='same'))(layer_in)
        pool =TimeDistributed( Conv1D(f1, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(pool)
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out

    
    print(1)
    APPENDweights=[]
    size=377

    readConfigForLoading=pd.read_csv('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/modelConfigInceptionLSTM100.csv')
    length = readConfigForLoading['LSTMWindow'][0]
    n_features = readConfigForLoading['NumberOfFeatures'][0]
    def simple_sharpe_loss_function(y_actual,y_predicted):
        M=52
        M=K.cast(M,dtype='float32')
        sharpe_loss_value=K.mean(y_actual*y_predicted)/K.std(y_actual*y_predicted)*K.sqrt(M)
        return sharpe_loss_value
    #,'three','four','five','six','seven'
    visible = Input(shape=(None,n_features,1))
    layer=visible
    deepInceptionLayers={{choice(['one'])}}
    if deepInceptionLayers == 'one':
        NumberOfLayers=1
    elif deepInceptionLayers == 'two':
        NumberOfLayers=2
    elif deepInceptionLayers == 'three':
        NumberOfLayers=3
    elif deepInceptionLayers == 'four':
        NumberOfLayers=4
    elif deepInceptionLayers == 'five':
        NumberOfLayers=5
    elif deepInceptionLayers == 'six':
        NumberOfLayers=6
    elif deepInceptionLayers == 'seven':
        NumberOfLayers=7
        
    filter1D={{choice([1,3])}}
    filter3D={{choice([1,3])}}
    filter5D={{choice([1,3])}}
#    pool_size={{choice([ 1,2])}}
    momentum= 0.9
    for ii in np.arange(0,NumberOfLayers):
        layer = inception_module(layer, f1=filter1D, f2=filter3D, f3=filter5D)
        layer = TimeDistributed(BatchNormalization(momentum=momentum))(layer)
#        if {{choice(['one','two'])}} == 'one':
#            layer = TimeDistributed(MaxPooling1D(pool_size=pool_size))(layer)
#        else:
#            layer = TimeDistributed(AveragePooling1D(pool_size=pool_size))(layer)
   #  10,20,30,40         
    layer = TimeDistributed(Conv1D(1, kernel_size={{choice([20])}}, activation='relu', kernel_initializer='glorot_normal'))(layer)
    layer = TimeDistributed(Flatten())(layer)
    layer= LSTM(units={{choice([5,10,20,30,40,60,80,100,120])}}, kernel_initializer='glorot_normal',bias_initializer='glorot_normal',recurrent_dropout={{choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])}})(layer)
    if {{choice(['one','two'])}}=='one':
        layer = Dense(units={{choice([5,10,20])}},activation='relu',)(layer)
    layer = Dense(1, activation='linear')(layer)
#dropout={{choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])}}
    model = Model(inputs=visible,outputs=layer)    
    opt=Adam(lr={{choice([0.00001,0.0001,0.001,0.01,0.1])}},clipnorm={{choice([0.0001,0.001,0.01,0.1,1])}})
    model.compile(loss=simple_sharpe_loss_function, optimizer=opt)
    model.summary()
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
    checkpoint = ModelCheckpoint('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/BestModel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=10)               
    callback_List = [es, checkpoint]
    result=model.fit(X_train, y_train, batch_size=6000, epochs=5,callbacks = callback_List, validation_split=0.1,verbose=2)
    validation_acc = np.amin(result.history['val_loss'])
    print('Best validation acc of epoch:', -validation_acc)
    return {'loss': validation_acc,'status': STATUS_OK,'model':model}


# In[ ]:


def continueToTrainModel(params):
    
    def inception_module(layer_in, f1, f2, f3):

        conv1 =TimeDistributed( Conv1D(f1, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
        
        conv3 =TimeDistributed( Conv1D(f2, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
        conv3 = TimeDistributed(Conv1D(f2, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal'))(conv3)
        
        conv5 =TimeDistributed( Conv1D(f3, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
        conv5 = TimeDistributed(Conv1D(f3, kernel_size=5, padding='same', activation='relu', kernel_initializer='glorot_normal'))(conv5)
        
        pool = TimeDistributed(AveragePooling1D(pool_size=3, strides=1, padding='same'))(layer_in)
        pool =TimeDistributed( Conv1D(f1, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(pool)
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out

    
    print(1)
    APPENDweights=[]
    size=377
    #,'two','three','four','five','six','seven'
    clipnormToChoice = [0.0001,0.001,0.01,0.1,1,10]
    deepInceptionLayersToPick = ['one']
    filter1D = [1,3]
    filter1D_1 = [1,3]
    filter1D_2 = [1,3]
    kernel_size =  [2]
    learningRateToChoice = [0.00001,0.0001,0.001,0.01,0.1,1]
#    pool_size = [1]
#    pool_stride = [1]

    recurrent_dropout = [0,0.1,0.2,0.3,0.4,0.5]
    recurrent_dropout_1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    recurrent_dropout_2 =['one','two']
    unitsToChoice = [5, 10, 20, 40, 60, 80, 100, 120]
    units_1 = [5, 10, 20]
    readConfigForLoading=pd.read_csv('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/modelConfigInceptionLSTM100.csv')
    length = readConfigForLoading['LSTMWindow'][0]
    n_features = readConfigForLoading['NumberOfFeatures'][0]
    def simple_sharpe_loss_function(y_actual,y_predicted):
        M=52
        M=K.cast(M,dtype='float32')
        sharpe_loss_value=K.mean(y_actual*y_predicted)/K.std(y_actual*y_predicted)*K.sqrt(M)
        return sharpe_loss_value

    visible = Input(shape=(None,n_features,1))
    layer=visible
    deepInceptionLayers=deepInceptionLayersToPick[params['deepInceptionLayers']]
    if deepInceptionLayers == 'one':
        NumberOfLayers=1
    elif deepInceptionLayers == 'two':
        NumberOfLayers=2
    elif deepInceptionLayers == 'three':
        NumberOfLayers=3
    elif deepInceptionLayers == 'four':
        NumberOfLayers=4
    elif deepInceptionLayers == 'five':
        NumberOfLayers=5
    elif deepInceptionLayers == 'six':
        NumberOfLayers=6
    elif deepInceptionLayers == 'seven':
        NumberOfLayers=7
        
    filter1D=filter1D[params['filter1D']]
    filter3D=filter1D_1[params['filter1D_1']]
    filter5D=filter1D_2[params['filter1D_2']]
#    pool_size={{choice([3,5,9,16,25,34])}}
    #pool_stride={{choice([None,1, 2,3])}}
    momentum= 0.9
    for ii in np.arange(0,NumberOfLayers):
        layer = inception_module(layer, f1=filter1D, f2=filter3D, f3=filter5D)
        layer = TimeDistributed(BatchNormalization(momentum=momentum))(layer)
#        if {{choice(['one','two'])}} == 'one':
#            layer = MaxPooling1D(pool_size=pool_size)(layer)
#        else:
#            layer = AveragePooling1D(pool_size=pool_size)(layer)
              
    layer = TimeDistributed(Conv1D(1, kernel_size=kernel_size[params['kernel_size']], activation='relu', kernel_initializer='glorot_normal'))(layer)
    layer = TimeDistributed(Flatten())(layer)
    layer= LSTM(units=unitsToChoice[params['units']], kernel_initializer='glorot_normal',bias_initializer='glorot_normal',recurrent_dropout=recurrent_dropout_1[params['recurrent_dropout']])(layer)
    if recurrent_dropout_2[params['recurrent_dropout_1']] =='one':
        layer = Dense(units=units_1[params['units_1']],activation='relu')(layer)
    layer = Dense(1, activation='linear')(layer)

    model = Model(inputs=visible,outputs=layer)    
    opt=Adam(lr=learningRateToChoice[params['lr']],clipnorm=clipnormToChoice[params['clipnorm']])
    model.compile(loss=simple_sharpe_loss_function, optimizer=opt)
    model.summary()

    return  model


# In[ ]:



modelName='InceptionLSTM'
gc.collect()
predictionPeriod=1
LSTMWindow=21
yearsBack=np.arange(1,2)
NumberOfFeatures=100
epochs=5
batch_size=4000
for jj in yearsBack:
    years=np.arange(2008,2015)
    best_model = None
    for ii in years:
        print(years)
        lowYear=ii-jj
        config=pd.DataFrame([[ii, jj ,LSTMWindow, NumberOfFeatures]],columns=['Year','lookBackYear','LSTMWindow','NumberOfFeatures'])
        config.to_csv('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/modelConfigInceptionLSTM100.csv')
        if best_model is None:
            best_run, best_model = optim.minimize(model=create_model,
                                             data=data,
                                             algo=tpe.suggest,
                                             max_evals=2,
                                             trials=Trials())
            print('best model over the optmization')
            print(best_run)
            model=best_model
            
        else:
            model=continueToTrainModel(best_run)
            es=EarlyStopping(monitor='val_loss',mode='min',verbose=2,patience=25) 
            checkpoint = ModelCheckpoint('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/BestModelInceptionLSTM.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=10) 
#        tensorboard = TensorBoard(log_dir=r"D:\ML for Finance\data\logs\{}".format(time()),histogram_freq=10,write_graph=True,write_images=True,update_freq="epoch")
        #tensorboard
            callback_List = [es, checkpoint]            
            
            with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/CNNFeatureYear' + str(ii) +'lookBackYear' +str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
                X_train=pickle.load( handle)
    
            with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/CNNTargetYear' + str(ii) +'lookBackYear' +str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
                y_train=pickle.load( handle)

            result=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,callbacks = callback_List, validation_split=0.1,verbose=2)
            validation_acc = np.amin(result.history['val_loss'])
            print('Best validation acc of epoch:', -validation_acc)

        
        
        model_json = model.to_json()
        with open('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/ ' + modelName +'Model' + str(ii) + 'yearsBackHyperopt' + str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) + '.json',"w") as json_file:
            json_file.write(model_json)
        
        best_model.save_weights('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/ ' + modelName +'ModelWeights' + str(ii) +'yearsBackHyperopt' + str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures)  + '.h5')
          
        with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/CNNFeatureYear' + str(ii+1) +'lookBackYear' +str(1) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
            ValidationData=pickle.load( handle)
    
        with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/CNNTargetYear' + str(ii+1) +'lookBackYear' +str(1) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
            ValidationTarget=pickle.load( handle)
        
        with open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/indexObjectYear' + str(ii+1) +'lookBackYear' +str(1) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.csv', 'rb') as handle:
            validationIndex=pd.read_csv( handle,parse_dates=['1'])        
#        
        validationIndex.rename(columns={'0':'entityID', '1':'date'},inplace=True)        
        validationIndex.set_index(['entityID','date'],inplace=True,drop=False)
        validationIndex.drop(columns='Unnamed: 0',inplace=True)

        
      
        pred1=best_model.predict(ValidationData, batch_size=2000)
        print(2)
        pred1=pd.DataFrame(pred1)
        pred1['targets']=ValidationTarget
        pred1['entityID']=validationIndex['entityID'].values
        pred1['date']=validationIndex['date'].values
        pred1.set_index(['entityID','date'],inplace=True)
        pred1.to_csv('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/data' + modelName +'Prediction' + str(ii) +'yearsBackHyperopt' + str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) + '.csv')

