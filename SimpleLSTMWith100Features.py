#!/usr/bin/env python
# coding: utf-8

# In[ ]:




### load Data 
import tensorflow as tf
import gzip
from time import time
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import keras as ks
import keras
import numpy as np
import tensorflow.keras.backend as K
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
from keras.layers.merge import concatenate
#from tensorflow.keras.utils import plot_model
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
from tensorflow.python.framework import ops
ops.reset_default_graph()

# In[ ]:




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


# In[ ]:



multiprocessing.cpu_count()



# In[ ]:



def data():
    readConfigForLoading=pd.read_csv('/beegfs/sr4376/Finance Data/ModelConfig/ConfigLSTMSimple100.csv')
    Year=readConfigForLoading['Year'][0]
    lookBackYear=readConfigForLoading['lookBackYear'][0]
    LSTMWindow = readConfigForLoading['LSTMWindow'][0]
    NumberOfFeatures = readConfigForLoading['NumberOfFeatures'][0]
    
    
    with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/FeatureYear' + str(Year) +'lookBackYear' +str(lookBackYear) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
        X_train=pickle.load( handle)
    
    with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/TargetYear' + str(Year) +'lookBackYear' +str(lookBackYear) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
        y_train=pickle.load( handle)

        
    #y_train=np.load('/beegfs/sr4376/Finance Data/hyperopt/hyperas/tempOpt/tempYtrainHyper5.pkl.npy')
    #X_test=np.load('/beegfs/sr4376/Finance Data/hyperopt/hyperas/tempOpt/tempXtestHyper5.pkl.npy')
    #y_test=np.load('/beegfs/sr4376/Finance Data/hyperopt/hyperas/tempOpt/tempYtestHyper5.pkl.npy')
    print(1)
    #, X_test, y_test
    return X_train, y_train


# In[ ]:



def create_model(X_train, y_train):
    readConfigForLoading=pd.read_csv('/beegfs/sr4376/Finance Data/ModelConfig/ConfigLSTMSimple100.csv')
    length = readConfigForLoading['LSTMWindow'][0]
    n_features = readConfigForLoading['NumberOfFeatures'][0]
    def simple_sharpe_loss_function(y_actual,y_predicted):
        M=52
        M=K.cast(M,dtype='float32')
        sharpe_loss_value=K.mean(y_actual*y_predicted)/K.std(y_actual*y_predicted)*K.sqrt(M)
        return sharpe_loss_value
    
    model = Sequential()
    model.add(LSTM(units={{choice([5, 10, 20, 40, 60, 80, 100, 120])}}, input_shape=(length,n_features),recurrent_dropout={{choice([0,0.1,0.2,0.3,0.4,0.5])}}))
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(units={{choice([5, 10, 20])}}))
    model.add(Dense(1,activation='linear'))
    
    opt=Adam(lr={{choice([0.00001,0.0001,0.001,0.01,0.1,1])}},clipnorm={{choice([0.0001,0.001,0.01,0.1,1,10])}})
    model.compile(loss=simple_sharpe_loss_function, optimizer=opt)
    model.summary()
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=2,patience=25) 
#    checkpoint = ModelCheckpoint('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/BestModel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=10) 
#    tensorboard = TensorBoard(log_dir=r"D:\ML for Finance\data\logs\{}".format(time()),histogram_freq=10,write_graph=True,write_images=True,update_freq="epoch")
    #,tensorboard
    callback_List = [es]            
    result=model.fit(X_train, y_train, batch_size=6000, epochs=30,callbacks = callback_List, validation_split=0.1,verbose=2)
    validation_acc = np.amin(result.history['val_loss'])
    print('Best validation acc of epoch:', -validation_acc)
    return {'loss': validation_acc,'status': STATUS_OK,'model':model}


# In[ ]:


def continueToTrainModelLSTM(params):
    readConfigForLoading=pd.read_csv('/beegfs/sr4376/Finance Data/ModelConfig/ConfigLSTMSimple100.csv')
    length = readConfigForLoading['LSTMWindow'][0]
    n_features = readConfigForLoading['NumberOfFeatures'][0]
    unitsToChoice = [5, 10, 20, 40, 60, 80, 100, 120]
    learningRateToChoice = [0.00001,0.0001,0.001,0.01,0.1,1]
    clipnormToChoice = [0.0001,0.001,0.01,0.1,1,10]
    recurrent_dropout = [0,0.1,0.2,0.3,0.4,0.5]
    unitsAfterLSTMToChoice = [5, 10, 20]
    def simple_sharpe_loss_function(y_actual,y_predicted):
        M=52
        M=K.cast(M,dtype='float32')
        sharpe_loss_value=K.mean(y_actual*y_predicted)/K.std(y_actual*y_predicted)*K.sqrt(M)
        return sharpe_loss_value
    
    model = Sequential()
    model.add(LSTM(units=unitsToChoice[params['units']], input_shape=(length,n_features),recurrent_dropout=recurrent_dropout[params['recurrent_dropout']]))
    if params['recurrent_dropout_1'] == 1:
        model.add(Dense(units=unitsAfterLSTMToChoice[params['units_1']]))
    model.add(Dense(1,activation='linear'))
    
    opt=Adam(lr=learningRateToChoice[params['lr']],clipnorm=clipnormToChoice[params['clipnorm']])
    model.compile(loss=simple_sharpe_loss_function, optimizer=opt)
    model.summary()
    return model



gc.collect()
predictionPeriod=1
LSTMWindow=21
yearsBack=np.arange(1,2)
NumberOfFeatures=100
epochs=30
batch_size=6000
for jj in yearsBack:
    years=np.arange(2008,2015)
    best_model = None
    for ii in years:
        print(years)
        lowYear=ii-jj
        config=pd.DataFrame([[ii, jj ,LSTMWindow, NumberOfFeatures]],columns=['Year','lookBackYear','LSTMWindow','NumberOfFeatures'])
        config.to_csv('/beegfs/sr4376/Finance Data/ModelConfig/ConfigLSTMSimple100.csv')
        if best_model is None:
            best_run, best_model = optim.minimize(model=create_model,
                                             data=data,
                                             algo=tpe.suggest,
                                             max_evals=10,
                                             trials=Trials())
            print('best model over the optmization')
            print(best_run)
            model=best_model
            
        else:
            model=continueToTrainModelLSTM(best_run)
            es=EarlyStopping(monitor='val_loss',mode='min',verbose=2,patience=25) 
#            checkpoint = ModelCheckpoint('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/SimpleLSTMBestModel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=10) 
#        tensorboard = TensorBoard(log_dir=r"D:\ML for Finance\data\logs\{}".format(time()),histogram_freq=10,write_graph=True,write_images=True,update_freq="epoch")
        #tensorboard
            callback_List = [es]            
            
            with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/FeatureYear' + str(ii) +'lookBackYear' +str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
                X_train=pickle.load( handle)
    
            with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/TargetYear' + str(ii) +'lookBackYear' +str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
                y_train=pickle.load( handle)

            result=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,callbacks = callback_List, validation_split=0.1,verbose=2)
            validation_acc = np.amin(result.history['val_loss'])
            print('Best validation acc of epoch:', -validation_acc)

        
        
        model_json = model.to_json()
        with open('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/LSTMSimpleModel' + str(ii) + 'yearsBackHyperopt' + str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) + '.json',"w") as json_file:
            json_file.write(model_json)
        
        best_model.save_weights('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/LSTMSimpleModelWeights' + str(ii) +'yearsBackHyperopt' + str(jj) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures)  + '.h5')
          
        with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/FeatureYear' + str(ii+1) +'lookBackYear' +str(1) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
            ValidationData=pickle.load( handle)
    
        with gzip.open ('/beegfs/sr4376/Finance Data/LSTM/yearsBack/TargetYear' + str(ii+1) +'lookBackYear' +str(1) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) +  '.pklz', 'rb') as handle:
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
        pred1.to_csv('/beegfs/sr4376/Finance Data/LSTMResults/yearsBack/LSTMSimplePrediction' + str(ii) +'yearsBackHyperopt' + str(11) + 'LSTMWindow' + str(LSTMWindow) + 'NumberOfFeatures' + str(NumberOfFeatures) + '.csv')

