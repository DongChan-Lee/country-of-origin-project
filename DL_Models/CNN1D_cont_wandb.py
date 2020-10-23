# title: Convolutional Neural Network (CNN)
# author: Lee Dong Chan, Lee Sun Yeop

# <목차>
# 1. 라이브러리 임포트
# 2. wandb 설정
# 3. 데이터 로드
# 4. 전체데이터를 트레이닝셋, 테스트셋으로 분리
# 5. 모델 빌딩
# 6. 모델 설정
# 7. 모델 피팅


# 1. 라이브러리 임포트
import numpy as np
import pandas as pd
import pickle
import os
from random import shuffle

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, GlobalMaxPool1D, MaxPool1D, AvgPool1D, Masking
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
import wandb
from wandb.keras import WandbCallback

# 2. wandb 설정  ★★★
## 1. Login by typing 'wandb login' in the terminal
## 2. Retrieve an API key from the link and enter it in the terminal
## 3. run the wandb.init code below
## 4. view project by clicking the link
## 5. sweep in the wandb project page
## 6. set the desired range for hyperparameters and match the program name to this python code file
## 7. initialize sweep and copy the command to the terminal
## 8. To terminate the sweep early, add --count 100, or the number of runs you want, at the end of the command above

# 처음 실행시킬 때 적용될 하이퍼파라미터 초기값 설정단계
default_config = {'filters1' : 20, 'filters2' : 20, 'kernel_size1' : 3, 'kernel_size2' : 2, 'strides1' : 1, 'strides2' : 1,
                  'hidden1' : 32,  'dropout_rate' : 0.2, 'learning_rate' : 0.001, 'batch_size' : 32, 
                  'epochs' : 100}
wandb.init(project="Final_wandb_origin_CNN_MSE", config=default_config)  # 초기값을 적용하여 wandb 실행
config = wandb.config

# 디렉토리 설정
os.chdir(r"C:\Users\PC\Desktop\제출용\데이터") # 데이터 폴더로 디렉토리 설정

# 3. 데이터 로드
x = np.load('Deep_input_X_scaled_full_36.npy')  # (16235, 108, 36)
y = np.load('Deep_input_Y_MSE_scaled.npy')  # (16235, 1) ; 회귀 mse
# y = np.load('Deep_input_Y_BINARY.npy')  # (16235, 1) ; 이진분류 binary

# 4. 전체데이터를 트레이닝셋, 테스트셋으로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)
# print(x_train.shape, y_train.shape, 'train examples')
# print(x_val.shape, y_val.shape, 'train examples')
# print(x_test.shape, y_test.shape, 'train examples')

# 5. 모델 빌딩
# 하이퍼파라미터 설정
n_features = x.shape[2]

model = Sequential()
## convolutional layer
model.add(Masking(mask_value=-1, input_shape=(108, n_features))) # subsequent layers must be able to take masking function
model.add(Conv1D(filters=config.filters1, kernel_size=config.kernel_size1, padding='valid', strides=config.strides1, input_shape=(108, n_features)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D())
model.add(Conv1D(filters=config.filters2, kernel_size=config.kernel_size2, padding='valid', strides=config.strides2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalMaxPool1D())
## dense layer
# model.add(Flatten())
model.add(Dense(config.hidden1, activation='relu'))
model.add(Dropout(config.dropout_rate))
# output layer
model.add(Dense(units=1, activation='linear'))  # 회귀 mse
# model.add(Dense(units=1, activation='sigmoid')) # 이진분류 binary 

model.summary()


# 6. 모델 설정
rmse = tf.keras.metrics.RootMeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                               amsgrad=False, name="Adam")
model.compile(loss='mse', optimizer=opt, metrics=[rmse])  # 회귀 mse
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])  # 이진분류 binary
callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto', restore_best_weights=False)
callbacks_list = [callback, WandbCallback()]

# 7. 모델 피팅
history = model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.epochs, validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=2)