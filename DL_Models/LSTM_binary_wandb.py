# title: LSTM_binary_wandb
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
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, GlobalMaxPool1D, MaxPool1D, AvgPool1D, LSTM, Bidirectional, Masking
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

import wandb
from wandb.keras import WandbCallback

# 디렉토리 설정
os.chdir(r"C:\Users\PC\Desktop\제출용\데이터")  # 데이터 폴더로 디렉토리 설정

# 2. wandb 설정
## 1. Login by typing 'wandb login' in the terminal
## 2. Retrieve an API key from the link and enter it in the terminal
## 3. run the wandb.init code below
## 4. view project by clicking the link
## 5. sweep in the wandb project page
## 6. set the desired range for hyperparameters and match the program name to this python code file
## 7. initialize sweep and copy the command to the terminal
## 8. To terminate the sweep early, add --count 100, or the number of runs you want, at the end of the command above

# 처음 실행시킬 때 적용될 하이퍼파라미터 초기값 설정단계
default_config = {'learning_rate': 0.001, 'lstmhidden': 20, 'batch_size': 32}
wandb.init(project='your-own-project-name', config=default_config)  # 초기값을 적용하여 wandb 실행
config = wandb.config

# 3. 데이터 로드
lstmdata_x = np.load('Deep_input_X_scaled_36.npy')   # scaled features (16235, 108, 36)
lstmdata_y = np.load('Deep_input_Y_BINARY.npy')      # binary (16235, 1)

x = lstmdata_x
y = lstmdata_y

# 4. 전체데이터를 트레이닝셋, 테스트셋으로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
# print(x_train.shape, y_train.shape, 'train examples')
# print(x_test.shape, y_test.shape, 'test examples')

# 5. 모델 빌딩
# 하이퍼파라미터 설정
n_features = x.shape[2]

model = Sequential()
## masking layer
model.add(Masking(mask_value=-1, input_shape=(108, n_features))) # mask -1
## LSTM layer
model.add(LSTM(units=config.lstmhidden, activation='tanh', input_shape=(108, n_features)))
## output layer
model.add(Dense(units=1, activation='sigmoid'))   # binary

model.summary()

# 6. 모델 설정
binary = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')
opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                               amsgrad=False, name="Adam")
model.compile(loss=binary, optimizer=opt, metrics=['acc'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                  mode='min', restore_best_weights=False)
wandb_check = WandbCallback()
callbacks_list = [early_stop, wandb_check]

# 7. 모델 피팅
history = model.fit(x_train, y_train, epochs=100, batch_size=config.batch_size, validation_split=0.2, verbose=2,
                    callbacks=callbacks_list)

