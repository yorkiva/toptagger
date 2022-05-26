import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import keras 
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import Adam



k_batch_size = 96

# Load inputs
df_train = pd.read_pickle("datasets/train.pkl")
df_val = pd.read_pickle("datasets/val.pkl")
x_train = df_train.loc[:, df_train.columns != 'is_signal_new']
y_train = df_train["is_signal_new"]
x_val = df_val.loc[:, df_train.columns != 'is_signal_new']
y_val = df_val["is_signal_new"]
del df_train
del df_val


model = Sequential()
model.add(Dense(300, input_dim=x_train.shape[1]))
model.add(Activation('relu'))
model.add(Dense(102))
model.add(Activation('relu'))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
                x_train,
                y_train,
                batch_size=k_batch_size,
                callbacks=[
                    EarlyStopping(
                        verbose=True,
                        patience=5,
                        monitor='val_loss'),
                    ModelCheckpoint(
                        'topdnnmodel',
                        monitor='val_loss',
                        verbose=True,
                        save_best_only=True)],
                        epochs=40,
                        validation_data=(
                    x_val,
                    y_val))

