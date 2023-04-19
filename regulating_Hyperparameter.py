from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os, copy, zipfile, re,sys
from keras import layers
from keras import models
from  keras import metrics
from keras.optimizers import RMSprop
import keras
import tensorflow as tf
from keras.models import load_model
import datetime
import pandas as pd


def au_Exp():
    now = datetime.datetime.now()
    now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    ## 准备几个参数，用于后续的自动化
    epochs_au = 50
    batch_size_au = 30
    jihuo = 'tanh'


    callback_list_test =[
        keras.callbacks.ModelCheckpoint(
        filepath= now_s+'.h5',      ##文件路径 存在当前路径下吧 还好找
        monitor= 'val_loss',         ## 监控指标
        save_best_only= True        ## 保持最佳模型
        )
    ]
    DATA_PATH = ''
    train_data_bak = pd.read_parquet(os.path.join(DATA_PATH, 'track3_train.parquet'))
    test_data_bak = pd.read_parquet(os.path.join(DATA_PATH, 'track3_a.parquet'))
    test_data_bak = test_data_bak.drop(['ID'], axis=1)
    test_data_bak -=test_data_bak.mean(axis=0)  # 将数据压缩到0-1之间
    test_data_bak /=test_data_bak.std(axis=0)

    trainLen=len(train_data_bak.index)
    train_data = train_data_bak.drop(['ID', 'wdir_2min', 'spd_2min', 'spd_inst_max'], axis=1).head(int(trainLen*0.8))
    test_data = train_data_bak.drop(['ID', 'wdir_2min', 'spd_2min', 'spd_inst_max'], axis=1).tail(trainLen-int(trainLen*0.8))

    train_data -=train_data.mean(axis=0)  # 将数据压缩到0-1之间
    train_data /=train_data.std(axis=0)
    test_data -=test_data.mean(axis=0)  # 将数据压缩到0-1之间
    test_data /=test_data.std(axis=0)

    train_lable_list = [
        train_data_bak['wdir_2min'].head(int(trainLen*0.8)),
        train_data_bak['spd_2min'].head(int(trainLen*0.8)),
        train_data_bak['spd_inst_max'].head(int(trainLen*0.8))
    ]
    test_lable_list = [
        train_data_bak['wdir_2min'].tail(trainLen-int(trainLen*0.8)),
        train_data_bak['spd_2min'].tail(trainLen-int(trainLen*0.8)),
        train_data_bak['spd_inst_max'].tail(trainLen-int(trainLen*0.8))
    ]
    indexs=0
    train_lable = train_lable_list[indexs]
    test_lable = test_lable_list[indexs]


    model = models.Sequential()
    model.add(layers.Conv1D(64,3,activation=jihuo,input_shape=(72,1)))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(32,3,activation=jihuo))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(16,3,activation=jihuo))
#     model.add(layers.MaxPooling1D(2))

#     model.add(layers.Conv1D(8,7,activation=jihuo))
    # model.add(layers.GlobalMaxPooling1D())  ## 实际效果极差1
    model.add(layers.Flatten())

    model.add(layers.Dense(16))
    model.add(layers.Dense(16))
    # model.add(layers.Dense(4))
    # model.add(layers.Dense(2))
    model.add(layers.Dense(1))

    model.summary()
    model.compile(optimizer=RMSprop(),loss='mse')
    history = model.fit(train_data,train_lable,
                            epochs=epochs_au,
                            batch_size=batch_size_au,
                            validation_data=(test_data,test_lable),
                            callbacks= callback_list_test
                            )

#     sF.drawLoss(history)  ## 绘制当前的验证曲线

    model = load_model(now_s+'.h5')
#     model = load_model('2023-03-21-16-12-56.h5')
    result_trian = model.predict(train_data)
    result_predict = model.predict(test_data)
    print(result_predict)
    aDataFrame = pd.DataFrame(result_predict)
    aDataFrame.to_csv(f'./processed_data_{indexs}.csv', index=False)

if __name__ == "__main__":
    au_Exp()