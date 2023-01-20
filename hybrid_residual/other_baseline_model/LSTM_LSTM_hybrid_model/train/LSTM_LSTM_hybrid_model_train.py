#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from itertools import combinations
from pathlib import Path
import sys
import warnings
import logging
from pprint import pformat
import traceback
import os

import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA, auto_arima
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l1_l2
import dynamic_yaml
import yaml

sys.path.append("/tf/correlation-coef-predict/ywt_library")
import data_generation
from data_generation import data_gen_cfg
from ywt_arima import arima_model, arima_err_logger_init

with open('../../../config/data_config.yaml') as f:
    data = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data))

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)


# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501
logging.debug(pformat(data_cfg, indent=1, width=100, compact=True))


# # Prepare data

# ## Data implement & output setting & trainset setting

# In[ ]:


# setting of output files
save_corr_data = True
save_lstm_resid_data = True
# data implement setting
data_implement = "SP500_20082017"  # watch options by operate: print(data_cfg["DATASETS"].keys())
# train set setting
train_items_setting = "-train_train"  # -train_train|-train_all
# data split  period setting, only suit for only settings of Korean paper
data_split_settings = ["-data_sp_train", "-data_sp_dev", "-data_sp_test1", "-data_sp_test2", ]
# lstm_hyper_params
first_stage_lstm_hyper_param = "-kS_hyper"
second_stage_lstm_hyper_param = "-kS_hyper"


# In[ ]:


# data loading & implement setting
dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
dataset_df = dataset_df.set_index('Date')
all_set = list(dataset_df.columns)  # all data
train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
test_set = data_cfg['DATASETS'][data_implement]['TEST_SET'] if data_cfg['DATASETS'][data_implement].get('TEST_SET') else [p for p in all_set if p not in train_set]  # all data - train data
logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

# train items implement settings
items_implement = train_set if train_items_setting == "-train_train" else all_set
logging.info(f"===== len(train set): {len(items_implement)} =====")

# setting of name of output files and pictures title
output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
logging.info(f"===== file_name basis:{output_file_name} =====")
# display(dataset_df)

# output folder settings
corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-corr_data"
first_stage_lstm_model_dir = Path('./save_models/first_stage_lstm_weights')
first_stage_lstm_log_dir = Path('./save_models/first_stage_lstm_train_logs')
first_stage_lstm_result_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-first_stage_lstm_res"
second_stage_lstm_model_dir = Path('./save_models/second_stage_lstm_weights')
second_stage_lstm_log_dir = Path('./save_models/second_stage_lstm_train_logs')
res_dir = Path('./results/')
corr_data_dir.mkdir(parents=True, exist_ok=True)
first_stage_lstm_model_dir.mkdir(parents=True, exist_ok=True)
first_stage_lstm_log_dir.mkdir(parents=True, exist_ok=True)
first_stage_lstm_result_dir.mkdir(parents=True, exist_ok=True)
second_stage_lstm_model_dir.mkdir(parents=True, exist_ok=True)
second_stage_lstm_log_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)


# ## Load or Create Correlation Data

# In[ ]:


data_length = int(len(dataset_df)/data_gen_cfg["CORR_WINDOW"])*data_gen_cfg["CORR_WINDOW"]
corr_ser_len_max = int((data_length-data_gen_cfg["CORR_WINDOW"])/data_gen_cfg["CORR_STRIDE"])

train_df_path = corr_data_dir/f"{output_file_name}-corr_train.csv"
dev_df_path = corr_data_dir/f"{output_file_name}-corr_dev.csv"
test1_df_path = corr_data_dir/f"{output_file_name}-corr_test1.csv"
test2_df_path = corr_data_dir/f"{output_file_name}-corr_test2.csv"
all_corr_df_paths = dict(zip(["train_df", "dev_df", "test1_df", "test2_df"],
                             [train_df_path, dev_df_path, test1_df_path, test2_df_path]))
if all([df_path.exists() for df_path in all_corr_df_paths.values()]):
    corr_datasets = [pd.read_csv(df_path, index_col=["items"]) for df_path in all_corr_df_paths.values()]
else:
    corr_datasets = data_generation.gen_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths, corr_ser_len_max=corr_ser_len_max, save_file=save_corr_data)


## # LSTM model for first stage prediciton
#
## ## settings of input data of first stage LSTM
#
## In[ ]:
#
#
#first_stage_lstm_X_len = corr_datasets[0].shape[1]-1
#first_stage_lstm_Y_len = corr_datasets[0].shape[1]
#first_stage_lstm_train_X = corr_datasets[0].iloc[::, :first_stage_lstm_X_len].values.reshape(-1, first_stage_lstm_X_len, 1)
#first_stage_lstm_train_Y = corr_datasets[0].values.reshape(-1, first_stage_lstm_Y_len, 1)
#first_stage_lstm_dev_X = corr_datasets[1].iloc[::, :first_stage_lstm_X_len].values.reshape(-1, first_stage_lstm_X_len, 1)
#first_stage_lstm_dev_Y = corr_datasets[1].values.reshape(-1, first_stage_lstm_Y_len, 1)
#
#
## ## settings of first stage LSTM
#
## In[ ]:
#
#
#first_stage_lstm_model_log = TensorBoard(log_dir=first_stage_lstm_log_dir)
#first_stage_lstm_model_earlystop = EarlyStopping(patience=500, monitor="val_loss")
#first_stage_lstm_save_model = ModelCheckpoint(Path(first_stage_lstm_model_dir)/"epoch{epoch}_{val_loss:.5f}.h5",
#                                             monitor='val_loss', verbose=1, mode='min', save_best_only=False)
#first_stage_lstm_callbacks_list = [first_stage_lstm_model_log, first_stage_lstm_model_earlystop, first_stage_lstm_save_model]
#first_stage_lstm_max_epoch = 5000
#first_stage_lstm_batch_size = 64
#first_stage_lstm_metrics = ['mse', 'mae']
#
#if first_stage_lstm_hyper_param == "-kS_hyper":
#    lstm_layer = LSTM(units=10, kernel_regularizer=l1_l2(0.2, 0.0), bias_regularizer=l1_l2(0.2, 0.0), activation="tanh", dropout=0.1, name=f"lstm{first_stage_lstm_hyper_param}")  # LSTM hyper params from 【Something Old, Something New — A Hybrid Approach with ARIMA and LSTM to Increase Portfolio Stability】
#
#
## In[ ]:
#
#
#def double_tanh(x):
#    return (tf.math.tanh(x) *2)
#
#
#def build_first_stage_lstm():
#    inputs = Input(shape=(20, 1))
#    lstm_1 = lstm_layer(inputs)
#    outputs = Dense(units=21, activation=double_tanh)(lstm_1)
#    return keras.Model(inputs, outputs, name=f"lstm_lstm_first_stage-lstm1_fc1{first_stage_lstm_hyper_param}")
#
#
## inputs = Input(shape=(20, 1))
## lstm_1 = LSTM(units=20, kernel_regularizer=l1_l2(0.0, 0.0), bias_regularizer=l1_l2(0.0, 0.0))(inputs)
## outputs = Dense(units=21, activation="relu")(lstm_1)
## first_stage_lstm_model = keras.Model(inputs, outputs, name="first_stage_lstm")
#
#lstm_lstm_1st_stage_model = build_first_stage_lstm()
#lstm_lstm_1st_stage_model.summary()
#lstm_lstm_1st_stage_model.compile(loss='mean_squared_error', optimizer='adam', metrics=first_stage_lstm_metrics )
#train_history = lstm_lstm_1st_stage_model.fit(x=first_stage_lstm_train_X, y=first_stage_lstm_train_Y, validation_data=(first_stage_lstm_dev_X, first_stage_lstm_dev_Y), epochs=first_stage_lstm_max_epoch, batch_size=first_stage_lstm_batch_size, callbacks=first_stage_lstm_callbacks_list, shuffle=True, verbose=1)
#best_epoch_num = np.argmin(np.array(train_history.history['val_loss'])) + 1
#best_val_loss = train_history.history['val_loss'][best_epoch_num-1]
#best_first_stage_lstm_weight_path = first_stage_lstm_model_dir/f"epoch{best_epoch_num}_{best_val_loss:.5f}.h5"
#logging.info(f"The best first stage lstm weight is: epoch{best_epoch_num}_{best_val_loss:.5f}.h5")
#
#
## In[ ]:
#
#
#def first_stage_lstm_model(lstm_weight_path: "pathlib.PosixPath", dataset: "pd.DataFrame", first_stage_lstm_result_path_basis: "pathlib.PosixPath", data_split_setting: str = "", save_file: bool = False) -> ("pd.DataFrame", "pd.DataFrame", "pd.DataFrame"):
#    best_first_stage_lstm_model = load_model(lstm_weight_path, custom_objects={'double_tanh':double_tanh})
#    pred_input_len = dataset.shape[1]-1
#    pred_input = dataset.iloc[::, :pred_input_len].values.reshape(-1, pred_input_len, 1)
#    dataset.columns = pd.RangeIndex(dataset.shape[1])  # in order to align dataset & first_stage_lstm_output_df
#    fisrt_stage_lstm_pred = best_first_stage_lstm_model.predict(pred_input)
#    first_stage_lstm_output_df = pd.DataFrame(fisrt_stage_lstm_pred, index=dataset.index)
#    first_stage_lstm_resid_df = dataset - first_stage_lstm_output_df
#
#    if save_file:
#        first_stage_lstm_output_df.to_csv(first_stage_lstm_result_path_basis.parent/(str(first_stage_lstm_result_path_basis.stem) + f'-first_stage_lstm_output{data_split_setting}.csv'))
#        first_stage_lstm_resid_df.to_csv(first_stage_lstm_result_path_basis.parent/(str(first_stage_lstm_result_path_basis.stem) + f'-first_stage_lstm_resid{data_split_setting}.csv'))
#
#    return first_stage_lstm_output_df, first_stage_lstm_resid_df
#
#first_stage_lstm_result_path_basis = first_stage_lstm_result_dir/f'{output_file_name}.csv'
#first_stage_lstm_result_paths = []
#first_stage_lstm_result_types = ["-first_stage_lstm_output", "-first_stage_lstm_resid"]
#
#for data_sp_setting in data_split_settings:
#    for first_stage_lstm_result_type in first_stage_lstm_result_types:
#        first_stage_lstm_result_paths.append(first_stage_lstm_result_dir/f'{output_file_name}{first_stage_lstm_result_type}{data_sp_setting}.csv')
#
#if all([df_path.exists() for df_path in first_stage_lstm_result_paths]):
#    pass
#else:
#    for (data_sp_setting, dataset) in tqdm(zip(data_split_settings, corr_datasets)):
#         first_stage_lstm_model(best_first_stage_lstm_weight_path, dataset, first_stage_lstm_result_path_basis=first_stage_lstm_result_path_basis, data_split_setting=data_sp_setting, save_file=save_lstm_resid_data)
#

# # LSTM for second stage prediction (for residual)

# ## settings of input data of second stage LSTM

# In[ ]:


# Dataset.from_tensor_slices(dict(pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv')))
second_stage_lstm_train_X = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_train.csv', index_col=["items"]).iloc[::, :-1]
second_stage_lstm_train_Y = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_train.csv', index_col=["items"]).iloc[::, -1]
second_stage_lstm_dev_X = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_dev.csv', index_col=["items"]).iloc[::, :-1]
second_stage_lstm_dev_Y = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_dev.csv', index_col=["items"]).iloc[::, -1]
second_stage_lstm_test1_X = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_test1.csv', index_col=["items"]).iloc[::, :-1]
second_stage_lstm_test1_Y = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_test1.csv', index_col=["items"]).iloc[::, -1]
second_stage_lstm_test2_X = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_test2.csv', index_col=["items"]).iloc[::, :-1]
second_stage_lstm_test2_Y = pd.read_csv(first_stage_lstm_result_dir/f'{output_file_name}-first_stage_lstm_resid-data_sp_test2.csv', index_col=["items"]).iloc[::, -1]

second_stage_lstm_X_len = second_stage_lstm_train_X.shape[1]
second_stage_lstm_Y_len = second_stage_lstm_train_Y.shape[1] if len(second_stage_lstm_train_Y.shape)>1 else 1
second_stage_lstm_train_X = second_stage_lstm_train_X.values.reshape(-1, second_stage_lstm_X_len, 1)
second_stage_lstm_train_Y = second_stage_lstm_train_Y.values.reshape(-1, second_stage_lstm_Y_len)
second_stage_lstm_dev_X = second_stage_lstm_dev_X.values.reshape(-1, second_stage_lstm_X_len, 1)
second_stage_lstm_dev_Y = second_stage_lstm_dev_Y.values.reshape(-1, second_stage_lstm_Y_len)
second_stage_lstm_test1_X = second_stage_lstm_test1_X.values.reshape(-1, second_stage_lstm_X_len, 1)
second_stage_lstm_test1_Y = second_stage_lstm_test1_Y.values.reshape(-1, second_stage_lstm_Y_len)
second_stage_lstm_test2_X = second_stage_lstm_test2_X.values.reshape(-1, second_stage_lstm_X_len, 1)
second_stage_lstm_test2_Y = second_stage_lstm_test2_Y.values.reshape(-1, second_stage_lstm_Y_len)


# ## settings of second stage LSTM

# In[ ]:


second_stage_lstm_model_log = TensorBoard(log_dir=second_stage_lstm_log_dir)
second_stage_lstm_max_epoch = 5000
second_stage_lstm_batch_size = 64
second_stage_lstm_metrics = ['mse', 'mae']

if second_stage_lstm_hyper_param == "-kS_hyper":
    lstm_layer = LSTM(units=10, kernel_regularizer=l1_l2(0.2, 0.0), bias_regularizer=l1_l2(0.2, 0.0), activation="tanh", dropout=0.1, name=f"lstm{second_stage_lstm_hyper_param}")  # LSTM hyper params from 【Something Old, Something New — A Hybrid Approach with ARIMA and LSTM to Increase Portfolio Stability】


# In[ ]:


def double_tanh(x):
    return (tf.math.tanh(x) *2)


def build_second_stage_lstm():
    inputs = Input(shape=(20, 1))
    lstm_1 = lstm_layer(inputs)
    outputs = Dense(units=1, activation=double_tanh)(lstm_1)
    return keras.Model(inputs, outputs, name=f"lstm_lstm_second_stage-lstm1_fc1{second_stage_lstm_hyper_param}")

lstm_model = build_second_stage_lstm()
lstm_model.summary()
lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=second_stage_lstm_metrics)


# In[ ]:


res_csv_path = res_dir/f'{output_file_name}{second_stage_lstm_hyper_param}-second_stage_lstm_evaluation.csv'
res_csv_path.touch(exist_ok=True)
with open(res_csv_path, 'r+') as f:
    if not f.read():
        f.write("epoch,TRAIN_MSE,DEV_MSE,TEST1_MSE,TEST2_MSE,TRAIN_MAE,DEV_MAE,TEST1_MAE,TEST2_MAE")

res_df = pd.read_csv(res_csv_path)
saved_model_list = [int(p.stem[p.stem.find("epoch")+len("epoch"):]) for p in  second_stage_lstm_model_dir.glob('*.h5')]
epoch_start = max(saved_model_list) if saved_model_list else 1

try:
    for epoch_num in tqdm(range(epoch_start, second_stage_lstm_max_epoch)):
        if epoch_num > 1:
            lstm_model = load_model(second_stage_lstm_model_dir/f"{output_file_name}{second_stage_lstm_hyper_param}-epoch{epoch_num - 1}.h5", custom_objects={'double_tanh':double_tanh})

        save_model = ModelCheckpoint(second_stage_lstm_model_dir/f"{output_file_name}{second_stage_lstm_hyper_param}-epoch{epoch_num}.h5",
                                                     monitor='loss', verbose=1, mode='min', save_best_only=False)
        lstm_model.fit(second_stage_lstm_train_X, second_stage_lstm_train_Y, epochs=1, batch_size=second_stage_lstm_batch_size, callbacks=[second_stage_lstm_model_log, save_model], shuffle=True, verbose=0)

        # test the model
        score_train = lstm_model.evaluate(second_stage_lstm_train_X, second_stage_lstm_train_Y)
        score_dev = lstm_model.evaluate(second_stage_lstm_dev_X, second_stage_lstm_dev_Y)
        score_test1 = lstm_model.evaluate(second_stage_lstm_test1_X, second_stage_lstm_test1_Y)
        score_test2 = lstm_model.evaluate(second_stage_lstm_test2_X, second_stage_lstm_test2_Y)
        metrics_mse_ind = second_stage_lstm_metrics.index('mse') + 1  # need to plus one, because first term of lstm_model.evaluate() is loss
        metrics_mae_ind = second_stage_lstm_metrics.index('mae') + 1  # need to plus one, because first term of lstm_model.evaluate() is loss
        res_each_epoch_df = pd.DataFrame(np.array([epoch_num, score_train[metrics_mse_ind], score_dev[metrics_mse_ind],
                                                   score_test1[metrics_mse_ind], score_test2[metrics_mse_ind],
                                                   score_train[metrics_mae_ind], score_dev[metrics_mae_ind],
                                                   score_test1[metrics_mae_ind], score_test2[metrics_mae_ind]]).reshape(-1, 9),
                                        columns=["epoch", "TRAIN_MSE", "DEV_MSE", "TEST1_MSE", 
                                                 "TEST2_MSE", "TRAIN_MAE", "DEV_MAE",
                                                 "TEST1_MAE","TEST2_MAE"])
        res_df = pd.concat([res_df, res_each_epoch_df])
        if (res_df.shape[0] % 100) == 0:
            res_df.to_csv(res_csv_path, index=False)  # insurance for 『finally』 part doesent'work
except Exception as e:
    error_class = e.__class__.__name__  # 取得錯誤類型
    detail = e.args[0]  # 取得詳細內容
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    last_call_stack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
    file_name = last_call_stack[0]  # 取得發生的檔案名稱
    line_num = last_call_stack[1]  # 取得發生的行號
    func_name = last_call_stack[2]  # 取得發生的函數名稱
    err_msg = "File \"{}\", line {}, in {}: [{}] {}".format(file_name, line_num, func_name, error_class, detail)
    logging.error(err_msg)
else:
    pass
finally:
    res_df.to_csv(res_csv_path, index=False)
