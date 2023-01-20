#/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
import dynamic_yaml
import yaml

sys.path.append("/tf/correlation-coef-predict/ywt_library")
import data_generation
from data_generation import data_gen_cfg
from ywt_arima import arima_model, arima_err_logger_init

with open('../../config/data_config.yaml') as f:
    data = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data))

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)


# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501
logging.debug(pformat(data_cfg, indent=1, width=100, compact=True))


# # Prepare data

# ## Data implement & output setting & trainset setting

# In[2]:


# setting of output files
save_corr_data = True
save_arima_resid_data = True
# data implement setting
data_implement = "SP500_20082017_CORR_SER_REG_HRCHY_10_CLUSTER"  # watch options by operate: print(data_cfg["DATASETS"].keys())
# train set setting
train_items_setting = "-train_train"  # -train_train|-train_all
# data split  period setting, only suit for only settings of Korean paper
data_split_settings = ["-data_sp_train", "-data_sp_dev", "-data_sp_test1", "-data_sp_test2", ]
# lstm_hyper_params
lstm_hyper_param = "-kS_hyper"


# In[3]:


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
arima_result_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-arima_res"
model_dir = Path('./save_models/')
lstm_log_dir = Path('./save_models/lstm_train_logs/')
res_dir = Path('./results/')
corr_data_dir.mkdir(parents=True, exist_ok=True)
arima_result_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
lstm_log_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)


# ## Load or Create Correlation Data

# In[4]:


data_length = int(len(dataset_df)/data_gen_cfg["CORR_WINDOW"])*data_gen_cfg["CORR_WINDOW"]
corr_ser_len_max = int((data_length-data_gen_cfg["CORR_WINDOW"])/data_gen_cfg["CORR_STRIDE"])

train_df_path = corr_data_dir/f"{output_file_name}-corr_train.csv"
dev_df_path = corr_data_dir/f"{output_file_name}-corr_dev.csv"
test1_df_path = corr_data_dir/f"{output_file_name}-corr_test1.csv"
test2_df_path = corr_data_dir/f"{output_file_name}-corr_test2.csv"
all_corr_df_paths = dict(zip(["train_df", "dev_df", "test1_df", "test2_df"],
                             [train_df_path, dev_df_path, test1_df_path, test2_df_path]))
if all([df_path.exists() for df_path in all_corr_df_paths.values()]):
    corr_datasets = [pd.read_csv(df_path).set_index("items") for df_path in all_corr_df_paths.values()]
else:
    corr_datasets = data_generation.gen_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths, corr_ser_len_max=corr_ser_len_max, save_file=save_corr_data)


# # ARIMA model

# In[5]:


arima_result_path_basis = arima_result_dir/f'{output_file_name}.csv'
arima_result_paths = []
arima_result_types = ["-arima_model_info", "-arima_output", "-arima_resid"]
arima_err_logger_init(Path(os.path.abspath(''))/f"save_models")


for data_sp_setting in data_split_settings:
    for arima_result_type in arima_result_types:
        arima_result_paths.append(arima_result_dir/f'{output_file_name}{arima_result_type}{data_sp_setting}.csv')

if all([df_path.exists() for df_path in arima_result_paths]):
    pass
else:
    for (data_sp_setting, dataset) in tqdm(zip(data_split_settings, corr_datasets)):
        arima_model(dataset, arima_result_path_basis=arima_result_path_basis, data_split_setting=data_sp_setting, save_file=save_arima_resid_data)


# # LSTM

# ## settings of input data of LSTM

# In[6]:


# Dataset.from_tensor_slices(dict(pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv')))
lstm_train_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_train.csv', index_col="items").iloc[::, :-1]
lstm_train_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_train.csv', index_col="items").iloc[::, -1]
lstm_dev_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_dev.csv', index_col="items").iloc[::, :-1]
lstm_dev_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_dev.csv', index_col="items").iloc[::, -1]
lstm_test1_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test1.csv', index_col="items").iloc[::, :-1]
lstm_test1_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test1.csv', index_col="items").iloc[::, -1]
lstm_test2_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test2.csv', index_col="items").iloc[::, :-1]
lstm_test2_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test2.csv', index_col="items").iloc[::, -1]

lstm_X_len = lstm_train_X.shape[1]
lstm_Y_len = lstm_train_Y.shape[1] if len(lstm_train_Y.shape)>1 else 1
lstm_train_X = lstm_train_X.values.reshape(-1, lstm_X_len, 1)
lstm_train_Y = lstm_train_Y.values.reshape(-1, lstm_Y_len)
lstm_dev_X = lstm_dev_X.values.reshape(-1, lstm_X_len, 1)
lstm_dev_Y = lstm_dev_Y.values.reshape(-1, lstm_Y_len)
lstm_test1_X = lstm_test1_X.values.reshape(-1,  lstm_X_len, 1)
lstm_test1_Y = lstm_test1_Y.values.reshape(-1, lstm_Y_len)
lstm_test2_X = lstm_test2_X.values.reshape(-1,  lstm_X_len, 1)
lstm_test2_Y = lstm_test2_Y.values.reshape(-1, lstm_Y_len)


# ## settings of LSTM

# In[7]:


model_log = TensorBoard(log_dir=lstm_log_dir)
max_epoch = 10000
batch_size = 64
lstm_metrics = ['mse', 'mae']

if lstm_hyper_param == "-kS_hyper":
    lstm_layer = LSTM(units=10, kernel_regularizer=l1_l2(0.2, 0.0), bias_regularizer=l1_l2(0.2, 0.0), activation="tanh", dropout=0.1, name=f"lstm{lstm_hyper_param}")  # LSTM hyper params from 【Something Old, Something New — A Hybrid Approach with ARIMA and LSTM to Increase Portfolio Stability】


# In[8]:


def double_tanh(x):
    return (tf.math.tanh(x) *2)


def build_many_one_lstm():
    inputs = Input(shape=(20, 1))
    lstm_1 = lstm_layer(inputs)
    outputs = Dense(units=1, activation=double_tanh)(lstm_1)
    return keras.Model(inputs, outputs, name=f"lstm1_fc1{lstm_hyper_param}")


opt = keras.optimizers.Adam(learning_rate=0.0001)
lstm_model = build_many_one_lstm()
lstm_model.summary()
lstm_model.compile(loss='mean_squared_error', optimizer=opt, metrics=lstm_metrics)


# In[ ]:


res_csv_path = res_dir/f'{output_file_name}{lstm_hyper_param}_lstm_evaluation.csv'
res_csv_path.touch(exist_ok=True)
with open(res_csv_path, 'r+') as f:
    if not f.read():
        f.write("epoch,TRAIN_MSE,DEV_MSE,TEST1_MSE,TEST2_MSE,TRAIN_MAE,DEV_MAE,TEST1_MAE,TEST2_MAE")

res_df = pd.read_csv(res_csv_path)
saved_model_list = [int(p.stem[p.stem.find("epoch")+len("epoch"):]) for p in model_dir.glob('*.h5')]
epoch_start = max(saved_model_list) if saved_model_list else 1

try:
    for epoch_num in tqdm(range(epoch_start, max_epoch)):
        if epoch_num > 1:
            lstm_model = load_model(model_dir/f"{output_file_name}{lstm_hyper_param}-epoch{epoch_num - 1}.h5", custom_objects={'double_tanh':double_tanh})

        save_model = ModelCheckpoint(model_dir/f"{output_file_name}{lstm_hyper_param}-epoch{epoch_num}.h5",
                                     monitor='loss', verbose=1, mode='min', save_best_only=False)
        lstm_model.fit(lstm_train_X, lstm_train_Y, epochs=1, batch_size=batch_size, callbacks=[model_log, save_model], shuffle=True, verbose=0)

        # test the model
        score_train = lstm_model.evaluate(lstm_train_X, lstm_train_Y)
        score_dev = lstm_model.evaluate(lstm_dev_X, lstm_dev_Y)
        score_test1 = lstm_model.evaluate(lstm_test1_X, lstm_test1_Y)
        score_test2 = lstm_model.evaluate(lstm_test2_X, lstm_test2_Y)
        metrics_mse_ind = lstm_metrics.index('mse') + 1  # need to plus one, because first term of lstm_model.evaluate() is loss
        metrics_mae_ind = lstm_metrics.index('mae') + 1  # need to plus one, because first term of lstm_model.evaluate() is loss
        res_each_epoch_df = pd.DataFrame(np.array([epoch_num, score_train[metrics_mse_ind], score_dev[metrics_mse_ind],
                                                   score_test1[metrics_mse_ind], score_test2[metrics_mse_ind],
                                                   score_train[metrics_mae_ind], score_dev[metrics_mae_ind],
                                                   score_test1[metrics_mae_ind], score_test2[metrics_mae_ind]]).reshape(-1, 9),
                                         columns=["epoch", "TRAIN_MSE", "DEV_MSE", "TEST1_MSE",
                                                  "TEST2_MSE", "TRAIN_MAE", "DEV_MAE",
                                                  "TEST1_MAE", "TEST2_MAE"])
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

