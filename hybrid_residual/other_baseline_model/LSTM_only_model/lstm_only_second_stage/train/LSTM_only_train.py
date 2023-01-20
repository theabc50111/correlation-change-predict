#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
import warnings
import logging

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
err_log_handler = logging.FileHandler(filename="./models/arima_train_err_log.txt", mode='a')
err_logger = logging.getLogger("arima_train_err")
err_logger.addHandler(err_log_handler)

# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501


# # Prepare data

# In[2]:


# setting of output files
save_raw_corr_data = True
save_train_info_arima_resid_data = True
# data implement setting
data_implement = "sp500_20082017"  # tw50|sp500_20082017|sp500_19972007|tetuan_power
                                                          # |sp500_20082017_consumer_discretionary
# train set setting
items_setting = "train"  # train|all


# In[3]:


# data loading & implement setting
dataset_path = Path("../dataset/")
if data_implement == "tw50":
    file_name = Path("tw50_hold_20082018_adj_close_pre.csv")
    train_set = ['萬海_adj_close', '豐泰_adj_close', '友達_adj_close', '欣興_adj_close', '台塑化_adj_close', '和泰車_adj_close', '元大金_adj_close', '南電_adj_close', '台塑_adj_close', '統一超_adj_close', '台泥_adj_close', '瑞昱_adj_close', '彰銀_adj_close', '富邦金_adj_close', '研華_adj_close', '中鋼_adj_close', '鴻海_adj_close', '台新金_adj_close', '遠傳_adj_close', '南亞_adj_close', '台達電_adj_close', '台灣大_adj_close', '台化_adj_close', '聯詠_adj_close', '廣達_adj_close', '聯發科_adj_close', '台積電_adj_close', '統一_adj_close', '中信金_adj_close', '長榮_adj_close']
elif data_implement == "sp500_19972007":
    file_name = Path("sp500_hold_19972007_adj_close_pre.csv")
    train_set = ['PXD', 'WAT', 'LH', 'AMGN', 'AOS', 'EFX', 'NEM', 'CTAS', 'MAT', 'VLO', 'APH', 'ADM', 'MLM', 'BK', 'NOV', 'BDX', 'RRC', 'IVZ', 'ED', 'SBUX', 'CI', 'ZION', 'COO', 'FDX', 'GLW', 'GPC', 'HPQ', 'ADI', 'AMG', 'MTB', 'YUM', 'SYK', 'KMX', 'AME', 'BMY', 'KMB', 'JPM', 'AET', 'DLTR', 'MGM', 'FL', 'HD', 'CLX', 'OKE', 'WMB', 'IFF', 'CMS', 'MMC', 'REG', 'ES', 'ITW', 'VRTX', 'QCOM', 'MSI', 'NKTR', 'AMAT', 'BWA', 'ESRX', 'TXT', 'VNO', 'WDC', 'PVH', 'NOC', 'PCAR', 'NSC', 'PHM', 'LUV', 'HUM', 'SPG', 'SJM', 'ABT', 'ALK', 'TAP', 'CAT', 'TMO', 'AES', 'MRK', 'RMD', 'MKC', 'HIG', 'DE', 'ATVI', 'O', 'UNM', 'VMC', 'CMA', 'RHI', 'RE', 'FMC', 'MU', 'CB', 'LNT', 'GE', 'SNA', 'LLY', 'LEN', 'MAA', 'OMC', 'F', 'APA', 'CDNS', 'SLG', 'HP', 'SHW', 'AFL', 'STT', 'PAYX', 'AIG']
elif data_implement in ["sp500_20082017", "paper_eva_1", "paper_eva_2", "paper_eva_3", "paper_eva_4", "paper_eva_5"]:
    file_name = Path("sp500_hold_20082017_adj_close_pre.csv")
    train_set = ['CELG', 'PXD', 'WAT', 'LH', 'AMGN', 'AOS', 'EFX', 'CRM', 'NEM', 'JNPR', 'LB', 'CTAS', 'MAT', 'MDLZ', 'VLO', 'APH', 'ADM', 'MLM', 'BK', 'NOV', 'BDX', 'RRC', 'IVZ', 'ED', 'SBUX', 'GRMN', 'CI', 'ZION', 'COO', 'TIF', 'RHT', 'FDX', 'LLL', 'GLW', 'GPN', 'IPGP', 'GPC', 'HPQ', 'ADI', 'AMG', 'MTB', 'YUM', 'SYK', 'KMX', 'AME', 'AAP', 'DAL', 'A', 'MON', 'BRK', 'BMY', 'KMB', 'JPM', 'CCI', 'AET', 'DLTR', 'MGM', 'FL', 'HD', 'CLX', 'OKE', 'UPS', 'WMB', 'IFF', 'CMS', 'ARNC', 'VIAB', 'MMC', 'REG', 'ES', 'ITW', 'NDAQ', 'AIZ', 'VRTX', 'CTL', 'QCOM', 'MSI', 'NKTR', 'AMAT', 'BWA', 'ESRX', 'TXT', 'EXR', 'VNO', 'BBT', 'WDC', 'UAL', 'PVH', 'NOC', 'PCAR', 'NSC', 'UAA', 'FFIV', 'PHM', 'LUV', 'HUM', 'SPG', 'SJM', 'ABT', 'CMG', 'ALK', 'ULTA', 'TMK', 'TAP', 'SCG', 'CAT', 'TMO', 'AES', 'MRK', 'RMD', 'MKC', 'WU', 'ACN', 'HIG', 'TEL', 'DE', 'ATVI', 'O', 'UNM', 'VMC', 'ETFC', 'CMA', 'NRG', 'RHI', 'RE', 'FMC', 'MU', 'CB', 'LNT', 'GE', 'CBS', 'ALGN', 'SNA', 'LLY', 'LEN', 'MAA', 'OMC', 'F', 'APA', 'CDNS', 'SLG', 'HP', 'XLNX', 'SHW', 'AFL', 'STT', 'PAYX', 'AIG', 'FOX', 'MA']
elif data_implement == "tetuan_power":
    file_name = Path("Tetuan City power consumption_pre.csv")
    train_set = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows", "Zone 1 Power Consumption", "Zone 2 Power Consumption", "Zone 3 Power Consumption"]
elif data_implement == "sp500_20082017_consumer_discretionary":
    file_name = Path("sp500_hold_20082017_adj_close_pre_consumer_discretionary.csv")
    train_set = ['LKQ', 'LEN', 'TGT', 'YUM', 'TJX', 'GRMN', 'MCD', 'DRI', 'HBI', 'GPS', 'SBUX', 'TSCO', 'WYN', 'MGM', 'MAT', 'ROST', 'IPG', 'PVH', 'VFC', 'EXPE', 'JWN', 'GPC', 'DIS', 'FL', 'AAP', 'KSS', 'TIF', 'HAS', 'DHI', 'MHK', 'UAA', 'KMX', 'BBY', 'CMCSA', 'LEG', 'VIAB', 'CCL', 'LB', 'HOG', 'F', 'AZO', 'RL', 'DISCA', 'FOXA', 'PHM', 'AMZN', 'WHR', 'NKE', 'SNA', 'M', 'FOX', 'ULTA', 'GT', 'CMG', 'LOW', 'TWX', 'HD', 'CBS']


dataset_df = pd.read_csv(dataset_path/file_name)
dataset_df = dataset_df.set_index('Date')
all_set = list(dataset_df.columns.values[1:])  # all data
test_set = [p for p in all_set if p not in train_set]  # all data - train data
logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

# train set setting
if items_setting == "all":
    items_set = all_set
    output_set_name = "_all"
elif items_setting == "train":
    items_set = train_set
    output_set_name = "_train"
train_info = {"tw50": {"items":items_set, "output_file_name_basis": "tw50_20082017"},
              "sp500_19972007": {"items":items_set, "output_file_name_basis": f"sp500_19972007"},
              "sp500_20082017": {"items": items_set, "output_file_name_basis": f"sp500_20082017"},
              "tetuan_power": {"items": items_set, "output_file_name_basis":  f"tetuan_power"},
              "sp500_20082017_consumer_discretionary": {"items": items_set, "output_file_name_basis":  f"sp500_20082017_consumer_discretionary"}}
items_implement = train_info[data_implement]['items']
logging.info(f"===== len(train set): {len(items_implement)} =====")

# setting of name of output files and pictures title
output_file_name = train_info[data_implement]['output_file_name_basis'] + output_set_name
logging.info(f"===== file_name basis:{output_file_name} =====")

# display(dataset_df)


# ## Load or Create Correlation Data

# In[4]:


corr_window = 100
corr_stride = 100
data_length = int(len(dataset_df)/corr_window)*corr_window
corr_ind = list(range(99, 2400, corr_stride))  + list(range(99+20, 2500, corr_stride)) + \
           list(range(99+40, 2500, corr_stride)) + list(range(99+60, 2500, corr_stride)) + \
           list(range(99+80, 2500, corr_stride))  # only suit for settings of paper

corr_series_length = int((data_length-corr_window)/corr_stride)
corr_series_length_paper = 21  # only suit for settings of paper
data_diverse_stride = 20  # only suit for settings of paper


# In[5]:


def gen_data_corr(items: list, corr_ind: list) -> "pd.DataFrame":
    tmp_corr = dataset_df[items[0]].rolling(window=corr_window).corr(dataset_df[items[1]])
    tmp_corr = tmp_corr.iloc[corr_ind]
    data_df = pd.DataFrame(tmp_corr.values.reshape(-1, corr_series_length), columns=tmp_corr.index[:corr_series_length], dtype="float32")
    ind = [f"{items[0]} & {items[1]}_{i}" for i in range(0, 100, data_diverse_stride)]
    data_df.index = ind
    return data_df


def gen_train_data(items: list, corr_ind: list, save_file: bool = False)-> "four pd.DataFrame":
    train_df = pd.DataFrame(dtype="float32")
    dev_df = pd.DataFrame(dtype="float32")
    test1_df = pd.DataFrame(dtype="float32")
    test2_df = pd.DataFrame(dtype="float32")

    for pair in tqdm(combinations(items, 2)):
        data_df = gen_data_corr([pair[0], pair[1]], corr_ind=corr_ind)
        train_df = pd.concat([train_df, data_df.iloc[:, 0:21]])
        dev_df = pd.concat([dev_df, data_df.iloc[:, 1:22]])
        test1_df = pd.concat([test1_df, data_df.iloc[:, 2:23]])
        test2_df = pd.concat([test2_df, data_df.iloc[:, 3:24]])

    if save_file:
        before_arima_data_path = dataset_path/f"{output_file_name}_before_arima"
        before_arima_data_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(before_arima_data_path/f"{output_file_name}_train.csv")
        dev_df.to_csv(before_arima_data_path/f"{output_file_name}_dev.csv")
        test1_df.to_csv(before_arima_data_path/f"{output_file_name}_test1.csv")
        test2_df.to_csv(before_arima_data_path/f"{output_file_name}_test2.csv")

    return train_df, dev_df, test1_df, test2_df


before_arima_data_path = dataset_path/f"{output_file_name}_before_arima"
train_df_file = before_arima_data_path/f"{output_file_name}_train.csv"
dev_df_file = before_arima_data_path/f"{output_file_name}_dev.csv"
test1_df_file = before_arima_data_path/f"{output_file_name}_test1.csv"
test2_df_file = before_arima_data_path/f"{output_file_name}_test2.csv"
all_df_file = [train_df_file, dev_df_file, test1_df_file, test2_df_file]
if any([df_file.exists() for df_file in all_df_file]):
    corr_datasets = [pd.read_csv(df_file).set_index("Unnamed: 0") for df_file in all_df_file]
else:
    corr_datasets = gen_train_data(items_implement, corr_ind, save_file = save_raw_corr_data)


# # LSTM

# ## settings of input data of LSTM

# In[6]:


# Train - Dev - Test Generation
train_X = corr_datasets[0].iloc[:, :-1]
dev_X = corr_datasets[1].iloc[:, :-1]
test1_X = corr_datasets[2].iloc[:, :-1]
test2_X = corr_datasets[3].iloc[:, :-1]
train_Y = corr_datasets[0].iloc[:, -1]
dev_Y = corr_datasets[1].iloc[:, -1]
test1_Y = corr_datasets[2].iloc[:, -1]
test2_Y = corr_datasets[3].iloc[:, -1]


# data sampling
STEP = 20

lstm_train_X = train_X.values.reshape(-1, 20, 1)
lstm_train_Y = train_Y.values.reshape(-1, 1)
lstm_dev_X = dev_X.values.reshape(-1, 20, 1)
lstm_dev_Y = dev_Y.values.reshape(-1, 1)
lstm_test1_X = test1_X.values.reshape(-1, 20, 1)
lstm_test1_Y = test1_Y.values.reshape(-1, 1)
lstm_test2_X = test2_X.values.reshape(-1, 20, 1)
lstm_test2_Y = test2_Y.values.reshape(-1, 1)



# _train_X = np.asarray(train_X).reshape((int(1117500/STEP), 20, 1))
# _dev_X = np.asarray(dev_X).reshape((int(1117500/STEP), 20, 1))
# _test1_X = np.asarray(test1_X).reshape((int(1117500/STEP), 20, 1))
# _test2_X = np.asarray(test2_X).reshape((int(1117500/STEP), 20, 1))

# _train_Y = np.asarray(train_Y).reshape(int(1117500/STEP), 1)
# _dev_Y = np.asarray(dev_Y).reshape(int(1117500/STEP), 1)
# _test1_Y = np.asarray(test1_Y).reshape(int(1117500/STEP), 1)
# _test2_Y = np.asarray(test2_Y).reshape(int(1117500/STEP), 1)


# ## settings of LSTM

# In[7]:


lstm_layer = LSTM(units=10, kernel_regularizer=l1_l2(0.2, 0.0), bias_regularizer=l1_l2(0.2, 0.0), activation="tanh", dropout=0.1)  # LSTM hyper params from 【Something Old, Something New — A Hybrid Approach with ARIMA and LSTM to Increase Portfolio Stability】


# In[8]:


def double_tanh(x):
    return (tf.math.tanh(x) *2)


def build_many_one_lstm():
    inputs = Input(shape=(20, 1))
    lstm_1 = lstm_layer(inputs)
    outputs = Dense(units=1, activation=double_tanh)(lstm_1)
    return keras.Model(inputs, outputs, name="many_one_lstm")


opt = keras.optimizers.Adam(learning_rate=0.0001)
lstm_model = build_many_one_lstm()
lstm_model.summary()
lstm_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])


# In[ ]:


model_dir = Path('./models/')
log_dir = Path('./models/lstm_train_logs/')
res_dir = Path('./results/')
model_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)
res_csv_path = res_dir/f'{output_file_name}_LSTM_only_evaluation.csv'
res_csv_path.touch(exist_ok=True)
with open(res_csv_path, 'r+') as f:
    if not f.read():
        f.write("epoch,TRAIN_MSE,DEV_MSE,TEST1_MSE,TEST2_MSE,TRAIN_MAE,DEV_MAE,TEST1_MAE,TEST2_MAE")

res_df = pd.read_csv(res_csv_path)
saved_model_list = [int(p.stem.split('_')[1]) for p in model_dir.glob('*.h5')]
model_cbk = TensorBoard(log_dir=log_dir)
epoch_start = max(saved_model_list) if saved_model_list else 1
max_epoch = 5000
batch_size = 64

for epoch_num in tqdm(range(epoch_start, max_epoch)):
    if epoch_num > 1:
        lstm_model = load_model(model_dir/f"{output_file_name}_LSTM_only_epoch_{epoch_num - 1}.h5", custom_objects={'double_tanh':double_tanh})

    save_model = ModelCheckpoint(model_dir/f"{output_file_name}_LSTM_only_epoch_{epoch_num}.h5",
                                                 monitor='loss', verbose=1, mode='min', save_best_only=False)
    lstm_model.fit(lstm_train_X, lstm_train_Y, epochs=1, batch_size=batch_size, shuffle=True, callbacks=[model_cbk, save_model])

    # test the model
    score_train = lstm_model.evaluate(lstm_train_X, lstm_train_Y)
    score_dev = lstm_model.evaluate(lstm_dev_X, lstm_dev_Y)
    score_test1 = lstm_model.evaluate(lstm_test1_X, lstm_test1_Y)
    score_test2 = lstm_model.evaluate(lstm_test2_X, lstm_test2_Y)
    res_each_epoch_df = pd.DataFrame(np.array([epoch_num, score_train[0], score_dev[0], 
                                               score_test1[0], score_test2[0], 
                                               score_train[1], score_dev[1], 
                                               score_test1[1], score_test2[1]]).reshape(-1, 9),
                                    columns=["epoch", "TRAIN_MSE", "DEV_MSE", "TEST1_MSE", 
                                             "TEST2_MSE", "TRAIN_MAE", "DEV_MAE",
                                             "TEST1_MAE","TEST2_MAE"])
    res_df = pd.concat([res_df, res_each_epoch_df])

res_df.to_csv(res_csv_path, index=False)


# In[ ]:


# def double_tanh(x):
#     return (K.tanh(x) * 2)

# #get_custom_objects().update({'double_tanh':Activation(double_tanh)})

# # Model Generation
# model = Sequential()
# #check https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
# model.add(LSTM(25, input_shape=(20,1), dropout=0.0, kernel_regularizer=l1_l2(0.00,0.00), bias_regularizer=l1_l2(0.00,0.00)))
# model.add(Dense(1, activation=double_tanh))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
# #, kernel_regularizer=l1_l2(0,0.1), bias_regularizer=l1_l2(0,0.1),

# model.summary()


# In[ ]:


# # Fitting the Model
# model_scores = {}
# Reg = False
# d = 'LSTM_only_new_api'

# if Reg :
#     d += '_with_reg'

# epoch_num=1
# max_epoch = 3500
# for _ in range(max_epoch):

#     # train the model
#     dir_ = './lstm_only_models/'+d
#     file_list = os.listdir(dir_)
#     if len(file_list) != 0 :
#         epoch_num = len(file_list) + 1
#         recent_model_name = 'epoch'+str(epoch_num-1)
#         filepath = './lstm_only_models/' + d + '/' + recent_model_name
#         # custom_objects = {"double_tanh": double_tanh}
#         # with keras.utils.custom_object_scope(custom_objects):
#         model = load_model(filepath)

#     filepath = './lstm_only_models/' + d + '/epoch'+str(epoch_num)

#     # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
#     model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True)
#     model.save(filepath)
    
#     #callbacks_list = [checkpoint]
#     #if len(callbacks_list) == 0:
#     #    model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True)
#     #else:
#     #    model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True, callbacks=callbacks_list)

#     # test the model
#     score_train = model.evaluate(_train_X, _train_Y)
#     score_dev = model.evaluate(_dev_X, _dev_Y)
#     score_test1 = model.evaluate(_test1_X, _test1_Y)
#     score_test2 = model.evaluate(_test2_X, _test2_Y)

#     print('train set score : mse - ' + str(score_train[1]) +' / mae - ' + str(score_train[2]))
#     print('dev set score : mse - ' + str(score_dev[1]) +' / mae - ' + str(score_dev[2]))
#     print('test1 set score : mse - ' + str(score_test1[1]) +' / mae - ' + str(score_test1[2]))
#     print('test2 set score : mse - ' + str(score_test2[1]) +' / mae - ' + str(score_test2[2]))
# #.history['mean_squared_error'][0]
#     # get former score data
#     df = pd.read_csv("./lstm_only_scores/"+d+".csv")
#     train_mse = list(df['TRAIN_MSE'])
#     dev_mse = list(df['DEV_MSE'])
#     test1_mse = list(df['TEST1_MSE'])
#     test2_mse = list(df['TEST2_MSE'])

#     train_mae = list(df['TRAIN_MAE'])
#     dev_mae = list(df['DEV_MAE'])
#     test1_mae = list(df['TEST1_MAE'])
#     test2_mae = list(df['TEST2_MAE'])

#     # append new data
#     train_mse.append(score_train[1])
#     dev_mse.append(score_dev[1])
#     test1_mse.append(score_test1[1])
#     test2_mse.append(score_test2[1])

#     train_mae.append(score_train[2])
#     dev_mae.append(score_dev[2])
#     test1_mae.append(score_test1[2])
#     test2_mae.append(score_test2[2])

#     # organize newly created score dataset
#     model_scores['TRAIN_MSE'] = train_mse
#     model_scores['DEV_MSE'] = dev_mse
#     model_scores['TEST1_MSE'] = test1_mse
#     model_scores['TEST2_MSE'] = test2_mse

#     model_scores['TRAIN_MAE'] = train_mae
#     model_scores['DEV_MAE'] = dev_mae
#     model_scores['TEST1_MAE'] = test1_mae
#     model_scores['TEST2_MAE'] = test2_mae
    
#     # save newly created score dataset
#     model_scores_df = pd.DataFrame(model_scores)
#     model_scores_df.to_csv("./lstm_only_scores/"+d+".csv")

