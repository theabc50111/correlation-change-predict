#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
from itertools import combinations, product
from pathlib import Path
import sys
import warnings
import logging
from pprint import pformat
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

sys.path.append("/tf/correlation-coef-predict/ywt_library")
import data_generation
from data_generation import data_gen_cfg
from ywt_arima import arima_model, arima_err_logger_init
from stl_decompn import stl_decompn
from corr_property import calc_corr_ser_property

with open('../../config/data_config.yaml') as f:
    data = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data))

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.ERROR)
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
# logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# print(logger_list)


# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501
logging.debug(pformat(data_cfg, indent=1, width=100, compact=True))


# # Prepare data

# ## Data implement & output setting & testset setting


test_datasets = ['SP500_20082017_CORR_SER_ABS_CORR_MAT_HRCHY_11_CLUSTER']
test_lstm_weights = ['SP500_20082017_CORR_SER_ABS_CORR_MAT_HRCHY_11_CLUSTER_KS_HYPER_LSTM', 'SP500_20082017_RAND_66_KS_HYPER_LSTM', 'SP500_20082017_TEST_5000_EPOCH_KS_HYPER_LSTM', 'SP500_20082017_KS_HYPER_LSTM']

for data_imp, lstm_weight_set in product(test_datasets, test_lstm_weights):
    # setting of output files
    save_corr_data = True
    save_arima_resid_data = True
    # data implement setting
    data_implement = data_imp  # watch options by operate: print(data_cfg["DATASETS"].keys())
    # test set setting
    test_items_setting = "-test_test"  # -test_test|-test_all
    # data split period setting, only suit for only settings of Korean paper
    data_split_setting = "-data_sp_test2"
    # lstm weight setting
    lstm_weight_setting = lstm_weight_set  # watch options by operate: print(data_cfg["LSTM_WEIGHT"].keys())


    # In[3]:


    # data loading & implement setting
    dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
    dataset_df = dataset_df.set_index('Date')
    all_set = list(dataset_df.columns)  # all data
    train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
    test_set = data_cfg['DATASETS'][data_implement]['TEST_SET'] if data_cfg['DATASETS'][data_implement].get('TEST_SET') else [p for p in all_set if p not in train_set]  # all data - train data
    logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

    # test items implement settings
    items_implement = test_set if test_items_setting == "-test_test" else all_set
    logging.info(f"===== len(test set): {len(items_implement)} =====")

    lstm_weight_filepath = data_cfg["ARIMA_LSTM_LSTM_WEIGHT"][lstm_weight_setting]["FILE_PATH"]
    lstm_weight_name = data_cfg["ARIMA_LSTM_LSTM_WEIGHT"][lstm_weight_setting]["LSTM_WEIGHT_NAME"]
    logging.info(f"===== LSTM weight:{lstm_weight_name} =====")

    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + test_items_setting
    fig_title = data_implement + test_items_setting + lstm_weight_name + data_split_setting
    logging.info(f"===== file_name basis:{output_file_name}, fig_title basis:{fig_title} =====")
    # display(dataset_df)
    # display(test_set)

    # output folder settings
    corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-corr_data"
    arima_result_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-arima_res"
    corr_property_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}{lstm_weight_name}-corr_property"
    res_dir = Path('./results/')

    corr_data_dir.mkdir(parents=True, exist_ok=True)
    arima_result_dir.mkdir(parents=True, exist_ok=True)
    corr_property_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)


    # ## Load or Create Correlation Data

    # In[4]:


    data_length = int(len(dataset_df)/data_gen_cfg["CORR_WINDOW"])*data_gen_cfg["CORR_WINDOW"]
    corr_ser_len_max = int((data_length-data_gen_cfg["CORR_WINDOW"])/data_gen_cfg["CORR_STRIDE"])
    max_data_div_start_add = 0  # In the Korea paper, each pair has 5 corr_series(due to diversifing train data).
                                # BUT we only need to take one, so take 0 as arg.
    corr_ind = []

    # DEFAULT SETTING: data_gen_cfg["DATA_DIV_STRIDE"] == 20, data_gen_cfg["CORR_WINDOW"]==100, data_gen_cfg["CORR_STRIDE"]==100
    data_end_init = corr_ser_len_max * data_gen_cfg["CORR_STRIDE"]
    for i in range(0, max_data_div_start_add+1, data_gen_cfg["DATA_DIV_STRIDE"]):
        corr_ind.extend(list(range(data_gen_cfg["CORR_WINDOW"]-1+i, data_end_init+bool(i)*data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_STRIDE"])))  # only suit for settings of paper

    train_df_path = corr_data_dir/f"{output_file_name}-corr_train.csv"
    dev_df_path = corr_data_dir/f"{output_file_name}-corr_dev.csv"
    test1_df_path = corr_data_dir/f"{output_file_name}-corr_test1.csv"
    test2_df_path = corr_data_dir/f"{output_file_name}-corr_test2.csv"
    all_corr_df_paths = dict(zip(["train_df", "dev_df", "test1_df", "test2_df"],
                                 [train_df_path, dev_df_path, test1_df_path, test2_df_path]))
    if all([df_path.exists() for df_path in all_corr_df_paths.values()]):
        corr_datasets = [pd.read_csv(df_path).set_index("items") for df_path in all_corr_df_paths.values()]
    else:
        corr_datasets = data_generation.gen_train_data(items_implement, raw_data_df=dataset_df, corr_ser_len_max=corr_ser_len_max, corr_df_paths=all_corr_df_paths, corr_ind=corr_ind, max_data_div_start_add=max_data_div_start_add, save_file=save_corr_data)


    # In[5]:


    if data_split_setting == "-data_sp_test2":
        corr_dataset = corr_datasets[3]
    logging.info(f"{corr_datasets[0].shape, corr_datasets[1].shape, corr_datasets[2].shape, corr_datasets[3].shape}")
    # display(corr_dataset.iloc[0,::])
    # display(corr_dataset.head())


    # # ARIMA model

    # In[ ]:


    arima_result_path_basis = arima_result_dir/f'{output_file_name}.csv'
    arima_result_types = ["-arima_output", "-arima_resid", "-arima_model_info"]
    arima_result_paths = []
    arima_err_logger_init(Path(os.path.abspath(''))/f"results")

    for arima_result_type in arima_result_types:
        arima_result_paths.append(arima_result_dir/f'{output_file_name}{arima_result_type}{data_split_setting}.csv')

    if all([df_path.exists() for df_path in arima_result_paths]):
        arima_output_df, arima_resid_df, arima_model_info_df = [pd.read_csv(arima_result_path, index_col="items") for arima_result_path in arima_result_paths]
    else:
        arima_output_df, arima_resid_df, arima_model_info_df = arima_model(corr_dataset, arima_result_path_basis=arima_result_path_basis, data_split_setting=data_split_setting, save_file=save_arima_resid_data)


    # # LSTM

    # In[7]:


    def double_tanh(x):
        return (tf.math.tanh(x) * 2)


    lstm_model = load_model(lstm_weight_filepath, custom_objects={'double_tanh':double_tanh})
    lstm_model.summary()


    # # Hybrid model

    # In[8]:


    lstm_input = arima_resid_df.iloc[::,:-1].values.reshape(-1, 20, 1)
    lstm_pred = lstm_model.predict(lstm_input)
    lstm_pred = pd.DataFrame(lstm_pred, index=arima_resid_df.index, columns=["lstm_pred"])


    # ## Results post-processing

    # In[9]:


    def res_df_postprocess(merge_dfs: list) -> "pd.DataFrame":

        tmp_df = pd.DataFrame(columns=["items"])
        for merge_df in merge_dfs:
            tmp_df = pd.merge(tmp_df, merge_df, on="items", how='outer')
        else:
            results_df = tmp_df.reset_index(drop=True)

        results_df["hybrid_model_pred"] =  results_df["arima_pred"] + results_df["lstm_pred"]
        results_df["error"] = results_df["ground_truth"] - results_df["hybrid_model_pred"]
        results_df["absolute_err"] = results_df["error"].abs()
        results_df['arima_pred_dir'] = np.sign(results_df['ground_truth'] * results_df['arima_pred'])
        results_df['arima_err'] = results_df['ground_truth'] - results_df['arima_pred']
        results_df["lstm_compensation_dir"] = np.sign(results_df['arima_err']) * np.sign(results_df['lstm_pred'])
        quantile_mask = np.logical_and(results_df['error'] < np.quantile(results_df['error'], 0.75), results_df['error'] > np.quantile(results_df['error'], 0.25)).tolist()
        results_df['high_pred_performance'] = quantile_mask
        results_df['items[0]'] = results_df.apply(lambda row:row['items'].split(" & ")[0], axis=1)
        results_df['items[1]'] = results_df.apply(lambda row:row['items'].split(" & ")[1][:-2], axis=1)
        results_df = results_df.set_index("items")

        return results_df


    # stl_decompn(corr_datasets[0].iloc[0,::], overview=True)
    corr_property_df_path = corr_property_dir/f"{output_file_name}{lstm_weight_name}{data_split_setting}-corr_series_property.csv"

    if corr_property_df_path.exists():
        corr_property_df = pd.read_csv(corr_property_df_path)
    else:
        corr_property_df = calc_corr_ser_property(corr_dataset=corr_dataset, corr_property_df_path=corr_property_df_path)

    ground_truth = corr_dataset.iloc[::, -1]
    ground_truth.name = "ground_truth"
    arima_pred = arima_output_df.iloc[::, -1]
    arima_pred.name = "arima_pred"
    merge_dfs = [arima_model_info_df, corr_property_df, arima_pred, lstm_pred, ground_truth]

    res_df = res_df_postprocess(merge_dfs)
    res_df.to_csv(res_dir/f"{output_file_name}{lstm_weight_name}{data_split_setting}-res.csv")


    # # Display results

    # In[10]:


    res_df = pd.read_csv(res_dir/f"{output_file_name}{lstm_weight_name}{data_split_setting}-res.csv", index_col=["items"])
    logging.info(f"Test {lstm_weight_setting} on {data_implement}")
    logging.info(f"""
                     mse :{(res_df['error']**2).mean()},
                     std of square_err :{(res_df['error']**2).std()},
                     rmse :{np.sqrt((res_df['error']**2).mean())},
                     mae : {res_df['absolute_err'].mean()},
                     std of abs_err: {res_df['absolute_err'].std()},
                     sklearn mse: {mean_squared_error(res_df['ground_truth'], res_df['hybrid_model_pred'])}
                  """)
    logging.info("-"*50)
    logging.info(f"""
                     mse of ARIMA :{(res_df['arima_err']**2).mean()},
                     std of square_err ARIMA :{(res_df['arima_err']**2).std()},
                     rmse of ARIMA :{np.sqrt((res_df['arima_err']**2).mean())},
                     sklearn mse of ARIMA: {mean_squared_error(res_df['ground_truth'], res_df['arima_pred'])}
                  """)

    # In[11]:


    def plot_exploration(target_df: pd.core.frame.DataFrame, title: str) -> None:
        fig, axes = plt.subplots(figsize=(20, 20), nrows=7, ncols=2, sharex=False, sharey=False, dpi=100)
        s0 = axes[0, 0]
        s0.set_title("ABS_err violin")
        sns.violinplot(y=target_df["absolute_err"], ax=s0)
        s1 = axes[0, 1]
        s1.set_title("Err violin")
        sns.violinplot(y=target_df["error"], ax=s1)
        s2 = axes[1, 0]
        s2.set_title("ABS_err hist")
        target_df['absolute_err'].hist(bins=[b/10 for b in range(11)], ax=s2)
        s3 = axes[1, 1]
        s3.set_title("Err hist")
        target_df['error'].hist(bins=[b/10 for b in range(-10, 11)], ax=s3)
        s4 = axes[2, 0]
        s4.set_title("LSTM_compensation_dir count")
        sns.countplot(x="lstm_compensation_dir", data=target_df, ax=s4)
        s5 = axes[2, 1]
        s5.set_title("LSTM_compensation_dir count groupby ARIMA_pred_dir")
        df_gb = target_df.groupby(['arima_pred_dir', 'lstm_compensation_dir']).size().unstack(level=1)
        df_gb.plot(kind='bar', ax=s5)
        s6 = axes[3, 0]
        s6.set_title("ARIMA_model prediction Err violin group by LSTM_compensation_dir")
        sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["arima_err"], ax=s6)
        s8 = axes[4, 0]
        s8.set_title("ARIMA_model prediction magnitude group by LSTM_compensation_dir")
        sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["arima_pred"], ax=s8)
        s9 = axes[4, 1]
        s9.set_title("LSTM compensation magnitude group by LSTM_compensation_dir")
        sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["lstm_pred"], ax=s9)
        s10 = axes[5, 0]
        s10.set_title("Correlation magnitude in last period group by LSTM_compensation_dir")
        sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["ground_truth"], ax=s10)
        s11 = axes[5, 1]
        s11.set_title("Hybrid Err violin group by LSTM_compensation_dir")
        sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["error"], ax=s11)
        s12 = axes[6,0]
        s12.set_title("LSTM_compensation_dir pie with wrong ARIMA_pred_dir")
        df_gb.loc[df_gb.index==-1, :].squeeze().plot(kind="pie", autopct='%1.1f%%', ax=s12)
        s13 = axes[6,1]
        s13.set_title("LSTM_compensation_dir pie with correct ARIMA_pred_dir")
        df_gb.loc[df_gb.index==1, :].squeeze().plot(kind="pie", autopct='%1.1f%%', ax=s13)

        fig.suptitle(f"{title}_basic_exploration")
        plt.tight_layout()
        plt.savefig(f"./results/hybrid_prediction_analysis_{title}.png")
        plt.show()
        plt.close()


    def plot_exploration_pred_perform(target_df: pd.core.frame.DataFrame, title: str) -> None:
        fig, axes = plt.subplots(figsize=(20, 20), nrows=6, ncols=2, sharex=False, sharey=False, dpi=100)
        s1 = axes[0, 0]
        s1.set_title("LSTM_compensation_dir count groupby prediction performance")
        df_gb = target_df.groupby(['high_pred_performance', 'lstm_compensation_dir']).size().unstack(level=1)
        df_gb.plot(kind='bar', ax=s1)
        s2 = axes[0, 1]
        s2.set_title("ARIMA_model prediction magnitude group by prediction performance")
        sns.violinplot(x=target_df["high_pred_performance"], y=target_df["arima_pred"], ax=s2)
        s3 = axes[1, 0]
        s3.set_title("LSTM compensation magnitude group by prediction performance")
        sns.violinplot(x=target_df["high_pred_performance"], y=target_df["lstm_pred"], ax=s3)
        s4 = axes[1, 1]
        s4.set_title("Correlation magnitude in last period group by prediction performance")
        sns.violinplot(x=target_df["high_pred_performance"], y=target_df["ground_truth"], ax=s4)
        s5 = axes[2, 0]
        s5.set_title("Correlation series mean groupby prediction performance")
        sns.violinplot(x=target_df['high_pred_performance'], y=target_df["corr_ser_mean"], ax=s5)
        s6 = axes[2, 1]
        s6.set_title("Correlation series std groupby prediction performance")
        sns.violinplot(x=target_df['high_pred_performance'], y=target_df["corr_ser_std"], ax=s6)
        s7 = axes[3, 0]
        s7.set_title("Correlation series stl_period group by prediction performance")
        sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_stl_period"], ax=s7)
        s8 = axes[3, 1]
        s8.set_title("Correlation series stl_residual group by prediction performance")
        sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_stl_resid"], ax=s8)
        s9 = axes[4, 0]
        s9.set_title("Correlation series stl_trend_std group by prediction performance")
        sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_stl_trend_std"], ax=s9)
        s10 = axes[4, 1]
        s10.set_title("Correlation series stl_trend_coef group by prediction performance")
        sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_stl_trend_coef"], ax=s10)
        s11 = axes[5, 0]
        s11.set_title("ARIMA_pred_dir count groupby prediction performance")
        df_gb = target_df.groupby(['high_pred_performance', 'arima_pred_dir']).size().unstack(level=1)
        df_gb.plot(kind='bar', ax=s11)

        fig.suptitle(F"{title}_groupby prediction")
        plt.tight_layout()
        plt.savefig(f"./results/hybrid_prediction_analysis_groupby_pred_perform_{title}.png")
        plt.show()
        plt.close()


    def plot_stock_freq(target_df: pd.core.frame.DataFrame, title: str) -> None:
        stocks_show_freq = target_df.loc[target_df['high_pred_performance'] == True, ['items[0]','items[1]']].stack().value_counts().to_dict()
        plt.figure(figsize=(80, 10), dpi=100)
        plt.bar(range(len(stocks_show_freq)), list(stocks_show_freq.values()))
        plt.xticks(range(len(stocks_show_freq)), list(stocks_show_freq.keys()), rotation=60)
        plt.title(F"{title}_items appearence frequence")
        plt.savefig(f"./results/items_appearence_frequence_{title}.png")
        plt.show()
        plt.close()


    # In[12]:


    plot_exploration(res_df, fig_title)
    plot_exploration_pred_perform(res_df, fig_title)
    plot_stock_freq(res_df, fig_title)
