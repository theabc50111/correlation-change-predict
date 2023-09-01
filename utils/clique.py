#!/usr/bin/env python
# coding: utf-8
import logging
import os
import sys
import traceback
import warnings
from itertools import combinations
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yaml
from scipy.stats import norm, uniform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import fbeta_score, make_scorer, silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tqdm import tqdm

sys.path.append("/workspace/correlation-change-predict/utils")
from cluster_utils import (calc_silhouette_label_freq_std,
                           hrchy_cluster_fixed_n_cluster,
                           hrchy_clustering_distance_threshold_rs,
                           hrchy_clustering_n_cluster_gs,
                           obs_hrchy_cluster_instances,
                           plot_cluster_labels_distribution, plot_dendrogram)
from gen_corr_graph_data import gen_corr_dist_mat
from utils import calc_corr_ser_property

current_dir = Path(os.getcwd())
data_config_path = current_dir / "../config/data_config.yaml"
with open(data_config_path) as f:
    data_cfg_yaml = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data_cfg_yaml))

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
# logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# print(logger_list)

# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501
logging.debug(pformat(data_cfg, indent=1, width=100, compact=True))


# # Prepare data
# ## Data implement & output setting & testset setting
# data implement setting
data_implement = "SP500_20112015"  # watch options by printing /config/data_config.yaml/["DATASETS"].keys()
# print(data_cfg["DATASETS"].keys())
# etl set setting
etl_items_setting = "-train_all"  # -train_train|-train_all
# data split period setting, only suit for only settings of Korean paper
data_split_setting = "data_sp_test2"
# set correlation type
corr_type = "pearson"  # "pearson" | "cross_corr"
# set CORR_WINDOW and CORR_STRIDE length
w_l=50 ; s_l = 1
# Decide how to calculate corr_ser
corr_ser_clac_method = "corr_ser_calc_regular"  # corr_ser_calc_regular|corr_ser_calc_abs
# Decide correlation series reduction method
corr_ser_reduction_method = "corr_ser_std" # "corr_ser_std" | "corr_ser_mean"
# Decide composition of correlation_matrix
corr_mat_compo = "sim"
# Decide how to filter distanc_mat
filtered_distance_mat_method = "positive_corr_ser_mean_filtered"  # no_filtered|large_corr_ser_mean_filtered|small_corr_ser_mean_filtered|positive_corr_ser_mean_filtered|negative_corr_ser_mean_filtered
# Decide whether save output or not
is_save_output = True

dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
dataset_df = dataset_df.set_index('Date')
all_set = list(dataset_df.columns)  # all data
train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
test_set = data_cfg['DATASETS'][data_implement]['TEST_SET'] if data_cfg['DATASETS'][data_implement].get('TEST_SET') else [p for p in all_set if p not in train_set]  # all data - train data
logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

# test items implement settings
items_implement = train_set if etl_items_setting == "-train_train" else all_set
logging.info(f"===== len(etl set): {len(items_implement)} =====")

# setting of name of output files and pictures title
output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + etl_items_setting
fig_title = f"{data_implement}{etl_items_setting}-{data_split_setting}"
logging.info(f"===== file_name basis:{output_file_name}, fig_title basis:{fig_title} =====")
# display(dataset_df)

# input folder settings
corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{corr_type}/corr_data"
assert corr_data_dir.exists(), f"\nOperate data_module.py to generate correlation data first, operate command:\n    python data_module.py --data_implement SP500_20082017 --train_items_setting -train_all --data_split_setting -data_sp_test2 --graph_mat_compo sim --save_corr_data"

# output folder settings
res_dir= Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{corr_type}/corr_property/corr_s{s_l}_w{w_l}/{corr_ser_clac_method}"
cluster_items_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{corr_type}/cluster/corr_s{s_l}_w{w_l}/{corr_ser_clac_method}/{corr_ser_reduction_method}/{corr_mat_compo}/{filtered_distance_mat_method}"
res_dir.mkdir(parents=True, exist_ok=True)
cluster_items_dir.mkdir(parents=True, exist_ok=True)

# ## Load or Create Correlation Data
train_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_train.csv"
dev_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_dev.csv"
test1_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_test1.csv"
test2_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_test2.csv"
if data_split_setting == "data_sp_train":
    df_path = train_df_path
elif data_split_setting == "data_sp_valid":
    df_path = dev_df_path
elif data_split_setting == "data_sp_test1":
    df_path = test1_df_path
elif data_split_setting == "data_sp_test2":
    df_path = test2_df_path
corr_dataset = pd.read_csv(df_path, index_col=["items"])
logging.info(f"===== corr_dataset.shape:{corr_dataset.shape} =====")
logging.info("===== corr_dataset.head() =====")
logging.info(corr_dataset.head())

# # Calculate properties of Corrlelation series
if corr_ser_clac_method == "corr_ser_calc_regular":
    corr_property_df_path = res_dir/f"{output_file_name}-{data_split_setting}-corr_series_property.csv"
    corr_property_df = calc_corr_ser_property(corr_dataset=corr_dataset, corr_property_df_path=corr_property_df_path)
elif corr_ser_clac_method == "corr_ser_calc_abs":
    # calculate corr_property_df with abs(corr_dataset)
    corr_property_df_path = res_dir/f"{output_file_name}-{data_split_setting}-corr_series_abs_property.csv"
    corr_property_df = calc_corr_ser_property(corr_dataset=corr_dataset.abs(), corr_property_df_path=corr_property_df_path)

corr_property_df_mean_reduct_series = corr_property_df.mean(axis=0)
corr_property_df_mean_reduct_series.name = "corr_property_df_mean_reduct_series"
corr_property_df_std_reduct_series = corr_property_df.std(axis=0)
corr_property_df_std_reduct_series.name = "corr_property_df_std_reduct_series"

logging.info(f"{fig_title} + corr_w{w_l} + corr_s{s_l} + corr_ser_clac_method:{corr_ser_clac_method}")
logging.info(f"Min of corr_ser_mean:{corr_property_df.loc[::,'corr_ser_mean'].min()}")
logging.info("===== corr_property_df.head(): =====")
logging.info(corr_property_df.head())

# # Clustring
# ## calculate distance matrix and
# ## output distance matrix
corr_ser_std = corr_property_df.loc[::, "corr_ser_std"]
corr_ser_mean = corr_property_df.loc[::, "corr_ser_mean"]
corr_reduction_ser = corr_ser_std if corr_ser_reduction_method == "corr_ser_std" else corr_ser_mean
selected_dataset_df = dataset_df.loc[::, items_implement]
distance_mat = gen_corr_dist_mat(corr_reduction_ser, selected_dataset_df, out_mat_compo=corr_mat_compo)
corr_ser_mean_q2 = corr_property_df['corr_ser_mean'].quantile(0.5)
filtered_distance_mat_settings = {"large_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) > corr_ser_mean_q2),
                                  "small_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) < corr_ser_mean_q2),
                                  "positive_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) > 0),
                                  "negative_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) < 0)}

if (filter_mask := filtered_distance_mat_settings.get(filtered_distance_mat_method)) is not None:
    distance_mat[filter_mask] = 0
    G = nx.from_pandas_adjacency(distance_mat)
    max_clique = []
    for i, clique in enumerate(nx.find_cliques(G)):
        logging.info(f"{i}th clique: {clique}")
        if len(clique) > len(max_clique):
            max_clique = clique
            with open(cluster_items_dir/"tmp_max_clique_nodes.txt", "w") as f:
                f.write(str(max_clique))
    distance_mat = distance_mat.loc[max_clique, max_clique]

if is_save_output:
    distance_mat.to_csv(cluster_items_dir/"distance_mat.csv")
    logging.info(f"distance_mat.csv has been save to {cluster_items_dir}")
    with open(cluster_items_dir/"max_clique_nodes.txt", "w") as f:
        f.write(str(max_clique))
