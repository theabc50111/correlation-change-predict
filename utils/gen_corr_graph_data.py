import argparse
import logging
import math
import re
import sys
import warnings
from itertools import combinations
from pathlib import Path
from pprint import pformat
from typing import Dict, List

import dynamic_yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append("/workspace/correlation-change-predict/utils")
from utils import find_abs_max_cross_corr

current_dir = Path(__file__).parent
data_config_path = current_dir/"../config/data_config.yaml"
with open(data_config_path) as f:
    data = dynamic_yaml.load(f)
    DATA_CFG = yaml.full_load(dynamic_yaml.dump(data))

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)
warnings.simplefilter("ignore")


def _process_index(items):
    item_1 = items[0].strip(" ")
    item_2 = items[1].strip(" ")
    item_2 = item_2[:-2]

    return (item_1, item_2)


def calculate_rolling_t_depend_pearson_corr(df: pd.DataFrame, x_column: str, y_column: str, window_size: int,  debug_plot: bool = False) -> (pd.Series, pd.Series):
    """
    1. Retrieve the values of df[x_column] and df[y_column] within rolling window.
    2. calculate time-dependent-pearson-correlation of df[x_column] and df[y_column] within rolling window.

    Args:
      df: signal dataframe.
      x_column: one of df's column, can regard as signla name of `seg_x`.
      y_column: one of df's column, can regard as signla name of `seg_y`.
      window_size: size of window of rolling window

    Returns:
      t_depend_pearson_corr_ser: Series of time-dependent-pearson-correlation in each rolling window.
      t_depend_pearson_corr_lag_ser: Series of best lag of time-dependent-pearson-correlation in each rolling window.
    """
    shortest_segment_ts_len = int(window_size/3)
    max_lag = window_size - shortest_segment_ts_len
    x_rolling = df[x_column].rolling(window_size)
    y_rolling = df[y_column].rolling(window_size)
    t_depend_pearson_corr_ser = pd.Series(np.nan, index=df.index)
    t_depend_pearson_corr_lag_ser = pd.Series(np.nan, index=df.index)
    x_idx_list = [[None]*max_lag+[None]+list(range(1, max_lag+1)), list(range(shortest_segment_ts_len, window_size))+[window_size]+[None]*max_lag]
    y_idx_list = [list(range(max_lag, 0, -1))+[0]+[None]*max_lag, [None]*max_lag+[None]+list(range(window_size-1, shortest_segment_ts_len-1, -1))]
    k_list = np.arange(((window_size*2-2+1)-(2*(shortest_segment_ts_len-1))))
    lag_list = k_list-np.median(k_list)
    for i, (seg_x, seg_y) in enumerate(zip(x_rolling, y_rolling)):
        if i >= window_size-1:
            max_corr_coef = 0
            for lag, ((x_start_idx, x_end_idx), (y_start_idx, y_end_idx)) in zip(lag_list, zip(zip(*x_idx_list), zip(*y_idx_list))):
                corr_coef = np.corrcoef(seg_x[x_start_idx:x_end_idx], seg_y[y_start_idx:y_end_idx])[0, 1]
                if max_corr_coef <= np.absolute(corr_coef):
                    max_corr_coef = np.absolute(corr_coef)
                    t_depend_pearson_corr_ser.iloc[i] = corr_coef
                    t_depend_pearson_corr_lag_ser.iloc[i] = lag
                    x_plot = seg_x[x_start_idx:x_end_idx].values
                    y_plot = seg_y[y_start_idx:y_end_idx].values
                    seg_x_index_plot = seg_x[x_start_idx:x_end_idx].index[[0, -1]].values
                    seg_y_index_plot = seg_y[y_start_idx:y_end_idx].index[[0, -1]].values
                    lag_plot = lag
            if debug_plot:
                plt.plot(x_plot)
                plt.plot(y_plot)
                plt.title(f"X_dates:{seg_x_index_plot}\nY_dates:{seg_y_index_plot}\nTime shift: {lag_plot}")
                plt.show()
                plt.close()

    return t_depend_pearson_corr_ser, t_depend_pearson_corr_lag_ser


def gen_item_pair_corr(items: list, dataset_df: pd.DataFrame, corr_ser_len_max: int, corr_ind: list, data_gen_cfg: dict) -> pd.DataFrame:
    """
    Generate correlation of a pair of items which select from `dataset_df`
    """
    if args.corr_type == "pearson":
        tmp_corr = dataset_df[items[0]].rolling(window=data_gen_cfg["CORR_WINDOW"]).corr(dataset_df[items[1]])
    elif args.corr_type == "cross_corr":
        tmp_corr, lag_ser = calculate_rolling_t_depend_pearson_corr(dataset_df, items[0], items[1], data_gen_cfg["CORR_WINDOW"])
        logger.debug(f"Using cross_correlation mode, lag_ser[:40]:\n{lag_ser[:40]}")
    tmp_corr = tmp_corr.iloc[corr_ind]
    corr_ser_len_max = corr_ser_len_max - max(0, math.ceil((data_gen_cfg["CORR_WINDOW"]-data_gen_cfg["CORR_STRIDE"])/data_gen_cfg["CORR_STRIDE"]))  # CORR_WINDOW > CORR_STRIDE => slide window
    item_pair_corr = pd.DataFrame(tmp_corr.values.reshape(-1, corr_ser_len_max), columns=tmp_corr.index[:corr_ser_len_max], dtype="float32")
    ind = [f"{items[0]} & {items[1]}_{i}" for i in range(0,  data_gen_cfg["MAX_DATA_DIV_START_ADD"]+1, data_gen_cfg["DATA_DIV_STRIDE"])]
    item_pair_corr.index = ind
    item_pair_corr.index.name = "items"
    return item_pair_corr


def gen_corr_train_data(items: list, raw_data_df: pd.DataFrame,
                        corr_df_paths: Dict[str, Path],
                        data_gen_cfg: dict,
                        save_file: bool = False) -> List[pd.DataFrame]:
    """
    Generate correlation of item-pairs which are subset of `raw_data_df`
    This function moving start date by stride-20 to diversified data, so it return four dataframe.
    """
    # DEFAULT SETTING: data_gen_cfg["DATA_DIV_STRIDE"] == 20, data_gen_cfg["CORR_WINDOW"]==100, data_gen_cfg["CORR_STRIDE"]==100
    data_len = raw_data_df.shape[0]  # only suit for dateset.shape == [datetime, features], e.g sp500_hold_20082017 dataset
    corr_ser_len_max = int((data_len-data_gen_cfg["CORR_WINDOW"])/data_gen_cfg["CORR_STRIDE"])
    data_end_init = corr_ser_len_max * data_gen_cfg["CORR_STRIDE"]
    corr_ind_list = []
    for i in range(0, data_gen_cfg["MAX_DATA_DIV_START_ADD"]+1, data_gen_cfg["DATA_DIV_STRIDE"]):
        corr_ind_list.extend(list(range(data_gen_cfg["CORR_WINDOW"]-1+i, data_end_init+bool(i)*data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_STRIDE"])))  # only suit for settings of paper

    for key_df in corr_df_paths:
        locals()[key_df] = pd.DataFrame(dtype="float32")

    for pair in tqdm(combinations(items, 2), desc="Generating training data"):
        data_df = gen_item_pair_corr([pair[0], pair[1]], raw_data_df,
                                     corr_ser_len_max=corr_ser_len_max, corr_ind=corr_ind_list,
                                     data_gen_cfg=data_gen_cfg)
        corr_ser_len = corr_ser_len_max-(len(corr_df_paths)-1)  # dataset needs to be split into len(corr_df_paths) parts which means corr_ser_len will be corr_ser_len_max-(len(corr_df_paths)-1)
        for i, key_df in enumerate(corr_df_paths):  # only suit for settings of Korea paper
            locals()[key_df] = pd.concat([locals()[key_df], data_df.iloc[:, i:corr_ser_len+i]])

    if save_file:
        for key_df in corr_df_paths:
            locals()[key_df].to_csv(corr_df_paths[key_df])

    ret_vals = []
    for key_df in corr_df_paths:
        ret_vals.append(locals()[key_df])
    return ret_vals


# Prepare data
def set_corr_data(data_implement, data_cfg: dict, data_gen_cfg: dict,
                  data_split_setting: str = "data_sp_test2", train_items_setting: str = "train_train",
                  save_corr_data: bool = False):
    """
    # Data implement & output setting & testset setting
          data_implement: data implement setting  # watch options by operate: print(data_cfg["DATASETS"].keys())
          data_cfg: dict of pre-processed-data info, which is from 「config/data_config.yaml」
          data_gen_cfg: dict data generation configuration
          data_split_setting: data split period setting, only suit for only settings of Korean paper
          train_items_setting: train set setting  # train_train|train_all
          save_corr_data: setting of output files
    """

    # data loading & implement setting
    dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
    dataset_df = dataset_df.set_index('Date')
    all_set = list(dataset_df.columns)  # all data
    train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
    test_set = data_cfg['DATASETS'][data_implement]['TEST_SET'] if data_cfg['DATASETS'][data_implement].get('TEST_SET') else [p for p in all_set if p not in train_set]  # all data - train data
    logger.info(f"\n===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

    # train items implement settings
    items_implement = train_set if train_items_setting == "train_train" else all_set
    target_df = dataset_df.loc[::, items_implement]
    logger.info(f"\n===== len(train set): {len(items_implement)} =====")

    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + "-" + train_items_setting
    logger.info(f"\n===== file_name basis:{output_file_name} =====")
    logger.info(f"\n===== overview dataset_df =====\n{dataset_df}")
    if (col_1 := 'ABT') not in dataset_df.columns or (col_2 := 'ADI') not in dataset_df.columns:
        col_1, col_2 = dataset_df.columns[0], dataset_df.columns[1],
    logger.debug(f"\n===== head of dataset_df[col_1, col_2] =====\n{dataset_df.loc[:'2008-01-30', [col_1, col_2]]}")
    logger.debug(f"\n===== corr of dataset_df[col_1, col_2]  ====="
                 f"\n--- arround 2008/01/02~2008/01/15 ---\n{dataset_df.loc['2008-01-02':'2008-01-15', [col_1, col_2]].corr()}"
                 f"\n--- arround 2008/01/03~2008/01/16 ---\n{dataset_df.loc['2008-01-03':'2008-01-16', [col_1, col_2]].corr()}"
                 f"\n--- arround 2008/01/04~2008/01/17 ---\n{dataset_df.loc['2008-01-04':'2008-01-17', [col_1, col_2]].corr()}"
                 f"\n--- arround 2008/01/07~2008/01/18 ---\n{dataset_df.loc['2008-01-07':'2008-01-18', [col_1, col_2]].corr()}"
                 f"\n--- arround 2008/01/08~2008/01/22 ---\n{dataset_df.loc['2008-01-08':'2008-01-22', [col_1, col_2]].corr()}"
                 f"\n--- arround 2008/01/09~2008/01/23 ---\n{dataset_df.loc['2008-01-09':'2008-01-23', [col_1, col_2]].corr()}")

    # input folder settings
    corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/f"{args.corr_type}"/"corr_data"
    corr_data_dir.mkdir(parents=True, exist_ok=True)

    # Load or Create Correlation Data
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    train_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_train.csv"
    dev_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_dev.csv"
    test1_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_test1.csv"
    test2_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_test2.csv"
    all_corr_df_paths = dict(zip(["train_df", "dev_df", "test1_df", "test2_df"],
                                 [train_df_path, dev_df_path, test1_df_path, test2_df_path]))
    if all([df_path.exists() for df_path in all_corr_df_paths.values()]):
        corr_datasets = [pd.read_csv(df_path, index_col=["items"]) for df_path in all_corr_df_paths.values()]
    else:
        corr_datasets = gen_corr_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths,
                                            save_file=save_corr_data,  data_gen_cfg= data_gen_cfg)
        #corr_datasets = gen_corr_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths, save_file=save_corr_data,
                                      #corr_ser_len_max=corr_ser_len_max, corr_ind=corr_ind_list, max_data_div_start_add=max_data_div_start_add)

    if data_split_setting == "data_sp_train":
        corr_dataset = corr_datasets[0]
    elif data_split_setting == "data_sp_valid":
        corr_dataset = corr_datasets[2]
    elif data_split_setting == "data_sp_test1":
        corr_dataset = corr_datasets[2]
    elif data_split_setting == "data_sp_test2":
        corr_dataset = corr_datasets[3]

    logger.info(f"\n===== overview corr_dataset =====\n{corr_dataset.head()}")
    return target_df, corr_dataset, output_file_name


def gen_corr_dist_mat(data_ser: pd.Series, raw_df: pd.DataFrame, out_mat_compo: str = "sim"):
    """
    out_mat_compo:
        - sim : output a matrix with similiarity data
        - dist : output a matrix with distance data
    """

    assert isinstance(data_ser.index, pd.core.indexes.base.Index) and re.match(".*\ \&\ .*_0", data_ser.index[0]), "Index of input series should be form of \"COM1 & COM2_0'\""

    data_ser = data_ser.copy()
    data_ser.index = data_ser.index.str.split('&').map(_process_index)
    tmp_df = data_ser.unstack()
    unstack_missing_items = [item for item in raw_df.columns if item not in tmp_df.columns or item not in tmp_df.index]   # 只靠.unstack()來轉換會漏掉兩組item，.unstack()後的df 不具有這兩組item的自己對自己的row｜col
    for item in unstack_missing_items:
        data_ser.loc[item, item] = 0

    non_symmetry_mat = data_ser.unstack(fill_value=0)
    non_symmetry_mat = non_symmetry_mat.reindex(sorted(non_symmetry_mat.columns), axis=1)  # reindex by column names
    non_symmetry_mat = non_symmetry_mat.reindex(sorted(non_symmetry_mat.index), axis=0)  # reindex by index names
    distance_mat = non_symmetry_mat.T + non_symmetry_mat
    assert not distance_mat.isnull().any().any(), "Distance matrix contains missing values."
    assert set(distance_mat.columns) == set(raw_df.columns) and distance_mat.shape == (len(raw_df.columns), len(raw_df.columns)), "Error happens during the computation of dissimilarity matrix."
    np.fill_diagonal(distance_mat.values, 1)  # check by: tmp_df = gen_item_pair_corr(["A", "A"], dataset_df, corr_ser_len_max=corr_ser_len_max, corr_ind=corr_ind, max_data_div_start_add=max_data_div_start_add); tmp_df.iloc[::, 3:].mean(axis=1)

    if out_mat_compo == "sim":
        pass
    elif out_mat_compo == "dist":
        distance_mat = 1 - distance_mat.abs()  # This might get wrong by using abs(), because that would mix up corr = -1 & corr = 1

    return distance_mat


def gen_corr_mat_thru_t(corr_dataset, target_df, data_gen_cfg: dict, save_dir: Path = None, graph_mat_compo: str = "sim", show_mat_info_inds: list = None):
    """
    - corr_dataset: input df consisting of each pair of correlation coefficients over time
    - target_df: input the dataset_df which only contains target-items
    - data_gen_cfg: dict of data generation config, used to write config on file name
    - save_dir: save directory of graph array
    - graph_mat_compo:
          - sim : output a matrix with similiarity dat
          - dist : output a matrix with distance data
    - show_mat_info_inds: input a list of matrix indices to display
    """

    tmp_graph_list = []
    for i in range(corr_dataset.shape[1]):
        corr_spatial = corr_dataset.iloc[::, i]
        corr_mat = gen_corr_dist_mat(corr_spatial, target_df, out_mat_compo=graph_mat_compo).to_numpy()
        tmp_graph_list.append(corr_mat)

    flat_graphs_arr = np.stack(tmp_graph_list, axis=0)  # concate correlation matrix across time
    if save_dir:
        s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
        np.save(save_dir/f"corr_s{s_l}_w{w_l}_adj_mat", flat_graphs_arr)

    show_mat_info_inds = show_mat_info_inds if show_mat_info_inds else []
    for i in show_mat_info_inds:
        corr_spatial = corr_dataset.iloc[::, i]
        display_corr_mat = gen_corr_dist_mat(corr_spatial, target_df, out_mat_compo=graph_mat_compo)
        logger.info(f"\ncorrelation graph of No.{i} time-step"
                    f"correlation graph.shape:{flat_graphs_arr[0].shape}"
                    f"number of correlation graph:{len(flat_graphs_arr)}"
                    f"\nMin of corr_mat:{display_corr_mat.min()}"
                    f"\n{display_corr_mat.head()}"
                    f"\n==================================================")


def gen_filtered_corr_mat_thru_t(src_dir: Path, data_gen_cfg: dict, filter_mode: str = None, quantile: float = 0, save_dir: Path = None):
    """
    Create filttered correlation matrix by given conditions.
    - data_gen_cfg: dict of data generation config, used to write config on file name
    """
    is_filter_centered_zero = False
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_mats = np.load(src_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    res_mats = corr_mats.copy()
    if filter_mode == "keep_positive":
        mask = corr_mats < 0
        res_mats[mask] = np.NaN
    elif filter_mode == "keep_abs":
        res_mats = np.absolute(res_mats)
    elif filter_mode == "keep_strong":
        assert quantile != 0, "Need to set quantile in keep_strong mode"
        is_filter_centered_zero = True

    if is_filter_centered_zero:
        mask_begin = np.quantile(res_mats[res_mats <= 0], 1-quantile)
        mask_end = np.quantile(res_mats[res_mats >= 0], quantile)
    else:
        mask_begin = 0
        mask_end = np.quantile(res_mats[~np.isnan(res_mats)], quantile)
    logger.info(f"==================== mask range: {mask_begin} ~ {mask_end} ======================")

    if quantile:
        num_ele_bf_quan_mask = res_mats[~np.isnan(res_mats)].size
        mask = np.logical_and(res_mats > mask_begin, res_mats < mask_end)
        res_mats[mask] = np.NaN
        logger.debug(f"Keeping percentage of filtered data: {res_mats[~np.isnan(res_mats)].size / num_ele_bf_quan_mask}")
        logger.debug(f"Keeping percentage of all data: {res_mats[~np.isnan(res_mats)].size / res_mats.size}")
        logger.debug(f"res_mats.shape:{res_mats.shape}, res_mats.size: {res_mats.size}, res_mats[~np.isnan(res_mats)].size: {res_mats[~np.isnan(res_mats)].size}")
        logger.debug(f"quantiles of res_mats:{[np.quantile(res_mats[~np.isnan(res_mats)], i/4) for i in range(5)]}")
    if save_dir:
        np.save(save_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy", res_mats)


def gen_quantile_discretize_corr_mat_thru_t(src_dir: Path, data_gen_cfg: dict, num_bins: int = 0, save_dir: Path = None):
    """
    Create discretize correlation matrix by given conditions, the discretize boundary is decide by quantile of all matrices.
    - data_gen_cfg: dict of data generation config, used to write config on file name
    """
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_mats = np.load(src_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    res_mats = corr_mats.copy()
    quan_bins = [np.quantile(res_mats, q) for q in np.linspace(0, 1, num_bins+1)]
    discretize_mats = np.digitize(res_mats, quan_bins, right=True)
    discretize_mats[discretize_mats == 0] = 1
    discretize_mats[discretize_mats > num_bins] = num_bins
    discretize_mats = discretize_mats.astype(np.float32)
    discretize_values = np.linspace(-1, 1, num_bins)
    for discretize_tag, discretize_value in zip(np.unique(discretize_mats), discretize_values):
        discretize_mats[discretize_mats == discretize_tag] = discretize_value

    logger.info(f"\nReturn discretize matrices.shape:{discretize_mats.shape}"
                f"\nThe boundary of discretize matrices: {quan_bins}"
                f"\nUnique values and correspond counts of discretize matrices:\n{np.unique(discretize_mats, return_counts=True)}")
    if save_dir:
        np.save(save_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy", discretize_mats)


def gen_custom_discretize_corr_mat_thru_t(src_dir: Path, data_gen_cfg: dict, bins: list, save_dir: Path = None):
    """
    Create discretize correlation matrix by given conditions, the discretize boundary is customized.
    - data_gen_cfg: dict of data generation config, used to write config on file name
    """
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_mats = np.load(src_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    res_mats = corr_mats.copy()
    num_bins = len(bins)-1
    discretize_mats = np.digitize(res_mats, bins, right=True)
    discretize_mats[discretize_mats == 0] = 1
    discretize_mats[discretize_mats > num_bins] = num_bins
    discretize_mats = discretize_mats.astype(np.float32)
    discretize_values = np.linspace(-1, 1, num_bins)
    for discretize_tag, discretize_value in zip(np.unique(discretize_mats), discretize_values):
        discretize_mats[discretize_mats == discretize_tag] = discretize_value

    logger.info(f"\nReturn discretize matrices.shape:{discretize_mats.shape}"
                f"\nThe customized boundary of discretize matrices:\n{bins}"
                f"\nUnique values and correspond counts of discretize matrices:\n{np.unique(discretize_mats, return_counts=True)}")
    if save_dir:
        np.save(save_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy", discretize_mats)


def gen_nodes_mat_thru_t(target_df, corr_dates: pd.Index, data_gen_cfg: dict, nodes_v_mode: str = "all_values", save_dir: Path = None):
    """
    Generate nodes' values matrix through time points
    - target_df: input the dataset_df which only contains target-items
    - data_gen_cfg: dict of data generation config, used to write config on file name
    - save_dir: save directory of graph array
    """
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_begin_idx = target_df.index.get_loc(corr_dates[0]) - w_l + 1
    corr_end_idx = target_df.index.get_loc(corr_dates[-1]) + 1
    logger.debug(f"\nbegin idx of target_df:{corr_begin_idx}, end idx of target_df:{corr_end_idx}"
                 f"\nselected dates of setted_corr_dataset:{corr_dates}")
    logger.debug(f"\nFirst 20 row of target_df:\n{target_df.head(20)}"
                 f"\nLast 20 row of target_df:\n{target_df.tail(20)}"
                 f"\nselected dates for target_df:\n{target_df.iloc[corr_begin_idx:corr_end_idx, ::]}")

    num_nodes = target_df.shape[1]  # number of nodes
    num_mats = (corr_end_idx - corr_begin_idx - w_l + s_l) // s_l  # calculate the number of output matrices
    computing_mats = np.empty((num_mats, w_l, num_nodes))  # create an array to store the output matrices
    # fill the output array by slicing the input array
    for i in range(num_mats):
        sp_begin_idx = corr_begin_idx + i*s_l
        sp_end_idx = sp_begin_idx + w_l
        computing_mats[i] = target_df.iloc[sp_begin_idx:sp_end_idx, :].values

    logger.debug(f"\nNodes matrices.shape:{computing_mats.shape}"
                 f"\nValues of 2 nodes of {w_l} day in first 3 window:"
                 f"\n{computing_mats[:3, ::, :2]}")

    assert nodes_v_mode in ["all_values", "mean", "std", "mean_std", "min", "max"], "The given mode of nodes values is not in the options"
    if nodes_v_mode == "all_values":
        ret_mats = computing_mats
    elif nodes_v_mode == "mean":
        ret_mats = np.mean(computing_mats, axis=1).reshape(num_mats, -1, num_nodes)
    elif nodes_v_mode == "std":
        ret_mats = np.std(computing_mats, axis=1).reshape(num_mats, -1, num_nodes)
    elif nodes_v_mode == "mean_std":
        ret_mats = np.stack([np.mean(computing_mats, axis=1), np.std(computing_mats, axis=1)], axis=1)
    elif nodes_v_mode == "min":
        ret_mats = np.min(computing_mats, axis=1).reshape(num_mats, -1, num_nodes)
    elif nodes_v_mode == "max":
        ret_mats = np.max(computing_mats, axis=1).reshape(num_mats, -1, num_nodes)

    logger.info(f"\nReturn nodes's matrices.shape:{ret_mats.shape}"
                f"\nall node-features of 7 nodes in first 3 window:\n{ret_mats[:3, ::, :7]}'")

    if save_dir:
        np.save(save_dir/f"{nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy", ret_mats)

    return ret_mats


if __name__ == "__main__":
    data_args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    data_args_parser.add_argument("--corr_type", type=str, nargs='?', default="pearson",
                                  choices=["pearson", "cross_corr"],
                                  help="input the type of correlation computing")
    data_args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                                  help="input the number of stride length of correlation computing")
    data_args_parser.add_argument("--corr_window", type=int, nargs='?', default=10,
                                  help="input the number of window length of correlation computing")
    data_args_parser.add_argument("--data_div_stride", type=int, nargs='?', default=20,
                                  help="input the number of stride length of diversifing data")    # 20 is ONLY SUIT for data generation setting in that Korea paper. Because each pair need to be diversified to 5 corr_series
    data_args_parser.add_argument("--max_data_div_start_add", type=int, nargs='?', default=0,
                                  help="input the number of diversifing setting")   # value:range(0,80,20);
                                                                                    # 0 is strongly recommnded;
                                                                                    # 80 is ONLY SUIT for  disjoint-method(CORR_WINDOW=CORR_STRIDE);
                                                                                    # 80 is data generation setting in Korea paper. Because each pair need to be diversified to 5 corr_series,
                                                                                    # if diversified to 3 corr_series, MAX_DATA_DIV_START_ADD should be 40.
    data_args_parser.add_argument("--data_implement", type=str, nargs='?', default="PW_CONST_DIM_70_BKPS_0_NOISE_STD_10",  # data implement setting
                                  help="input the name of implemented dataset, watch options by printing /config/data_config.yaml/[\"DATASETS\"].keys()")  # watch options by operate: print(data_cfg["DATASETS"].keys())
    data_args_parser.add_argument("--train_items_setting", type=str, nargs='?', default="train_train",  # train set setting
                                  help="input the setting of training items, options:\n    - 'train_train'\n    - 'train_all'")
    data_args_parser.add_argument("--data_split_setting", type=str, nargs='?', default="data_sp_test2",  # data split period setting, only suit for only settings of Korean paper
                                  help=f"input the the setting of which splitting data to be used\n"
                                       f"    - data_sp_train\n"
                                       f"    - data_sp_valid\n"
                                       f"    - data_sp_test1\n"
                                       f"    - data_sp_test2")
    data_args_parser.add_argument("--graph_mat_compo", type=str, nargs='?', default="sim",
                                  help=f"Decide composition of graph_adjacency_matrix\n"
                                       f"    - sim : output a matrix with similiarity dat\n"
                                       f"    - dist : output a matrix with distance data")
    data_args_parser.add_argument("--filt_gra_mode", type=str, nargs='?', default="keep_positive",
                                  help=f"Decide filtering mode of graph_adjacency_matrix\n"
                                       f"    - keep_positive : remove all negative correlation \n"
                                       f"    - keep_strong : remove weak correlation \n"
                                       f"    - keep_abs : transform all negative correlation to positive")
    data_args_parser.add_argument("--filt_gra_quan", type=float, nargs='?', default=0.5,
                                  help="Decide filtering quantile")
    data_args_parser.add_argument("--quan_discrete_bins", type=int, nargs='?', default=3,
                                  help="Decide the number of quantile discrete bins")
    data_args_parser.add_argument("--custom_discrete_bins", type=float, nargs='*', default=None,
                                  help="Decide the custom discrete bins")
    data_args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default="all_values",
                                  help=f"Decide mode of nodes' vaules of graph_nodes_matrix\n"
                                       f"    - all_values : use all values inside window as nodes' values\n"
                                       f"    - mean: use average of all values inside window as nodes' values\n"
                                       f"    - std: use std of all values inside window as nodes' values\n"
                                       f"    - mean_std: use both mean and std of all values inside window as nodes' values\n"
                                       f"    - min: use min of all values inside window as nodes' values\n"
                                       f"    - max: use max of all values inside window as nodes' values\n")
    data_args_parser.add_argument("--save_corr_data", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                  help="input --save_corr_data to save correlation data")
    data_args_parser.add_argument("--save_corr_graph_arr", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                  help="input --save_corr_graph_arr to save correlation graph data")
    args = data_args_parser.parse_args()
    logger.info(pformat(vars(args), indent=1, width=100, compact=True))

    # generate correlation matrix across time
    DATA_GEN_CFG = {}
    DATA_GEN_CFG['CORR_STRIDE'] = args.corr_stride
    DATA_GEN_CFG['CORR_WINDOW'] = args.corr_window
    DATA_GEN_CFG['DATA_DIV_STRIDE'] = args.data_div_stride
    DATA_GEN_CFG['MAX_DATA_DIV_START_ADD'] = args.max_data_div_start_add
    aimed_target_df, setted_corr_dataset, output_file_name = set_corr_data(data_implement=args.data_implement,
                                                                           data_cfg=DATA_CFG,
                                                                           data_gen_cfg=DATA_GEN_CFG,
                                                                           data_split_setting=args.data_split_setting,
                                                                           train_items_setting=args.train_items_setting,
                                                                           save_corr_data=args.save_corr_data)
    gra_adj_mat_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/f"{args.corr_type}"/"graph_adj_mat"
    filtered_gra_adj_mat_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/f"{args.corr_type}"/f"filtered_graph_adj_mat/{args.filt_gra_mode}-quan{str(args.filt_gra_quan).replace('.', '')}"
    quan_discretize_gra_adj_mat_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/f"{args.corr_type}"/f"quan_discretize_graph_adj_mat/bins{args.quan_discrete_bins}"
    custom_discretize_gra_adj_mat_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/f"{args.corr_type}"/f"custom_discretize_graph_adj_mat/bins_{'_'.join((str(f) for f in args.custom_discrete_bins)).replace('.', '')}"
    gra_node_mat_dir = Path(DATA_CFG["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/"graph_node_mat"
    gra_adj_mat_dir.mkdir(parents=True, exist_ok=True)
    filtered_gra_adj_mat_dir.mkdir(parents=True, exist_ok=True)
    gra_node_mat_dir.mkdir(parents=True, exist_ok=True)
    quan_discretize_gra_adj_mat_dir.mkdir(parents=True, exist_ok=True)
    custom_discretize_gra_adj_mat_dir.mkdir(parents=True, exist_ok=True)
    gen_corr_mat_thru_t(corr_dataset=setted_corr_dataset,
                        target_df=aimed_target_df,
                        data_gen_cfg=DATA_GEN_CFG,
                        graph_mat_compo=args.graph_mat_compo,
                        save_dir=gra_adj_mat_dir if args.save_corr_graph_arr else None,
                        show_mat_info_inds=[0, 1, 2, 12])
    gen_filtered_corr_mat_thru_t(src_dir=gra_adj_mat_dir,
                                 data_gen_cfg=DATA_GEN_CFG,
                                 filter_mode=args.filt_gra_mode,
                                 quantile=args.filt_gra_quan,
                                 save_dir=filtered_gra_adj_mat_dir if args.save_corr_graph_arr else None)
    gen_quantile_discretize_corr_mat_thru_t(src_dir=gra_adj_mat_dir,
                                            data_gen_cfg=DATA_GEN_CFG,
                                            num_bins=args.quan_discrete_bins,
                                            save_dir=quan_discretize_gra_adj_mat_dir if args.save_corr_graph_arr else None)
    gen_custom_discretize_corr_mat_thru_t(src_dir=gra_adj_mat_dir,
                                          data_gen_cfg=DATA_GEN_CFG,
                                          bins=args.custom_discrete_bins,
                                          save_dir=custom_discretize_gra_adj_mat_dir if args.save_corr_graph_arr else None)
    gen_nodes_mat_thru_t(target_df=aimed_target_df,
                         corr_dates=setted_corr_dataset.columns,
                         data_gen_cfg=DATA_GEN_CFG,
                         nodes_v_mode=args.graph_nodes_v_mode,
                         save_dir=gra_node_mat_dir if args.save_corr_graph_arr else None)
