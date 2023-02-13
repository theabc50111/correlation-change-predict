from itertools import combinations
from tqdm import tqdm
import math
import re
import logging

import numpy as np
import pandas as pd


data_gen_cfg = {"CORR_WINDOW": 10,
                "CORR_STRIDE": 1,
                "DATA_DIV_STRIDE": 20,  # 20 is ONLY SUIT for data generation setting in that Korea paper. Because each pair need to be diversified to 5 corr_series
                "MAX_DATA_DIV_START_ADD": 0  # value:range(0,80,20);
                                             # 0 is strongly recommnded;
                                             # 80 is ONLY SUIT for  disjoint-method(CORR_WINDOW=CORR_STRIDE); 
                                             # 80 is data generation setting in Korea paper. Because each pair need to be diversified to 5 corr_series, if diversified to 3 corr_series, MAX_DATA_DIV_START_ADD should be 40.
                }

_data_len = 2519  # only suit for sp500_hold_20082017 dataset 
_corr_ser_len_max = int((_data_len-data_gen_cfg["CORR_WINDOW"])/data_gen_cfg["CORR_STRIDE"])
_data_end_init = _corr_ser_len_max * data_gen_cfg["CORR_STRIDE"]
_corr_ind_list = []
for i in range(0, data_gen_cfg["MAX_DATA_DIV_START_ADD"]+1, data_gen_cfg["DATA_DIV_STRIDE"]):
    _corr_ind_list.extend(list(range(data_gen_cfg["CORR_WINDOW"]-1+i, _data_end_init+bool(i)*data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_STRIDE"])))  # only suit for settings of paper
    

def _process_index(items):
    item_1 = items[0].strip(" ")
    item_2 = items[1].strip(" ")
    item_2 = item_2[:-2]

    return (item_1, item_2)


def gen_data_corr(items: list, dataset_df: "pd.DataFrame", corr_ser_len_max: int, corr_ind: list, max_data_div_start_add: int) -> "pd.DataFrame":
    tmp_corr = dataset_df[items[0]].rolling(window=data_gen_cfg["CORR_WINDOW"]).corr(dataset_df[items[1]])
    tmp_corr = tmp_corr.iloc[corr_ind]
    corr_ser_len_max = corr_ser_len_max - max(0, math.ceil((data_gen_cfg["CORR_WINDOW"]-data_gen_cfg["CORR_STRIDE"])/data_gen_cfg["CORR_STRIDE"]))  #  CORR_WINDOW > CORR_STRIDE => slide window
    data_df = pd.DataFrame(tmp_corr.values.reshape(-1, corr_ser_len_max), columns=tmp_corr.index[:corr_ser_len_max], dtype="float32")
    ind = [f"{items[0]} & {items[1]}_{i}" for i in range(0,  max_data_div_start_add+1, data_gen_cfg["DATA_DIV_STRIDE"])]
    data_df.index = ind
    data_df.index.name = "items"
    return data_df


def gen_train_data(items: list, raw_data_df: "pd.DataFrame",
                   corr_df_paths: "dict of pathlib.PosixPath",
                   corr_ser_len_max: int = _corr_ser_len_max,
                   corr_ind: list = _corr_ind_list,
                   max_data_div_start_add: int = data_gen_cfg["MAX_DATA_DIV_START_ADD"],
                   save_file: bool = False) -> "list of pd.DataFrame":
    for key_df in corr_df_paths:
        locals()[key_df] = pd.DataFrame(dtype="float32")

    for pair in tqdm(combinations(items, 2)):
        data_df = gen_data_corr([pair[0], pair[1]], raw_data_df,
                                corr_ser_len_max=corr_ser_len_max, corr_ind=corr_ind, max_data_div_start_add= max_data_div_start_add)
        
        corr_ser_len = corr_ser_len_max-(len(corr_df_paths)-1)  # dataset needs to be split into len(corr_df_paths) parts which means corr_ser_len will be corr_ser_len_max-(len(corr_df_paths)-1)
        for i, key_df in enumerate(corr_df_paths):  # only suit for settings of Korea paper
            locals()[key_df] = pd.concat([locals()[key_df], data_df.iloc[:, i:corr_ser_len+i]])

    if save_file:
        for key_df in corr_df_paths:
            locals()[key_df].to_csv(corr_df_paths[key_df])

    ret_vals= []
    for key_df in corr_df_paths:
        ret_vals.append(locals()[key_df])
    return ret_vals


def gen_corr_dist_mat(data_ser: "pd.Series", raw_df: "pd.DataFrame", out_mat_compo: str = "sim"):
    
    """
    out_mat_compo: 
    - sim : output a matrix with similiarity dat
    - dist : output a matrix with distance data
    """
    
    assert isinstance(data_ser.index, pd.core.indexes.base.Index) and re.match(f".*\ \&\ .*_0", data_ser.index[0]), "Index of input series should be form of \"COM1 & COM2_0'\""

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
    assert set(distance_mat.columns) == set(raw_df.columns) and distance_mat.shape == (len(raw_df.columns),len(raw_df.columns)), f"Error happens during the computation of dissimilarity matrix."
    np.fill_diagonal(distance_mat.values, 1)  # check by: tmp_df = gen_data_corr(["A", "A"], dataset_df, corr_ser_len_max=corr_ser_len_max, corr_ind=corr_ind, max_data_div_start_add=max_data_div_start_add); tmp_df.iloc[::, 3:].mean(axis=1)

    if out_mat_compo == "sim":
        return distance_mat
    elif out_mat_compo == "dist":
        distance_mat = 1 - distance_mat.abs()  # This might get wrong by using abs(), because that would mix up corr = -1 & corr = 1
        return distance_mat


def gen_corr_graph(corr_dataset, corr_dist_mat_df, save_dir, graph_mat_compo: str = "sim", save_file: bool = False, show_mat_i_info: int = 1):
    """
    corr_dist_mat_df : input the dataset_df which only contains target-items
    """
    tmp_graph_list = []
    for i in range(corr_dataset.shape[1]):
        corr_spatial = corr_dataset.iloc[::,i]
        corr_mat = gen_corr_dist_mat(corr_spatial, corr_dist_mat_df, out_mat_compo=graph_mat_compo).to_numpy()
        tmp_graph_list.append(corr_mat)

    flat_graphs_arr = np.stack(tmp_graph_list, axis=0)
    if save_file:
        np.save(save_dir/f"corr_calc_reg-corr_graph", flat_graphs_arr)

    if show_mat_i_info:
        corr_spatial = corr_dataset.iloc[::,show_mat_i_info]
        display_corr_mat = gen_corr_dist_mat(corr_spatial, corr_dist_mat_df, out_mat_compo=graph_mat_compo)
        logging.info(f"correlation graph.shape:{flat_graphs_arr[0].shape}")
        logging.info(f"number of correlation graph:{len(flat_graphs_arr)}")
        logging.info(f"\nMin of corr_mat:{display_corr_mat.min()}")
        logging.info(f"\n{display_corr_mat.shape}")
        logging.info(f"\n{display_corr_mat.head()}")