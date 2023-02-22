from itertools import combinations
from tqdm import tqdm
from pathlib import Path
import math
import re
import logging

import numpy as np
import pandas as pd
import dynamic_yaml
import yaml

from stl_decompn import stl_decompn


data_gen_cfg = {"CORR_WINDOW": 10,
                "CORR_STRIDE": 10,
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


# Prepare data
def set_corr_data(data_implement, data_cfg: dict, data_split_setting: str = "-data_sp_test2",
                  train_items_setting: str = "-train_train",
                  save_corr_data: bool = False):
    """
    # Data implement & output setting & testset setting
          data_implement: data implement setting  # watch options by operate: print(data_cfg["DATASETS"].keys())
          data_cfg: dict of data info, which is from 「config/data_config.yaml」
          data_split_setting: data split period setting, only suit for only settings of Korean paper
          train_items_setting: train set setting  # -train_train|-train_all
          save_corr_data: setting of output files 
    """
    
    
    # data loading & implement setting
    dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
    dataset_df = dataset_df.set_index('Date')
    all_set = list(dataset_df.columns)  # all data
    train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
    test_set = data_cfg['DATASETS'][data_implement]['TEST_SET'] if data_cfg['DATASETS'][data_implement].get('TEST_SET') else [p for p in all_set if p not in train_set]  # all data - train data
    logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

    # train items implement settings
    items_implement = train_set if train_items_setting == "-train_train" else all_set
    target_df = dataset_df.loc[::,items_implement]
    logging.info(f"===== len(train set): {len(items_implement)} =====")

    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
    logging.info(f"===== file_name basis:{output_file_name} =====")
    logging.info(f"\n{dataset_df}")
    
    # input folder settings
    corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-corr_data"
    corr_data_dir.mkdir(parents=True, exist_ok=True)


    # Load or Create Correlation Data
    # DEFAULT SETTING: data_gen_cfg["DATA_DIV_STRIDE"] == 20, data_gen_cfg["CORR_WINDOW"]==100, data_gen_cfg["CORR_STRIDE"]==100
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
        corr_datasets = gen_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths, save_file=save_corr_data)

    if data_split_setting == "-data_sp_test2":
        corr_dataset = corr_datasets[3]
        logging.info(f"{corr_dataset.head()}")
    
    return target_df, corr_dataset, output_file_name


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


def gen_corr_mat_thru_t(corr_dataset, target_df, save_dir: Path = None, graph_mat_compo: str = "sim", show_mat_info_inds: list = []):
    """
    corr_dataset: input df consisting of each pair of correlation coefficients over time
    target_df: input the dataset_df which only contains target-items
    save_dir: save directory of graph array
    graph_mat_compo: 
        - sim : output a matrix with similiarity dat
        - dist : output a matrix with distance data
    show_mat_info_inds: input a list of matrix indices to display
    """
    # output folder settings
    
    
    tmp_graph_list = []
    for i in range(corr_dataset.shape[1]):
        corr_spatial = corr_dataset.iloc[::,i]
        corr_mat = gen_corr_dist_mat(corr_spatial, target_df, out_mat_compo=graph_mat_compo).to_numpy()
        tmp_graph_list.append(corr_mat)

    flat_graphs_arr = np.stack(tmp_graph_list, axis=0)  # concate correlation matrix across time
    if save_dir:
        s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
        np.save(save_dir/f"corr_s{s_l}_w{w_l}_graph", flat_graphs_arr)

    for i in show_mat_info_inds:
        corr_spatial = corr_dataset.iloc[::,i]
        display_corr_mat = gen_corr_dist_mat(corr_spatial, target_df, out_mat_compo=graph_mat_compo)
        logging.info(f"correlation graph of No.{i} time-step")
        logging.info(f"correlation graph.shape:{flat_graphs_arr[0].shape}")
        logging.info(f"number of correlation graph:{len(flat_graphs_arr)}")
        logging.info(f"\nMin of corr_mat:{display_corr_mat.min()}")
        logging.info(f"\n{display_corr_mat.shape}")
        logging.info(f"\n{display_corr_mat.head()}")
        logging.info("=" * 70)


def calc_corr_ser_property(corr_dataset: "pd.DataFrame", corr_property_df_path: "pathlib.PosixPath"):
    if corr_property_df_path.exists():
        corr_property_df = pd.read_csv(corr_property_df_path).set_index("items")
    else:
        corr_mean = corr_dataset.mean(axis=1)
        corr_std = corr_dataset.std(axis=1)
        corr_stl_series = corr_dataset.apply(stl_decompn, axis=1)
        corr_stl_array = [[stl_period, stl_resid, stl_trend_std, stl_trend_coef] for stl_period, stl_resid, stl_trend_std, stl_trend_coef in corr_stl_series.values]
        corr_property_df = pd.DataFrame(corr_stl_array, index=corr_dataset.index)
        corr_property_df = pd.concat([corr_property_df, corr_mean, corr_std], axis=1)
        corr_property_df.columns = ["corr_stl_period", "corr_stl_resid", "corr_stl_trend_std", "corr_stl_trend_coef", "corr_ser_mean", "corr_ser_std"]
        corr_property_df.index.name = "items"
        corr_property_df.to_csv(corr_property_df_path)
    return corr_property_df

