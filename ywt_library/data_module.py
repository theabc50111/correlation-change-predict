from itertools import combinations
from tqdm import tqdm
from pathlib import Path
import warnings
import sys
import math
import re
import logging
from pprint import pformat, pprint
import argparse

import numpy as np
import pandas as pd
import dynamic_yaml
import yaml

sys.path.append("/workspace/correlation-change-predict/ywt_library")
from stl_decompn import stl_decompn
current_dir = Path(__file__).parent
data_config_path = current_dir/"../config/data_config.yaml"
with open(data_config_path) as f:
    data = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data))

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


def gen_data_corr(items: list, dataset_df: "pd.DataFrame", corr_ser_len_max: int, corr_ind: list, data_gen_cfg: dict) -> "pd.DataFrame":
    tmp_corr = dataset_df[items[0]].rolling(window=data_gen_cfg["CORR_WINDOW"]).corr(dataset_df[items[1]])
    tmp_corr = tmp_corr.iloc[corr_ind]
    corr_ser_len_max = corr_ser_len_max - max(0, math.ceil((data_gen_cfg["CORR_WINDOW"]-data_gen_cfg["CORR_STRIDE"])/data_gen_cfg["CORR_STRIDE"]))  #  CORR_WINDOW > CORR_STRIDE => slide window
    data_df = pd.DataFrame(tmp_corr.values.reshape(-1, corr_ser_len_max), columns=tmp_corr.index[:corr_ser_len_max], dtype="float32")
    ind = [f"{items[0]} & {items[1]}_{i}" for i in range(0,  data_gen_cfg["MAX_DATA_DIV_START_ADD"]+1, data_gen_cfg["DATA_DIV_STRIDE"])]
    data_df.index = ind
    data_df.index.name = "items"
    return data_df


def gen_train_data(items: list, raw_data_df: "pd.DataFrame",
                   corr_df_paths: "dict of pathlib.PosixPath",
                    data_gen_cfg: dict,
                   save_file: bool = False) -> "list of pd.DataFrame":
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
        data_df = gen_data_corr([pair[0], pair[1]], raw_data_df,
                                corr_ser_len_max=corr_ser_len_max, corr_ind=corr_ind_list,
                                data_gen_cfg=data_gen_cfg)
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
    target_df = dataset_df.loc[::,items_implement]
    logger.info(f"\n===== len(train set): {len(items_implement)} =====")

    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + "-" + train_items_setting
    logger.info(f"\n===== file_name basis:{output_file_name} =====")
    logger.info(f"\n===== overview dataset_df =====\n{dataset_df}")

    # input folder settings
    corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/"corr_data"
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
        corr_datasets = gen_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths,
                                       save_file=save_corr_data,  data_gen_cfg= data_gen_cfg)
        #corr_datasets = gen_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths, save_file=save_corr_data,
                                      #corr_ser_len_max=corr_ser_len_max, corr_ind=corr_ind_list, max_data_div_start_add=max_data_div_start_add)

    if data_split_setting == "data_sp_test2":
        corr_dataset = corr_datasets[3]
        logger.info(f"\n===== overview corr_dataset =====\n{corr_dataset.head()}")
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
        logger.info(f"correlation graph of No.{i} time-step")
        logger.info(f"correlation graph.shape:{flat_graphs_arr[0].shape}")
        logger.info(f"number of correlation graph:{len(flat_graphs_arr)}")
        logger.info(f"\nMin of corr_mat:{display_corr_mat.min()}")
        logger.info(f"\n{display_corr_mat.shape}")
        logger.info(f"\n{display_corr_mat.head()}")
        logger.info("=" * 70)

def gen_filtered_corr_mat_thru_t(src_dir: Path, filter_mode: str = None, quantile: float = 0, save_dir: Path = None):
    """
    Create filttered correlation matrix by given conditions.
    """
    is_filter_centered_zero = False
    s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
    corr_mats = np.load(src_dir/f"corr_s{s_l}_w{w_l}_graph.npy")
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
        mask_begin = np.quantile(res_mats[res_mats<=0], 1-quantile)
        mask_end = np.quantile(res_mats[res_mats>=0], quantile)
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
        np.save(save_dir/f"corr_s{s_l}_w{w_l}_graph.npy", res_mats)

def calc_corr_ser_property(corr_dataset: pd.DataFrame, corr_property_df_path: Path):
    """
    Produce property of correlation series in form of dataframe
    """
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


if __name__ == "__main__":
    data_args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
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
    data_args_parser.add_argument("--data_implement", type=str, nargs='?', default="SP500_20082017_CORR_SER_REG_CORR_MAT_HRCHY_11_CLUSTER",  # data implement setting
                        help="input the name of implemented dataset, watch options by printing /config/data_config.yaml/[\"DATASETS\"].keys()")  # watch options by operate: print(data_cfg["DATASETS"].keys())
    data_args_parser.add_argument("--train_items_setting", type=str, nargs='?', default="train_train",  # train set setting
                        help="input the setting of training items, options:\n    - 'train_train'\n    - 'train_all'")
    data_args_parser.add_argument("--data_split_setting", type=str, nargs='?', default="data_sp_test2",  # data split period setting, only suit for only settings of Korean paper
                        help="input the the setting of which splitting data to be used")
    data_args_parser.add_argument("--graph_mat_compo", type=str, nargs='?', default="sim",
                        help="Decide composition of graph_matrix\n    - sim : output a matrix with similiarity dat\n    - dist : output a matrix with distance data")
    data_args_parser.add_argument("--filt_gra_mode", type=str, nargs='?', default="keep_positive",
                        help="Decide filtering mode of graph_matrix\n    - keep_positive : remove all negative correlation \n    - keep_strong : remove weak correlation \n    - keep_abs : transform all negative correlation to positive")
    data_args_parser.add_argument("--filt_gra_quan", type=float, nargs='?', default=0.5,
                        help="Decide filtering quantile")
    data_args_parser.add_argument("--save_corr_data", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                        help="input --save_corr_data to save correlation data")
    data_args_parser.add_argument("--save_corr_graph_arr", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                        help="input --save_corr_graph_arr to save correlation graph data")
    args = data_args_parser.parse_args()
    logger.debug(pformat(data_cfg, indent=1, width=100, compact=True))
    logger.info(pformat(vars(args), indent=1, width=100, compact=True))

    # generate correlation matrix across time
    data_gen_cfg = {}
    data_gen_cfg['CORR_STRIDE'] = args.corr_stride
    data_gen_cfg['CORR_WINDOW'] = args.corr_window
    data_gen_cfg['DATA_DIV_STRIDE'] = args.data_div_stride
    data_gen_cfg['MAX_DATA_DIV_START_ADD'] = args.max_data_div_start_add
    target_df, corr_dataset, output_file_name = set_corr_data(data_implement=args.data_implement,
                                                              data_cfg=data_cfg,
                                                              data_gen_cfg=data_gen_cfg,
                                                              data_split_setting=args.data_split_setting,
                                                              train_items_setting=args.train_items_setting,
                                                              save_corr_data=args.save_corr_data)
    gra_res_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/"graph_data"
    filtered_gra_res_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}"/f"filtered_graph_data/{args.filt_gra_mode}-quan{str(args.filt_gra_quan).replace('.', '')}"
    gra_res_dir.mkdir(parents=True, exist_ok=True)
    filtered_gra_res_dir.mkdir(parents=True, exist_ok=True)
    gen_corr_mat_thru_t(corr_dataset,
                        target_df,
                        graph_mat_compo=args.graph_mat_compo,
                        save_dir=gra_res_dir if args.save_corr_graph_arr else None,
                        show_mat_info_inds=[0,1,2,12])
    gen_filtered_corr_mat_thru_t(gra_res_dir, filter_mode=args.filt_gra_mode, quantile=args.filt_gra_quan, save_dir=filtered_gra_res_dir if args.save_corr_graph_arr else None)
