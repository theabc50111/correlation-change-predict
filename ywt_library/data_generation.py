from itertools import combinations
from tqdm import tqdm

import pandas as pd


data_gen_cfg = {"CORR_WINDOW": 100,
                "CORR_STRIDE": 100,
                "DATA_DIV_STRIDE": 20,  # 20 is ONLY SUIT for data generation method in that Korea paper. Because each pair need to be diversified to 5 corr_series
                "MAX_DATA_DIV_START_ADD": 80  # 80 is ONLY SUIT for data generation method in that Korea paper. Because each pair need to be diversified to 5 corr_series
                }

_data_len = 2500  # only suit for length of data is between 2500 to 2600 
_corr_ser_len_max = int((_data_len-data_gen_cfg["CORR_WINDOW"])/data_gen_cfg["CORR_STRIDE"])
_data_end_init = _corr_ser_len_max * data_gen_cfg["CORR_STRIDE"]
_corr_ind_list = []
for i in range(0, data_gen_cfg["MAX_DATA_DIV_START_ADD"]+1, data_gen_cfg["DATA_DIV_STRIDE"]):
    _corr_ind_list.extend(list(range(data_gen_cfg["CORR_WINDOW"]-1+i, _data_end_init+bool(i)*data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_STRIDE"])))  # only suit for settings of paper


def gen_data_corr(items: list, dataset_df: "pd.DataFrame", corr_ser_len_max: int, corr_ind: list, max_data_div_start_add: int) -> "pd.DataFrame":
    tmp_corr = dataset_df[items[0]].rolling(window=data_gen_cfg["CORR_WINDOW"]).corr(dataset_df[items[1]])
    tmp_corr = tmp_corr.iloc[corr_ind]
    data_df = pd.DataFrame(tmp_corr.values.reshape(-1, corr_ser_len_max), columns=tmp_corr.index[:corr_ser_len_max], dtype="float32")
    ind = [f"{items[0]} & {items[1]}_{i}" for i in range(0,  max_data_div_start_add+1, data_gen_cfg["DATA_DIV_STRIDE"])]
    data_df.index = ind
    data_df.index.name = "items"
    return data_df


def gen_train_data(items: list, raw_data_df: "pd.DataFrame",
                   corr_df_paths: "dict of pathlib.PosixPath",
                   corr_ser_len_max: int,
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
