#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
from pathlib import Path
import warnings
import sys
import logging
from pprint import pformat

import pandas as pd
import numpy as np
import matplotlib as mpl
import dynamic_yaml
import yaml

sys.path.append("/workspace/correlation-change-predict/ywt_library")
import data_module
from data_module import data_gen_cfg,  set_corr_data, gen_corr_mat_thru_t
with open('../config/data_config.yaml') as f:
    data = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data))

warnings.simplefilter("ignore")
logging.basicConfig(format='%(levelname)-8s [%(filename)s] \n%(message)s',
                    level=logging.INFO)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.ERROR)
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
# logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# print(logger_list)


if __name__ == "__main__":
    logging.debug(pformat(data_cfg, indent=1, width=100, compact=True))
    logging.info(pformat(data_gen_cfg, indent=1, width=100, compact=True))

    # Data implement & output setting & testset setting
    # data implement setting
    data_implement = "SP500_20082017_CORR_SER_REG_CORR_MAT_HRCHY_11_CLUSTER"  # watch options by operate: print(data_cfg["DATASETS"].keys())
    # data split period setting, only suit for only settings of Korean paper
    data_split_setting = "-data_sp_test2"
    # train set setting
    train_items_setting = "-train_train"  # -train_train|-train_all
    # Decide composition of graph_matrix
    #     - sim : output a matrix with similiarity dat
    #     - dist : output a matrix with distance data
    graph_mat_compo = "sim"
    # setting of output files
    save_corr_graph_arr = True
    save_corr_data = True
    
    
    # generate correlation matrix across time
    target_df, corr_dataset, output_file_name = set_corr_data(data_implement=data_implement,
                                                           data_cfg=data_cfg,
                                                           data_split_setting=data_split_setting,
                                                           train_items_setting=train_items_setting,
                                                           save_corr_data=save_corr_data)
    
    res_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-graph_data"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    gen_corr_mat_thru_t(corr_dataset,
                        target_df,
                        graph_mat_compo=graph_mat_compo,
                        save_dir=res_dir if save_corr_graph_arr else None,
                        show_mat_info_inds=[0,1,2,12])
