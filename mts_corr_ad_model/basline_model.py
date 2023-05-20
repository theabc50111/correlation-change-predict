#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import sys
import traceback
import warnings
from datetime import datetime
from itertools import islice
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import matplotlib as mpl
import numpy as np
import torch
import torch_geometric
import yaml
from torch.nn import (GRU, BatchNorm1d, Dropout, Linear, MSELoss, ReLU,
                      Sequential)
from tqdm import tqdm

current_dir = Path(__file__).parent
data_config_path = current_dir / "../config/data_config.yaml"
with open(data_config_path) as f:
    data_cfg_yaml = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data_cfg_yaml))

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    baseline_args_parser = argparse.ArgumentParser()
    baseline_args_parser.add_argument("--batch_size", type=int, nargs='?', default=32,  # each graph contains 5 days correlation, so 4 graphs means a month, 12 graphs means a quarter
                                      help="input the number of training batch")
    baseline_args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=1000,
                                      help="input the number of training epochs")
    baseline_args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                      help="input --save_model to save model weight and model info")
    baseline_args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                                      help="input the number of stride length of correlation computing")
    baseline_args_parser.add_argument("--corr_window", type=int, nargs='?', default=10,
                                      help="input the number of window length of correlation computing")
    baseline_args_parser.add_argument("--filt_mode", type=str, nargs='?', default=None,
                                      help="input the filtered mode of graph edges, look up the options by execute python ywt_library/data_module.py -h")
    baseline_args_parser.add_argument("--filt_quan", type=float, nargs='?', default=0.5,
                                      help="input the filtered quantile of graph edges")
    baseline_args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default=None,
                                      help="Decide mode of nodes' vaules of graph_nodes_matrix, look up the options by execute python ywt_library/data_module.py -h")
    baseline_args_parser.add_argument("--drop_p", type=float, default=0,
                                      help="input 0~1 to decide the probality of drop layers")
    baseline_args_parser.add_argument("--gru_l", type=int, nargs='?', default=3,  # range:1~n, for gru
                                      help="input the number of stacked-layers of gru")
    baseline_args_parser.add_argument("--gru_h", type=int, nargs='?', default=24,
                                      help="input the number of gru hidden size")
    args = baseline_args_parser.parse_args()
    logger.debug(pformat(data_cfg, indent=1, width=100, compact=True))
    logger.info(pformat(f"\n{vars(args)}", indent=1, width=40, compact=True))

    # ## Data implement & output setting & testset setting
    # data implement setting
    data_implement = "SP500_20082017_CORR_SER_REG_CORR_MAT_HRCHY_11_CLUSTER"  # watch options by operate: logger.info(data_cfg["DATASETS"].keys())
    #data_implement = "ARTIF_PARTICLE"  # watch options by operate: logger.info(data_cfg["DATASETS"].keys())
    # train set setting
    train_items_setting = "-train_train"  # -train_train|-train_all
    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
    # setting of output files
    save_model_info = args.save_model
    # set devide of pytorch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    logger.info(f"===== file_name basis:{output_file_name} =====")
    logger.info(f"===== pytorch running on:{device} =====")

    s_l, w_l = args.corr_stride, args.corr_window
    graph_adj_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/filtered_graph_adj_mat/{args.filt_mode}-quan{str(args.filt_quan).replace('.', '')}" if args.filt_mode else Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_adj_mat"
    graph_node_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_node_mat"
    model_dir = current_dir / f'save_models/{output_file_name}/corr_s{s_l}_w{w_l}'
    model_log_dir = current_dir / f'save_models/{output_file_name}/corr_s{s_l}_w{w_l}/train_logs/'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_log_dir.mkdir(parents=True, exist_ok=True)
