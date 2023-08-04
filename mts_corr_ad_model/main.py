#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import sys
import traceback
import warnings
from math import ceil
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import numpy as np
import torch
import yaml
from torch.nn import CrossEntropyLoss, MSELoss

sys.path.append("/workspace/correlation-change-predict/utils")
from metrics_utils import BinsEdgeAccuracyLoss, EdgeAccuracyLoss
from utils import convert_str_bins_list, split_and_norm_data

from baseline_model import BaselineGRU
from class_mts_corr_ad_model import ClassMTSCorrAD
from encoder_decoder import (GineEncoder, GinEncoder, MLPDecoder,
                             ModifiedInnerProductDecoder)
from graph_auto_encoder import GAE
from mts_corr_ad_model import MTSCorrAD
from mts_corr_ad_model_2 import MTSCorrAD2
from mts_corr_ad_model_3 import MTSCorrAD3

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
metrics_logger = logging.getLogger("metrics")
utils_logger = logging.getLogger("utils")
mts_corr_ad_model_logger = logging.getLogger("mts_corr_ad_model")
baseline_model_logger = logging.getLogger("baseline_model")
gae_model_logger = logging.getLogger("graph_auto_encoder")
logger.setLevel(logging.INFO)
metrics_logger.setLevel(logging.INFO)
utils_logger.setLevel(logging.INFO)
mts_corr_ad_model_logger.setLevel(logging.INFO)
baseline_model_logger.setLevel(logging.INFO)
gae_model_logger.setLevel(logging.INFO)
warnings.simplefilter("ignore")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--data_implement", type=str, nargs='?', default="PW_CONST_DIM_70_BKPS_0_NOISE_STD_10",
                             help="input the data implement name, watch options by operate: logger.info(data_cfg['DATASETS'].keys())")
    args_parser.add_argument("--batch_size", type=int, nargs='?', default=64,
                             help="input the number of batch size")
    args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=1000,
                             help="input the number of training epochs")
    args_parser.add_argument("--seq_len", type=int, nargs='?', default=10,
                             help="input the number of sequence length")
    args_parser.add_argument("--corr_type", type=str, nargs='?', default="pearson",
                             choices=["pearson", "cross_corr"],
                             help="input the type of correlation computing, the choices are [pearson, cross_corr]")
    args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                             help="input the number of stride length of correlation computing")
    args_parser.add_argument("--corr_window", type=int, nargs='?', default=50,
                             help="input the number of window length of correlation computing")
    args_parser.add_argument("--filt_mode", type=str, nargs='?', default=None,
                             help="input the filtered mode of graph edges, look up the options by execute python ywt_library/data_module.py -h")
    args_parser.add_argument("--filt_quan", type=float, nargs='?', default=None,
                             help="input the filtered quantile of graph edges")
    args_parser.add_argument("--quan_discrete_bins", type=int, nargs='?', default=None,
                             help="input the number of quantile discrete bins of graph edges")
    args_parser.add_argument("--custom_discrete_bins", type=float, nargs='*', default=None,
                             help="input the custom discrete boundaries(bins) of graph edges")
    args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default=None,
                             help="Decide mode of nodes' vaules of graph_nodes_matrix, look up the options by execute python ywt_library/data_module.py -h")
    args_parser.add_argument("--target_mats_path", type=str, nargs='?', default=None,
                             help="input the relative path of target matrices, the base directory of path is data_cfg[DIR][PIPELINE_DATA_DIR])/data_cfg[DATASETS][data_implement][OUTPUT_FILE_NAME_BASIS] + train_items_setting")
    args_parser.add_argument("--cuda_device", type=int, nargs='?', default=0,
                             help="input the number of cuda device")
    args_parser.add_argument("--train_models", type=str, nargs='+', default=["MTSCorrAD"],
                             choices=["MTSCorrAD", "MTSCorrAD2", "MTSCorrAD3", "ClassMTSCorrAD", "Baseline", "GAE"],
                             help="input to decide which models to train, the choices are [MTSCorrAD, Baseline, GAE]")
    args_parser.add_argument("--pretrain_encoder", type=str, nargs='?', default="",
                             help="input the path of pretrain encoder weights")
    args_parser.add_argument("--pretrain_decoder", type=str, nargs='?', default="",
                             help="input the path of pretrain decoder weights")
    args_parser.add_argument("--learning_rate", type=float, nargs='?', default=0.001,
                             help="input the learning rate of training")
    args_parser.add_argument("--weight_decay", type=float, nargs='?', default=0,
                             help="input the weight decay of training")
    args_parser.add_argument("--graph_enc_weight_l2_reg_lambda", type=float, nargs='?', default=0,
                             help="input the weight of graph encoder weight l2 norm loss")
    args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                             choices=["gru", "gru decoder", "decoder gru graph_encoder"],
                             help="input to decide the position of drop layers, the choices are [gru, decoder, graph_encoder]")
    args_parser.add_argument("--drop_p", type=float, default=0,
                             help="input 0~1 to decide the probality of drop layers")
    args_parser.add_argument("--gra_enc", type=str, nargs='?', default="gine",
                             help="input the type of graph encoder")
    args_parser.add_argument("--gra_enc_aggr", type=str, nargs='?', default="add",
                             help="input the type of aggregator of graph encoder")
    args_parser.add_argument("--gra_enc_l", type=int, nargs='?', default=2,  # range:1~n, for graph encoder after the second layer,
                             help="input the number of graph laryers of graph_encoder")
    args_parser.add_argument("--gra_enc_h", type=int, nargs='?', default=4,
                             help="input the number of graph embedding hidden size of graph_encoder")
    args_parser.add_argument("--gru_l", type=int, nargs='?', default=2,  # range:1~n, for gru
                             help="input the number of stacked-layers of gru")
    args_parser.add_argument("--gru_h", type=int, nargs='?', default=80,
                             help="input the number of gru hidden size")
    args_parser.add_argument("--edge_acc_loss_atol", type=float, nargs='?', default=None,
                             help="input the absolute tolerance of edge acc loss")
    args_parser.add_argument("--use_bin_edge_acc_loss", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                             help="input --use_bin_edge_acc_loss to use BinsEdgeaccuracyLoss")
    args_parser.add_argument("--output_type", type=str, nargs='?', default=None,
                             choices=["discretize", "class_probability"],
                             help="input the type of output, the choices are [discretize]")
    args_parser.add_argument("--output_bins", type=float, nargs='*', default=None,
                             help="input the bins of output")
    args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                             help="input --save_model to save model weight and model info")
    ARGS = args_parser.parse_args()
    assert bool(ARGS.drop_pos) == bool(ARGS.drop_p), "drop_pos and drop_p must be both input or not input"
    assert bool(ARGS.filt_mode) == bool(ARGS.filt_quan), "filt_mode and filt_quan must be both input or not input"
    assert (bool(ARGS.filt_mode) != bool(ARGS.quan_discrete_bins)) or (ARGS.filt_mode is None and ARGS.quan_discrete_bins is None), "filt_mode and quan_discrete_bins must be both not input or one input"
    assert ARGS.output_bins is None or ARGS.output_type == "discretize", "output_bins must be input when output_type is discretize"
    assert (ARGS.use_bin_edge_acc_loss is False and ARGS.edge_acc_loss_atol is None) or bool(ARGS.use_bin_edge_acc_loss) != bool(ARGS.edge_acc_loss_atol), "use_bin_edge_acc_loss and edge_acc_loss_atol must be both not input or one input"
    assert ARGS.use_bin_edge_acc_loss is None or ARGS.target_mats_path is not None, "target_mats_path must be input when use_bin_edge_acc_loss is input"
    assert "ClassMTSCorrAD" not in ARGS.train_models or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models is ClassMTSCorrAD"
    assert not (ARGS.use_bin_edge_acc_loss and ARGS.output_type == "class_probability"), "use_bin_edge_acc_loss and output_type can not be both input"
    assert not (ARGS.edge_acc_loss_atol and ARGS.output_type == "class_probability"), "edge_acc_loss_atol and output_type can not be both input"
    logger.info(pformat(f"\n{vars(ARGS)}", indent=1, width=40, compact=True))

    # Data implement & output setting & testset setting
    # data implement setting
    data_implement = ARGS.data_implement
    # train set setting
    train_items_setting = "-train_train"  # -train_train|-train_all
    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
    # setting of output files
    save_model_info = ARGS.save_model
    # set devide of pytorch
    device = torch.device(f'cuda:{ARGS.cuda_device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor)
    torch.autograd.set_detect_anomaly(True)  # for debug grad
    logger.info(f"===== file_name basis:{output_file_name} =====")
    logger.info(f"===== pytorch running on:{device} =====")

    s_l, w_l = ARGS.corr_stride, ARGS.corr_window
    if ARGS.filt_mode:
        graph_adj_mode_dir = f"filtered_graph_adj_mat/{ARGS.filt_mode}-quan{str(ARGS.filt_quan).replace('.', '')}"
    elif ARGS.quan_discrete_bins:
        graph_adj_mode_dir = f"quan_discretize_graph_adj_mat/bins{ARGS.quan_discrete_bins}"
    elif ARGS.custom_discrete_bins:
        graph_adj_mode_dir = f"custom_discretize_graph_adj_mat/bins_{'_'.join((str(f) for f in ARGS.custom_discrete_bins)).replace('.', '')}"
    else:
        graph_adj_mode_dir = "graph_adj_mat"
    graph_adj_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.corr_type}/{graph_adj_mode_dir}"
    graph_node_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/graph_node_mat"
    target_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.target_mats_path}"
    mts_corr_ad_model_dir = current_dir/f'save_models/mts_corr_ad_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    mts_corr_ad_model_log_dir = current_dir/f'save_models/mts_corr_ad_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    mts_corr_ad_model_2_dir = current_dir/f'save_models/mts_corr_ad_model_2/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    mts_corr_ad_model_2_log_dir = current_dir/f'save_models/mts_corr_ad_model_2/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    mts_corr_ad_model_3_dir = current_dir/f'save_models/mts_corr_ad_model_3/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    mts_corr_ad_model_3_log_dir = current_dir/f'save_models/mts_corr_ad_model_3/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    class_mts_corr_ad_model_dir = current_dir/f'save_models/class_mts_corr_ad_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    class_mts_corr_ad_model_log_dir = current_dir/f'save_models/class_mts_corr_ad_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    baseline_model_dir = current_dir/f'save_models/baseline_gru/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    baseline_model_log_dir = current_dir/f'save_models/baseline_gru/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    gae_model_dir = current_dir/f'save_models/gae_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    gae_model_log_dir = current_dir/f'save_models/gae_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    mts_corr_ad_model_dir.mkdir(parents=True, exist_ok=True)
    mts_corr_ad_model_log_dir.mkdir(parents=True, exist_ok=True)
    mts_corr_ad_model_2_dir.mkdir(parents=True, exist_ok=True)
    mts_corr_ad_model_2_log_dir.mkdir(parents=True, exist_ok=True)
    mts_corr_ad_model_3_dir.mkdir(parents=True, exist_ok=True)
    mts_corr_ad_model_3_log_dir.mkdir(parents=True, exist_ok=True)
    class_mts_corr_ad_model_dir.mkdir(parents=True, exist_ok=True)
    class_mts_corr_ad_model_log_dir.mkdir(parents=True, exist_ok=True)
    baseline_model_dir.mkdir(parents=True, exist_ok=True)
    baseline_model_log_dir.mkdir(parents=True, exist_ok=True)
    gae_model_dir.mkdir(parents=True, exist_ok=True)
    gae_model_log_dir.mkdir(parents=True, exist_ok=True)

    # model configuration
    is_training, train_count = True, 0
    gra_edges_data_mats = np.load(graph_adj_mat_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    gra_nodes_data_mats = np.load(graph_node_mat_dir/f"{ARGS.graph_nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy") if ARGS.graph_nodes_v_mode else np.ones((gra_edges_data_mats.shape[0], 1, gra_edges_data_mats.shape[2]))
    target_mats = np.load(target_mat_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy") if ARGS.target_mats_path else None
    norm_train_dataset, norm_val_dataset, norm_test_dataset, scaler = split_and_norm_data(edges_mats=gra_edges_data_mats, nodes_mats=gra_nodes_data_mats, target_mats=target_mats)
    basic_model_cfg = {"filt_mode": ARGS.filt_mode,
                       "filt_quan": ARGS.filt_quan,
                       "quan_discrete_bins": ARGS.quan_discrete_bins,
                       "custom_discrete_bins": '_'.join((str(f) for f in ARGS.custom_discrete_bins)).replace('.', '') if ARGS.custom_discrete_bins else None,
                       "graph_nodes_v_mode": ARGS.graph_nodes_v_mode,
                       "tr_epochs": ARGS.tr_epochs,
                       "batch_size": ARGS.batch_size,
                       "num_batches": {"train": ((len(norm_train_dataset["edges"])-1)//ARGS.batch_size),
                                       "val": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size),
                                       "test": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size)},
                       "seq_len": ARGS.seq_len,
                       "learning_rate": ARGS.learning_rate,
                       "weight_decay": ARGS.weight_decay,
                       "graph_enc_weight_l2_reg_lambda": ARGS.graph_enc_weight_l2_reg_lambda,
                       "drop_pos": ARGS.drop_pos,
                       "drop_p": ARGS.drop_p,
                       "gra_enc_aggr": ARGS.gra_enc_aggr,
                       "gra_enc_l": ARGS.gra_enc_l,
                       "gra_enc_h": ARGS.gra_enc_h,
                       "gru_l": ARGS.gru_l,
                       "gru_h": ARGS.gru_h if ARGS.gru_h else ARGS.gra_enc_l*ARGS.gra_enc_h,
                       "num_nodes": (norm_train_dataset["nodes"].shape[2]),
                       "num_node_features": norm_train_dataset["nodes"].shape[1],
                       "num_edge_features": 1,
                       "graph_encoder": GineEncoder if ARGS.gra_enc == "gine" else GinEncoder,
                       "decoder": MLPDecoder,
                       "output_type": ARGS.output_type,
                       "output_bins": ARGS.output_bins,
                       "target_mats_bins": ARGS.target_mats_path.split("/")[-1] if ARGS.target_mats_path else None,
                       "edge_acc_loss_atol": ARGS.edge_acc_loss_atol,
                       "use_bin_edge_acc_loss": ARGS.use_bin_edge_acc_loss}

    mts_corr_ad_cfg = basic_model_cfg.copy()
    baseline_gru_cfg = basic_model_cfg.copy()
    gae_cfg = basic_model_cfg.copy()
    mts_corr_ad_cfg["pretrain_encoder"] = ARGS.pretrain_encoder
    mts_corr_ad_cfg["pretrain_decoder"] = ARGS.pretrain_decoder
    baseline_gru_cfg["gru_in_dim"] = (norm_train_dataset['edges'].shape[1])**2
    gae_cfg.pop("seq_len"); gae_cfg.pop("gru_l"); gae_cfg.pop("gru_h")

    # show info
    logger.info(f"gra_edges_data_mats.shape:{gra_edges_data_mats.shape}, gra_nodes_data_mats.shape:{gra_nodes_data_mats.shape}")
    logger.info(f"gra_edges_data_mats.max:{np.nanmax(gra_edges_data_mats)}, gra_edges_data_mats.min:{np.nanmin(gra_edges_data_mats)}")
    logger.info(f"gra_nodes_data_mats.max:{np.nanmax(gra_nodes_data_mats)}, gra_nodes_data_mats.min:{np.nanmin(gra_nodes_data_mats)}")
    logger.info(f"norm_train_nodes_data_mats.max:{np.nanmax(norm_train_dataset['nodes'])}, norm_train_nodes_data_mats.min:{np.nanmin(norm_train_dataset['nodes'])}")
    logger.info(f"norm_val_nodes_data_mats.max:{np.nanmax(norm_val_dataset['nodes'])}, norm_val_nodes_data_mats.min:{np.nanmin(norm_val_dataset['nodes'])}")
    logger.info(f"norm_test_nodes_data_mats.max:{np.nanmax(norm_test_dataset['nodes'])}, norm_test_nodes_data_mats.min:{np.nanmin(norm_test_dataset['nodes'])}")
    logger.info(f'Training set   = {len(norm_train_dataset["edges"])} graphs')
    logger.info(f'Validation set = {len(norm_val_dataset["edges"])} graphs')
    logger.info(f'Test set       = {len(norm_test_dataset["edges"])} graphs')
    logger.info("="*80)

    loss_fns_dict = {"fns": [MSELoss()],
                     "fn_args": {"MSELoss()": {}}}
    if ARGS.output_type == "class_probability":
        loss_fns_dict["fns"].clear(); loss_fns_dict["fn_args"].clear()
        loss_fns_dict["fns"].append(CrossEntropyLoss())
        loss_fns_dict["fn_args"].update({"CrossEntropyLoss()": {}})
    elif ARGS.use_bin_edge_acc_loss is True:
        bins_list = convert_str_bins_list(ARGS.target_mats_path.split("/")[-1])
        loss_fns_dict["fns"].append(BinsEdgeAccuracyLoss())
        loss_fns_dict["fn_args"].update({"BinsEdgeAccuracyLoss()": {"bins_list": bins_list}})
    else:
        loss_fns_dict["fns"].append(EdgeAccuracyLoss())
        loss_fns_dict["fn_args"].update({"EdgeAccuracyLoss()": {"atol": ARGS.edge_acc_loss_atol}})
    while (is_training is True) and (train_count < 100):
        try:
            train_count += 1
            if "MTSCorrAD" in ARGS.train_models:
                model = MTSCorrAD(mts_corr_ad_cfg)
                best_model, best_model_info = model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
            if "MTSCorrAD2" in ARGS.train_models:
                model_2 = MTSCorrAD2(mts_corr_ad_cfg)
                best_mts_corr_ad_2_model, best_mts_corr_ad_2_model_info = model_2.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
            if "MTSCorrAD3" in ARGS.train_models:
                model_3 = MTSCorrAD3(mts_corr_ad_cfg)
                best_mts_corr_ad_3_model, best_mts_corr_ad_3_model_info = model_3.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
            if "ClassMTSCorrAD" in ARGS.train_models:
                class_model = ClassMTSCorrAD(mts_corr_ad_cfg)
                best_class_mts_corr_ad_model, best_class_mts_corr_ad_model_info = class_model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
            if "Baseline" in ARGS.train_models:
                baseline_model = BaselineGRU(baseline_gru_cfg)
                best_baseline_model, best_baseline_model_info = baseline_model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs)
            if "GAE" in ARGS.train_models:
                gae_model = GAE(gae_cfg)
                best_gae_model, best_gae_model_info = gae_model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
        except AssertionError as e:
            logger.error(f"\n{e}")
        except Exception as e:
            is_training = False
            error_class = e.__class__.__name__  # 取得錯誤類型
            detail = e.args[0]  # 取得詳細內容
            cl, exc, tb = sys.exc_info()  # 取得Call Stack
            last_call_stack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
            file_name = last_call_stack[0]  # 取得發生的檔案名稱
            line_num = last_call_stack[1]  # 取得發生的行號
            func_name = last_call_stack[2]  # 取得發生的函數名稱
            err_msg = "File \"{}\", line {}, in {}: [{}] {}".format(file_name, line_num, func_name, error_class, detail)
            logger.error(f"===\n{err_msg}")
            logger.error(f"===\n{traceback.extract_tb(tb)}")
        else:
            is_training = False
            if save_model_info:
                if "MTSCorrAD" in ARGS.train_models:
                    model.save_model(best_model, best_model_info, model_dir=mts_corr_ad_model_dir, model_log_dir=mts_corr_ad_model_log_dir)
                if "MTSCorrAD2" in ARGS.train_models:
                    model_2.save_model(best_mts_corr_ad_2_model, best_mts_corr_ad_2_model_info, model_dir=mts_corr_ad_model_2_dir, model_log_dir=mts_corr_ad_model_2_log_dir)
                if "MTSCorrAD3" in ARGS.train_models:
                    model_3.save_model(best_mts_corr_ad_3_model, best_mts_corr_ad_3_model_info, model_dir=mts_corr_ad_model_3_dir, model_log_dir=mts_corr_ad_model_3_log_dir)
                if "ClassMTSCorrAD" in ARGS.train_models:
                    class_model.save_model(best_class_mts_corr_ad_model, best_class_mts_corr_ad_model_info, model_dir=class_mts_corr_ad_model_dir, model_log_dir=class_mts_corr_ad_model_log_dir)
                if "Baseline" in ARGS.train_models:
                    baseline_model.save_model(best_baseline_model, best_baseline_model_info, model_dir=baseline_model_dir, model_log_dir=baseline_model_log_dir)
                if "GAE" in ARGS.train_models:
                    gae_model.save_model(best_gae_model, best_gae_model_info, model_dir=gae_model_dir, model_log_dir=gae_model_log_dir)
