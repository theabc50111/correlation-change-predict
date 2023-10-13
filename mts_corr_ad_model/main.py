#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import sys
import traceback
import warnings
from enum import Enum, auto
from itertools import zip_longest
from math import ceil, sqrt
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import numpy as np
import torch
import yaml
from torch.nn import CrossEntropyLoss, MSELoss

sys.path.append("/workspace/correlation-change-predict/utils")
from metrics_utils import (BinsEdgeAccuracyLoss, CustomIndicesEdgeAccuracy,
                           EdgeAccuracyLoss, TwoOrderPredProbEdgeAccuracy,
                           UpperTriangleEdgeAccuracy)
from utils import convert_str_bins_list, plot_heatmap, split_and_norm_data

from baseline_model import BaselineGRU
from class_baseline_model import (ClassBaselineGRU, ClassBaselineGRUOneFeature,
                                  ClassBaselineGRUWithoutSelfCorr)
from class_mts_corr_ad_model import ClassMTSCorrAD
from class_mts_corr_ad_model_3 import ClassMTSCorrAD3
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


class ModelType(Enum):
    MTSCORRAD = auto()
    MTSCORRAD2 = auto()
    MTSCORRAD3 = auto()
    CLASSMTSCORRAD = auto()
    CLASSMTSCORRAD3 = auto()
    BASELINE = auto()
    CLASSBASELINE = auto()
    CLASSBASELINEWITHOUTSELFCORR = auto()
    CLASSBASELINEONEFEATURE = auto()
    GAE = auto()

    def set_model(self, basic_model_cfg, args):
        mts_corr_ad_cfg = basic_model_cfg.copy()
        baseline_gru_cfg = basic_model_cfg.copy()
        gae_cfg = basic_model_cfg.copy()
        mts_corr_ad_cfg["pretrain_encoder"] = args.pretrain_encoder
        mts_corr_ad_cfg["pretrain_decoder"] = args.pretrain_decoder
        baseline_gru_cfg["gru_in_dim"] = basic_model_cfg["num_nodes"]**2
        baseline_gru_cfg["pretrain_encoder"] = None
        baseline_gru_cfg["pretrain_decoder"] = None
        baseline_gru_without_self_corr_cfg = baseline_gru_cfg.copy()
        baseline_gru_without_self_corr_cfg["gru_in_dim"] = int((basic_model_cfg["num_nodes"]-1)/2*(1+basic_model_cfg["num_nodes"]-1))
        baseline_gru_one_feature_cfg = baseline_gru_cfg.copy()
        baseline_gru_one_feature_cfg["gru_in_dim"] = 1
        baseline_gru_one_feature_cfg["input_feature_idx"] = args.gru_input_feature_idx
        gae_cfg.pop("seq_len"); gae_cfg.pop("gru_l"); gae_cfg.pop("gru_h")
        assert ((basic_model_cfg["num_nodes"]-1)/2*(1+basic_model_cfg["num_nodes"]-1)).is_integer(), "baseline_gru_without_self_corr_cfg[gru_in_dim] is not an integer"
        model_dict = {"MTSCORRAD": MTSCorrAD(mts_corr_ad_cfg),
                      "MTSCORRAD2": MTSCorrAD2(mts_corr_ad_cfg),
                      "MTSCORRAD3": MTSCorrAD3(mts_corr_ad_cfg),
                      "CLASSMTSCORRAD": ClassMTSCorrAD(mts_corr_ad_cfg),
                      "CLASSMTSCORRAD3": ClassMTSCorrAD3(mts_corr_ad_cfg),
                      "BASELINE": BaselineGRU(baseline_gru_cfg),
                      "CLASSBASELINE": ClassBaselineGRU(baseline_gru_cfg),
                      "CLASSBASELINEWITHOUTSELFCORR": ClassBaselineGRUWithoutSelfCorr(baseline_gru_without_self_corr_cfg),
                      "CLASSBASELINEONEFEATURE": ClassBaselineGRUOneFeature(baseline_gru_one_feature_cfg),
                      "GAE": GAE(gae_cfg)}
        model = model_dict[self.name]
        assert ModelType.__members__.keys() == model_dict.keys(), "ModelType members and model_dict must be the same keys"

        return model

    def set_save_model_dir(self, current_dir, output_file_name, corr_type, s_l, w_l):
        save_model_dir_base_dict = {"MTSCORRAD": "mts_corr_ad_model",
                                    "MTSCORRAD2": "mts_corr_ad_model_2",
                                    "MTSCORRAD3": "mts_corr_ad_model_3",
                                    "CLASSMTSCORRAD": "class_mts_corr_ad_model",
                                    "CLASSMTSCORRAD3": "class_mts_corr_ad_model_3",
                                    "BASELINE": "baseline_gru",
                                    "CLASSBASELINE": "class_baseline_gru",
                                    "CLASSBASELINEWITHOUTSELFCORR": "class_baseline_gru_without_self_corr",
                                    "CLASSBASELINEONEFEATURE": "class_baseline_gru_one_feature",
                                    "GAE": "gae_model"}
        assert ModelType.__members__.keys() == save_model_dir_base_dict.keys(), "ModelType members and save_model_dir_base_dict must be the same keys"
        model_dir = current_dir/f'save_models/{save_model_dir_base_dict[self.name]}/{output_file_name}/{corr_type}/corr_s{s_l}_w{w_l}'
        model_log_dir = current_dir/f'save_models/{save_model_dir_base_dict[self.name]}/{output_file_name}/{corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_log_dir.mkdir(parents=True, exist_ok=True)

        return model_dir, model_log_dir


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--data_implement", type=str, nargs='?', default="PW_CONST_DIM_70_BKPS_0_NOISE_STD_10",
                             help="input the data implement name, watch options by operate: logger.info(data_cfg['DATASETS'].keys())")
    args_parser.add_argument("--batch_size", type=int, nargs='?', default=64,
                             help="input the number of batch size")
    args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=1500,
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
    args_parser.add_argument("--train_models", type=str, nargs='+', default=[],
                             choices=["MTSCORRAD", "MTSCORRAD2", "MTSCORRAD3", "CLASSMTSCORRAD", "CLASSMTSCORRAD3", "BASELINE", "CLASSBASELINE", "CLASSBASELINEWITHOUTSELFCORR", "CLASSBASELINEONEFEATURE", "GAE"],
                             help="input to decide which models to train, the choices are [MTSCorrAD, Baseline, GAE]")
    args_parser.add_argument("--pretrain_encoder", type=str, nargs='?', default="",
                             help="input the path of pretrain encoder weights")
    args_parser.add_argument("--pretrain_decoder", type=str, nargs='?', default="",
                             help="input the path of pretrain decoder weights")
    args_parser.add_argument("--learning_rate", type=float, nargs='?', default=0.001,
                             help="input the learning rate of training")
    args_parser.add_argument("--weight_decay", type=float, nargs='?', default=0,
                             help="input the weight decay of training")
    args_parser.add_argument("--use_optim_scheduler", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --use_optim_scheduler to use optimizer scheduler")
    args_parser.add_argument("--graph_enc_weight_l2_reg_lambda", type=float, nargs='?', default=0,
                             help="input the weight of graph encoder weight l2 norm loss")
    args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                             choices=["gru", "decoder", "graph_encoder", "class_fc"],
                             help="input to decide the position of drop layers, the choices are [gru, decoder, graph_encoder]")
    args_parser.add_argument("--drop_p", type=float, default=0,
                             help="input 0~1 to decide the probality of drop layers")
    args_parser.add_argument("--gra_enc", type=str, nargs='?', default="gine",
                             help="input the type of graph encoder")
    args_parser.add_argument("--gra_enc_aggr", type=str, nargs='?', default="add",
                             help="input the type of aggregator of graph encoder")
    args_parser.add_argument("--gra_enc_l", type=int, nargs='?', default=2,  # range:1~n, for graph encoder layer,
                             help="input the number of gnn laryers of graph_encoder")
    args_parser.add_argument("--gra_enc_h", type=int, nargs='?', default=4,
                             help="input the number of graph embedding hidden size of graph_encoder")
    args_parser.add_argument("--gra_enc_mlp_l", type=int, nargs='?', default=2,  # range:1~n, for graph encoder mlp layer,
                             help="input the number of graph mlp laryers of graph_encoder")
    args_parser.add_argument("--gru_l", type=int, nargs='?', default=2,  # range:1~n, for gru
                             help="input the number of stacked-layers of gru")
    args_parser.add_argument("--gru_h", type=int, nargs='?', default=80,
                             help="input the number of gru hidden size")
    args_parser.add_argument("--gru_input_feature_idx", type=int, nargs='?', default=None,
                             help="input the order of input features of gru, the order is from 0 to combination(num_nodes, 2)-1")
    args_parser.add_argument("--two_ord_pred_prob_edge_accu_thres", type=float, nargs='?', default=None,
                             help="input the threshold of TwoOrderPredProbEdgeAccuracy")
    args_parser.add_argument("--edge_acc_loss_atol", type=float, nargs='?', default=None,
                             help="input the absolute tolerance of EdgeAccuracyLoss")
    args_parser.add_argument("--use_weighted_loss", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --use_weighted_loss to use CrossEntropyLoss weight")
    args_parser.add_argument("--use_bin_edge_acc_loss", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                             help="input --use_bin_edge_acc_loss to use BinsEdgeaccuracyLoss")
    args_parser.add_argument("--use_two_ord_pred_prob_edge_acc_metric", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --use_two_ord_pred_prob_edge_acc_metric to use TwoOrderPredProbEdgeAccuracy")
    args_parser.add_argument("--use_upper_tri_edge_acc_metric", type=bool, default=False, action=argparse.BooleanOptionalAction,
                             help="input --use_upper_tri_edge_acc_metric to use UpperTriangleEdgeAccuracy")
    args_parser.add_argument("--custom_indices_edge_acc_metric_indices", type=int, nargs='*', default=[],
                             help="input the indices of CustomIndicesEdgeAccuracy")
    args_parser.add_argument("--output_type", type=str, nargs='?', default=None,
                             choices=["discretize", "class_probability"],
                             help="input the type of output, the choices are [discretize]")
    args_parser.add_argument("--output_bins", type=float, nargs='*', default=None,
                             help="input the bins of output")
    args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                             help="input --save_model to save model weight and model info")
    args_parser.add_argument("--inference_models", type=str, nargs='+', default=[],
                             choices=["MTSCORRAD", "MTSCORRAD2", "MTSCORRAD3", "CLASSMTSCORRAD", "CLASSMTSCORRAD3", "BASELINE", "CLASSBASELINE", "CLASSBASELINEWITHOUTSELFCORR", "CLASSBASELINEONEFEATURE", "GAE"],
                             help="input to decide which models to train, the choices are [MTSCorrAD, MTSCORRAD2, BASELINE, GAE]")
    args_parser.add_argument("--inference_model_paths", type=str, nargs='+', default=[],
                             help="input the path of inference model weight")
    args_parser.add_argument("--inference_data_split", type=str, nargs='?', default="val",
                             help="input the data split of inference data, the choices are [train, val, test]")
    ARGS = args_parser.parse_args()
    assert bool(ARGS.train_models) != bool(ARGS.inference_models), "train_models and inference_models must be input one of them"
    assert bool(ARGS.drop_pos) == bool(ARGS.drop_p), "drop_pos and drop_p must be both input or not input"
    assert bool(ARGS.filt_mode) == bool(ARGS.filt_quan), "filt_mode and filt_quan must be both input or not input"
    assert (bool(ARGS.filt_mode) != bool(ARGS.quan_discrete_bins)) or (ARGS.filt_mode is None and ARGS.quan_discrete_bins is None), "filt_mode and quan_discrete_bins must be both not input or one input"
    assert ARGS.output_bins is None or ARGS.output_type == "discretize", "output_bins must be input when output_type is discretize"
    assert (ARGS.use_bin_edge_acc_loss is False and ARGS.edge_acc_loss_atol is None) or bool(ARGS.use_bin_edge_acc_loss) != bool(ARGS.edge_acc_loss_atol), "use_bin_edge_acc_loss and edge_acc_loss_atol must be both not input or one input"
    assert ARGS.use_bin_edge_acc_loss is None or ARGS.target_mats_path is not None, "target_mats_path must be input when use_bin_edge_acc_loss is input"
    assert ("MTSCORRAD" not in ARGS.train_models) or (ARGS.output_type != "class_probability"), "output_type can not be class_probability when train_models is MTSCorrAD"
    assert ("MTSCORRAD3" not in ARGS.train_models) or (ARGS.output_type != "class_probability"), "output_type can not be class_probability when train_models is MTSCorrAD3"
    assert "CLASSMTSCORRAD" not in ARGS.train_models or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models is ClassMTSCorrAD"
    assert "CLASSMTSCORRAD3" not in ARGS.train_models or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models is ClassMTSCorrAD3"
    assert "CLASSBASELINE" not in ARGS.train_models or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models is ClassBaseline"
    assert "CLASSBASELINEWITHOUTSELFCORR" not in ARGS.train_models or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models is ClassBaselineWithoutSelfCorr"
    assert "CLASSBASELINEONEFEATURE" not in ARGS.train_models or ARGS.output_type == "class_probability", "output_type must be class_probability when train_models is ClassBaselineOneFeatureGRUWithoutSelfCorr"
    assert ARGS.two_ord_pred_prob_edge_accu_thres is not None or ARGS.output_type == "class_probability", "output_type must be class_probability when two_ord_pred_prob_edge_accu_thres is input"
    assert "class_fc" not in ARGS.drop_pos or ARGS.output_type == "class_probability", "output_type must be class_probability when class_fc in drop_pos"
    assert "CLASSBASELINEONEFEATURE" not in ARGS.train_models or ARGS.gru_input_feature_idx is not None, "gru_input_feature_idx must be input when train_models is ClassBaselineOneFeatureGRUWithoutSelfCorr"
    assert not (ARGS.use_bin_edge_acc_loss and ARGS.output_type == "class_probability"), "use_bin_edge_acc_loss can't be input when output_type is class_probability"
    assert not (ARGS.edge_acc_loss_atol and ARGS.output_type == "class_probability"), "edge_acc_loss_atol and output_type can not be both input"
    assert not ARGS.use_two_ord_pred_prob_edge_acc_metric or ARGS.two_ord_pred_prob_edge_accu_thres is not None, "two_ord_pred_prob_edge_accu_thres must be input when use_two_ord_pred_prob_edge_acc_metric is input"
    ###assert not (ARGS.use_upper_tri_edge_acc_metric and ARGS.use_two_ord_pred_prob_edge_acc_metric), "use_upper_tri_edge_acc_metric and use_two_ord_pred_prob_edge_acc_metric can not be both input"
    assert ARGS.use_upper_tri_edge_acc_metric + ARGS.use_two_ord_pred_prob_edge_acc_metric + bool(ARGS.custom_indices_edge_acc_metric_indices) <= 1, "use_upper_tri_edge_acc_metric, use_two_ord_pred_prob_edge_acc_metric and custom_indices_edge_acc_metric_indices can not be both input"
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

    # model configuration
    gra_edges_data_mats = np.load(graph_adj_mat_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    gra_nodes_data_mats = np.load(graph_node_mat_dir/f"{ARGS.graph_nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy") if ARGS.graph_nodes_v_mode else np.ones((gra_edges_data_mats.shape[0], 1, gra_edges_data_mats.shape[2]))
    target_mats = np.load(target_mat_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy") if ARGS.target_mats_path else None
    norm_train_dataset, norm_val_dataset, norm_test_dataset, scaler = split_and_norm_data(edges_mats=gra_edges_data_mats, nodes_mats=gra_nodes_data_mats, target_mats=target_mats, batch_size= ARGS.batch_size)
    basic_model_cfg = {"filt_mode": ARGS.filt_mode,
                       "filt_quan": ARGS.filt_quan,
                       "quan_discrete_bins": ARGS.quan_discrete_bins,
                       "custom_discrete_bins": '_'.join((str(f) for f in ARGS.custom_discrete_bins)).replace('.', '') if ARGS.custom_discrete_bins else None,
                       "graph_nodes_v_mode": ARGS.graph_nodes_v_mode,
                       "tr_epochs": ARGS.tr_epochs,
                       "batch_size": ARGS.batch_size,
                       "num_batches": {"train": ceil((len(norm_train_dataset["edges"])-ARGS.seq_len)/ARGS.batch_size),
                                       "val": ceil((len(norm_val_dataset["edges"])-ARGS.seq_len)/ARGS.batch_size),
                                       "test": ceil((len(norm_val_dataset["edges"])-ARGS.seq_len)/ARGS.batch_size)},
                       "seq_len": ARGS.seq_len,
                       "learning_rate": ARGS.learning_rate,
                       "weight_decay": ARGS.weight_decay,
                       "can_use_optim_scheduler": ARGS.use_optim_scheduler,
                       "graph_enc_weight_l2_reg_lambda": ARGS.graph_enc_weight_l2_reg_lambda,
                       "drop_pos": ARGS.drop_pos,
                       "drop_p": ARGS.drop_p,
                       "gra_enc_aggr": ARGS.gra_enc_aggr,
                       "gra_enc_l": ARGS.gra_enc_l,
                       "gra_enc_h": ARGS.gra_enc_h,
                       "gra_enc_mlp_l": ARGS.gra_enc_mlp_l,
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
                       "two_ord_pred_prob_edge_accu_thres": ARGS.two_ord_pred_prob_edge_accu_thres,
                       "use_bin_edge_acc_loss": ARGS.use_bin_edge_acc_loss}

    loss_fns_dict = {"fns": [MSELoss()],
                     "fn_args": {"MSELoss()": {}}}
    if ARGS.output_type == "class_probability":
        loss_fns_dict["fns"].clear(); loss_fns_dict["fn_args"].clear()
        if ARGS.use_weighted_loss:
            tr_labels, tr_labels_freq_counts = np.unique(norm_train_dataset['target'], return_counts=True)
            weight = torch.tensor(np.reciprocal(tr_labels_freq_counts/tr_labels_freq_counts.sum()))
        loss_fns_dict["fns"].append(CrossEntropyLoss(weight if ARGS.use_weighted_loss else None))
        loss_fns_dict["fn_args"].update({"CrossEntropyLoss()": {}})
    elif ARGS.use_bin_edge_acc_loss is True:
        bins_list = convert_str_bins_list(ARGS.target_mats_path.split("/")[-1])
        loss_fns_dict["fns"].append(BinsEdgeAccuracyLoss())
        loss_fns_dict["fn_args"].update({"BinsEdgeAccuracyLoss()": {"bins_list": bins_list}})
    elif ARGS.edge_acc_loss_atol is not None:
        loss_fns_dict["fns"].append(EdgeAccuracyLoss())
        loss_fns_dict["fn_args"].update({"EdgeAccuracyLoss()": {"atol": ARGS.edge_acc_loss_atol}})

    # setting of metric function of edge_accuracy of model
    if ARGS.use_two_ord_pred_prob_edge_acc_metric:
        num_classes = ARGS.target_mats_path.split("/")[-1].replace("bins_", "").count("_")
        basic_model_cfg["edge_acc_metric_fn"] = TwoOrderPredProbEdgeAccuracy(threshold=ARGS.two_ord_pred_prob_edge_accu_thres, num_classes=num_classes)
    elif ARGS.use_upper_tri_edge_acc_metric:
        num_classes = ARGS.target_mats_path.split("/")[-1].replace("bins_", "").count("_")
        basic_model_cfg["edge_acc_metric_fn"] = UpperTriangleEdgeAccuracy(num_classes=num_classes)
    elif ARGS.custom_indices_edge_acc_metric_indices:
        num_classes = ARGS.target_mats_path.split("/")[-1].replace("bins_", "").count("_")
        basic_model_cfg["edge_acc_metric_fn"] = CustomIndicesEdgeAccuracy(selected_indices=ARGS.custom_indices_edge_acc_metric_indices, num_classes=num_classes)

    # show info
    logger.info(f"===== file_name basis:{output_file_name} =====")
    logger.info(f"===== pytorch running on:{device} =====")
    logger.info(f"gra_edges_data_mats.shape:{gra_edges_data_mats.shape}, gra_nodes_data_mats.shape:{gra_nodes_data_mats.shape}, target_mats.shape:{target_mats.shape if target_mats is not None else None}")
    logger.info(f"gra_edges_data_mats.max:{np.nanmax(gra_edges_data_mats)}, gra_edges_data_mats.min:{np.nanmin(gra_edges_data_mats)}")
    logger.info(f"gra_nodes_data_mats.max:{np.nanmax(gra_nodes_data_mats)}, gra_nodes_data_mats.min:{np.nanmin(gra_nodes_data_mats)}")
    logger.info(f"norm_train_nodes_data_mats.max:{np.nanmax(norm_train_dataset['nodes'])}, norm_train_nodes_data_mats.min:{np.nanmin(norm_train_dataset['nodes'])}")
    logger.info(f"norm_val_nodes_data_mats.max:{np.nanmax(norm_val_dataset['nodes'])}, norm_val_nodes_data_mats.min:{np.nanmin(norm_val_dataset['nodes'])}")
    logger.info(f"norm_test_nodes_data_mats.max:{np.nanmax(norm_test_dataset['nodes'])}, norm_test_nodes_data_mats.min:{np.nanmin(norm_test_dataset['nodes'])}")
    logger.info(f'Training set   = {len(norm_train_dataset["edges"])} graphs')
    logger.info(f'Validation set = {len(norm_val_dataset["edges"])} graphs')
    logger.info(f'Test set       = {len(norm_test_dataset["edges"])} graphs')
    logger.info("="*80)

    if len(ARGS.train_models) > 0:
        assert list(filter(lambda x: x in ModelType.__members__.keys(), ARGS.train_models)), f"train_models must be input one of {ModelType.__members__.keys()}"
        for model_type in ModelType:
            is_training, train_count = True, 0
            while (model_type.name in ARGS.train_models) and (is_training is True) and (train_count < 100):
                try:
                    logger.info(f"===== train model:{model_type.name} =====")
                    train_count += 1
                    model_dir, model_log_dir = model_type.set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
                    model = model_type.set_model(basic_model_cfg, ARGS)
                    best_model, best_model_info = model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
                except AssertionError as e:
                    logger.error(f"\n{e}")
                except Exception as e:
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
                        model_dir, model_log_dir = model_type.set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
                        model.save_model(best_model, best_model_info, model_dir=model_dir, model_log_dir=model_log_dir)
    elif len(ARGS.inference_models) > 0:
        logger.info(f"===== inference model:[{ARGS.inference_models}] on {ARGS.inference_data_split} data =====")
        logger.info(f"===== if inference_models is more than one, the inference result is ensemble result =====")
        assert list(filter(lambda x: x in ModelType.__members__.keys(), ARGS.inference_models)), f"inference_models must be input one of {ModelType.__members__.keys()}"
        if ARGS.inference_data_split == "train":
            inference_data = norm_train_dataset
        elif ARGS.inference_data_split == "val":
            inference_data = norm_val_dataset
        elif ARGS.inference_data_split == "test":
            inference_data = norm_test_dataset
        loss = None
        edge_acc = None
        if len(ARGS.inference_models) == 1:
            model_type = ModelType[ARGS.inference_models[0]]
            model = model_type.set_model(basic_model_cfg, ARGS)
            model_dir, _ = model_type.set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
            model_param_path = model_dir.parents[2].joinpath(ARGS.inference_model_paths[0])
            assert model_param_path.exists(), f"{model_param_path} not exists"
            model.load_state_dict(torch.load(model_param_path, map_location=device))
            model.eval()
            loss, edge_acc, preds, y_labels = model.test(inference_data, loss_fns=loss_fns_dict)
            conf_mat_save_fig_dir = current_dir/f"exploration_model_result/model_result_figs/{ARGS.inference_models[0]}/{model_param_path.stem}"
            conf_mat_save_fig_name = f'confusion_matrix-{ARGS.inference_data_split}.png'
        elif len(ARGS.inference_models) > 1:
            assert sorted(ARGS.inference_models) == ARGS.inference_models, f"inference_models must be input in order, but the input order is {ARGS.inference_models}"
            ensemble_pred_prob = 0
            ensemble_weights = [10, 12]
            for model_type, infer_model_path, weight in zip_longest(ARGS.inference_models, ARGS.inference_model_paths, ensemble_weights, fillvalue=None):
                model = ModelType[model_type].set_model(basic_model_cfg, ARGS)
                model_dir, _ = ModelType[model_type].set_save_model_dir(current_dir, output_file_name, ARGS.corr_type, s_l, w_l)
                model_param_path = model_dir.parents[2].joinpath(infer_model_path)
                assert model_param_path.exists(), f"{model_param_path} not exists"
                model.load_state_dict(torch.load(model_param_path, map_location=device))
                model.eval()
                if "MTSCORRAD" in model_type:
                    if ARGS.inference_data_split in ["val", "test"]:
                        model.model_cfg["batch_size"] = inference_data["edges"].shape[0]-1-ARGS.seq_len
                    test_loader = model.create_pyg_data_loaders(graph_adj_mats=inference_data["edges"],  graph_nodes_mats=inference_data["nodes"], target_mats=inference_data["target"], loader_seq_len=model.model_cfg["seq_len"], show_log=True)
                elif "BASELINE" in model_type:
                    test_loader = model.yield_batch_data(graph_adj_mats=inference_data['edges'], target_mats=inference_data['target'], batch_size=model.model_cfg['batch_size'], seq_len=model.model_cfg['seq_len'])
                for batch_idx, batch_data in enumerate(test_loader):
                    infer_res = model.infer_batch_data(batch_data)
                    batch_pred_prob, batch_preds, batch_y_labels = infer_res[0], infer_res[1], infer_res[2]
                    all_batch_pred_prob = batch_pred_prob if batch_idx == 0 else torch.cat((all_batch_pred_prob, batch_pred_prob), dim=0)
                    y_labels = batch_y_labels if batch_idx == 0 else torch.cat((y_labels, batch_y_labels), dim=0)
                ensemble_pred_prob += all_batch_pred_prob*weight
            preds = torch.argmax(ensemble_pred_prob, dim=1)
            if "edge_acc_metric_fn" in basic_model_cfg.keys():
                edge_acc = basic_model_cfg["edge_acc_metric_fn"](preds, y_labels)
            else:
                edge_acc = preds.eq(y_labels).to(torch.float).mean()
            model_param_paths = [Path(model_path).stem for model_path in ARGS.inference_model_paths]
            conf_mat_save_fig_dir = current_dir/f"exploration_model_result/model_result_figs/ensemble_{'_'.join(ARGS.inference_models)}"/'-'.join(model_param_paths)
            conf_mat_save_fig_name = 'confusion_matrix-ensemble_rate_'+'_'.join([str(w) for w in ensemble_weights])+f'-{ARGS.inference_data_split}.png'

        assert preds.shape == y_labels.shape, f"preds.shape:{preds.shape} != y_labels.shape:{y_labels.shape}"
        preds, y_labels = preds.cpu().numpy(), y_labels.cpu().numpy()
        conf_mat_save_fig_path = conf_mat_save_fig_dir/conf_mat_save_fig_name
        conf_mat_save_fig_dir.mkdir(parents=True, exist_ok=True)
        if ARGS.use_upper_tri_edge_acc_metric:
            assert sqrt(y_labels.shape[1]).is_integer(), f"y_labels.shape[1]:{y_labels.shape[1]} is not square number"
            num_edges = int(sqrt(y_labels.shape[1]))
            idx_upper_tri = np.triu_indices(num_edges, k=1)
            preds, y_labels = preds.reshape(-1, num_edges, num_edges), y_labels.reshape(-1, num_edges, num_edges)
            preds = preds[:, idx_upper_tri[0], idx_upper_tri[1]]
            y_labels = y_labels[:, idx_upper_tri[0], idx_upper_tri[1]]
        plot_heatmap(preds, y_labels, can_show_conf_mat=True, save_fig_path=conf_mat_save_fig_path)
        loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        edge_acc = edge_acc.item() if isinstance(edge_acc, torch.Tensor) else edge_acc
        logger.info(f"loss_fns:{loss_fns_dict['fns']}")
        logger.info(f"metric_fn:{basic_model_cfg['edge_acc_metric_fn'] if 'edge_acc_metric_fn' in basic_model_cfg.keys() else None}")
        logger.info(f"Special args of loss_fns: {[(loss_fn, loss_args) for loss_fn, loss_args in loss_fns_dict['fn_args'].items() for arg in loss_args if arg not in ['input', 'target']]}")
        logger.info(f"loss:{loss}, edge_acc:{edge_acc}")
