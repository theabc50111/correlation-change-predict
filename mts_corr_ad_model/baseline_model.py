#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import json
import logging
import sys
import traceback
import warnings
from collections import OrderedDict
from datetime import datetime
from itertools import islice
from math import ceil
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

sys.path.append("/workspace/correlation-change-predict/utils")
from utils import split_and_norm_data

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


class BaselineGRUModel(torch.nn.Module):
    def __init__(self, dim_in: int, gru_l: int, gru_h: int, dim_out: int, drop_p: float, **unused_kwargs):
        super(BaselineGRUModel, self).__init__()
        self.dim_in = dim_in
        self.gru_l = gru_l
        self.gru_h = gru_h
        self.dim_out = dim_out
        self.drop_p = drop_p
        self.gru = GRU(input_size=self.dim_in, hidden_size=self.gru_h, num_layers=self.gru_l, dropout=drop_p)
        self.fc = Linear(in_features=self.gru_h, out_features=self.dim_out)
        self.loss_fn = MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=50, gamma=0.5)


    def forward(self, x):
        gru_output, gru_hn = self.gru(x)
        pred = self.fc(gru_output[-1, :])  # Only use the output of the last time step
        return pred


    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, epochs: int = 1000, args: argparse.Namespace = None):
        super().train(mode=mode)
        if train_data is None:
            return self

        best_model_info = {"num_training_graphs": len(train_data),
                           "filt_mode": args.filt_mode,
                           "filt_quan": args.filt_quan,
                           "graph_nodes_v_mode": args.graph_nodes_v_mode,
                           "batchs_per_epoch": ceil(len(train_data)//args.batch_size),
                           "epochs": epochs,
                           "train_batch": args.batch_size,
                           "val_batch": args.batch_size,
                           "optimizer": str(self.optim),
                           "loss_fn": str(self.loss_fn.__name__ if hasattr(self.loss_fn, '__name__') else str(self.loss_fn)),
                           "min_val_loss": float('inf'),
                           "train_loss_history": [],
                           "val_loss_history": [],
                           "train_edge_acc_history": [],
                           "val_edge_acc_history": []}

        best_model = []
        num_batchs = ceil(len(train_data)//args.batch_size)
        for epoch_i in tqdm(range(epochs)):
            epoch_loss = {"tr": torch.zeros(1), "val": torch.zeros(1)}
            epoch_edge_acc = {"tr": torch.zeros(1), "val": torch.zeros(1)}
            # Train on batches
            batch_data_generator = self.yield_batch_data(graph_adj_arr=train_data, batch_size=args.batch_size)
            for batched_data in batch_data_generator:
                x, y = batched_data[0], batched_data[1]
                torch.autograd.set_detect_anomaly(True)
                pred = self.forward(x)
                pred, y = pred.reshape(1, -1), y.reshape(1, -1)
                loss = self.loss_fn(pred, y)
                edge_acc = np.isclose(pred.cpu().detach().numpy(), y.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                epoch_edge_acc["tr"] += edge_acc / num_batchs
                epoch_loss["tr"] += loss / num_batchs
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # Validation
            epoch_loss['val'], epoch_edge_acc['val'] = self.test(val_data, args=args)
            self.train()

            # record training history
            best_model_info["train_loss_history"].append(epoch_loss["tr"].item())
            best_model_info["val_loss_history"].append(epoch_loss['val'].item())
            best_model_info["train_edge_acc_history"].append(epoch_edge_acc["tr"].item())
            best_model_info["val_edge_acc_history"].append(epoch_edge_acc['val'].item())
            if epoch_i==0:
                best_model_info["model_structure"] = str(self)

            # record training history and save best model
            if epoch_loss['val'] < best_model_info["min_val_loss"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["min_val_loss"] = epoch_loss['val'].item()

        return best_model, best_model_info


    def test(self, test_data: np.ndarray = None, args: argparse.Namespace = None):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        with torch.no_grad():
            batch_data_generator = self.yield_batch_data(graph_adj_arr=test_data, batch_size=args.batch_size)
            num_batchs = ceil(len(test_data)//args.batch_size)
            for batched_data in batch_data_generator:
                x, y = batched_data[0], batched_data[1]
                torch.autograd.set_detect_anomaly(True)
                pred = self.forward(x)
                pred, y = pred.reshape(1, -1), y.reshape(1, -1)
                loss = self.loss_fn(pred, y)
                edge_acc = np.isclose(pred.cpu().detach().numpy(), y.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                test_edge_acc += edge_acc / num_batchs
                test_loss += loss / num_batchs

        return test_loss, test_edge_acc

    @staticmethod
    def save_model(unsaved_model: OrderedDict, model_info: dict, model_dir: Path, model_log_dir: Path):
        e_i = model_info.get("best_val_epoch")
        t_stamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        torch.save(unsaved_model, model_dir/f"epoch_{e_i}-{t_stamp}.pt")
        with open(model_log_dir/f"epoch_{e_i}-{t_stamp}.json", "w") as f:
            json_str = json.dumps(model_info)
            f.write(json_str)
        logger.info(f"model has been saved in:{model_dir}")

    @staticmethod
    def yield_batch_data(graph_adj_arr: np.ndarray, batch_size:int =11):
        graph_time_step = graph_adj_arr.shape[0] - 1  # the graph of last "t" can't be used as train data
        _, num_nodes, _ = graph_adj_arr.shape
        graph_adj_arr = graph_adj_arr.reshape(graph_time_step+1, -1)
        for g_t in range(0, graph_time_step, batch_size):
            begin_t, end_t = g_t, g_t+batch_size
            if end_t > graph_time_step+1: break
            x = torch.tensor(graph_adj_arr[begin_t:end_t]).float()
            y = torch.tensor(graph_adj_arr[end_t]).float()

            yield x, y


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
    ARGS = baseline_args_parser.parse_args()
    logger.debug(pformat(data_cfg, indent=1, width=100, compact=True))
    logger.info(pformat(f"\n{vars(ARGS)}", indent=1, width=40, compact=True))

    # ## Data implement & output setting & testset setting
    # data implement setting
    data_implement = "SP500_20082017_CORR_SER_REG_CORR_MAT_HRCHY_11_CLUSTER"  # watch options by operate: logger.info(data_cfg["DATASETS"].keys())
    #data_implement = "ARTIF_PARTICLE"  # watch options by operate: logger.info(data_cfg["DATASETS"].keys())
    # train set setting
    train_items_setting = "-train_train"  # -train_train|-train_all
    # setting of name of output files and pictures title
    output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
    # setting of output files
    save_model_info = ARGS.save_model
    # set devide of pytorch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    logger.info(f"===== file_name basis:{output_file_name} =====")
    logger.info(f"===== pytorch running on:{device} =====")

    s_l, w_l = ARGS.corr_stride, ARGS.corr_window
    graph_adj_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/filtered_graph_adj_mat/{ARGS.filt_mode}-quan{str(ARGS.filt_quan).replace('.', '')}" if ARGS.filt_mode else Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_adj_mat"
    graph_node_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_node_mat"
    g_model_dir = current_dir / f'save_models/baseline_gru/{output_file_name}/corr_s{s_l}_w{w_l}'
    g_model_log_dir = current_dir / f'save_models/baseline_gru/{output_file_name}/corr_s{s_l}_w{w_l}/train_logs/'
    g_model_dir.mkdir(parents=True, exist_ok=True)
    g_model_log_dir.mkdir(parents=True, exist_ok=True)

    gra_edges_data_mats = np.load(graph_adj_mat_dir / f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    gra_nodes_data_mats = np.load(graph_node_mat_dir / f"{ARGS.graph_nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy") if ARGS.graph_nodes_v_mode else np.ones((gra_edges_data_mats.shape[0], 1, gra_edges_data_mats.shape[2]))
    norm_train_dataset, norm_val_dataset, norm_test_dataset, scaler = split_and_norm_data(gra_edges_data_mats, gra_nodes_data_mats)
    mts_corr_ad_cfg = {"drop_p": ARGS.drop_p,
                       "dim_in": (norm_train_dataset['edges'].shape[1])**2,
                       "gru_l": ARGS.gru_l,
                       "gru_h": ARGS.gru_h,
                       "dim_out": (norm_train_dataset['edges'].shape[1])**2}
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

    model = BaselineGRUModel(**mts_corr_ad_cfg)
    best_model, best_model_info = model.train(train_data=norm_train_dataset['edges'], val_data=norm_val_dataset['edges'], epochs=ARGS.tr_epochs, args=ARGS)
    if save_model_info:
        model.save_model(best_model, best_model_info, model_dir=g_model_dir, model_log_dir=g_model_log_dir)
