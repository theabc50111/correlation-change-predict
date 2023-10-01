#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import functools
import json
import logging
import os
import sys
import traceback
import warnings
from collections import OrderedDict
from datetime import datetime
from itertools import product
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import matplotlib as mpl
import numpy as np
import torch
import yaml
from torch.nn import GRU, MSELoss
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR, SequentialLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import summary
from torch_geometric.utils import unbatch, unbatch_edge_index
from tqdm import tqdm

sys.path.append("/workspace/correlation-change-predict/utils")
from metrics_utils import EdgeAccuracyLoss
from utils import split_and_norm_data

from encoder_decoder import (GineEncoder, GinEncoder, MLPDecoder,
                             ModifiedInnerProductDecoder)
from mts_corr_ad_model import GraphTimeSeriesDataset, MTSCorrAD

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
matplotlib_logger = logging.getLogger("matplotlib")
logger.setLevel(logging.INFO)
metrics_logger.setLevel(logging.INFO)
utils_logger.setLevel(logging.INFO)
matplotlib_logger.setLevel(logging.ERROR)
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.simplefilter("ignore")


class MTSCorrAD2(MTSCorrAD):
    """
    Multi-Time Series Correlation Anomaly Detection2 (MTSCorrAD2)
    Structure of MTSCorrAD3:
        GRU_x ------↘
                      -> GraphEncoder -> GRU -> Decoder
        GRU_edges --↗
    Note: the nodes values of input of GraphEncoder is produced by first GRU
    """
    def __init__(self, model_cfg: dict):
        super(MTSCorrAD2, self).__init__(model_cfg)
        self.model_cfg = model_cfg
        # create data loader
        self.num_tr_batches = self.model_cfg["num_batches"]['train']
        self.num_val_batches = self.model_cfg["num_batches"]['val']

        # set model components
        self.graph_encoder = self.model_cfg['graph_encoder'](**self.model_cfg)
        graph_enc_emb_size = self.graph_encoder.gra_enc_l * self.graph_encoder.gra_enc_h  # the input size of GRU depend on the number of layers of GINconv
        num_nodes = self.model_cfg['num_nodes']
        num_edges = num_nodes**2
        num_node_features = self.model_cfg['num_node_features']
        num_edge_features = self.model_cfg['num_edge_features']
        self.gru1_x = GRU(num_nodes*num_node_features, num_nodes*num_node_features, self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0)
        self.gru1_edges = GRU(num_edges*num_edge_features, num_edges*num_edge_features, self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0)
        self.gru2 = GRU(graph_enc_emb_size, self.model_cfg["gru_h"], self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0)
        self.decoder = self.model_cfg['decoder'](self.model_cfg['gru_h'], num_nodes, drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        if self.model_cfg["pretrain_encoder"]:
            self.graph_encoder.load_state_dict(torch.load(self.model_cfg["pretrain_encoder"]))
        if self.model_cfg["pretrain_decoder"]:
            self.decoder.load_state_dict(torch.load(self.model_cfg["pretrain_decoder"]))
        self.init_optimizer()

    def forward(self, x, edge_index, seq_batch_node_id, edge_attr, *unused_args):
        """
        Operate when model called
        """

        x_edge_index_list = unbatch_edge_index(edge_index, seq_batch_node_id)
        num_nodes = self.model_cfg['num_nodes']
        seq_len = len(x_edge_index_list)
        batch_edge_attr_start_idx = 0
        seq_edge_attr = torch.zeros((seq_len, num_nodes*num_nodes))
        seq_batch_strong_connect_edge_index = torch.zeros((2, seq_len*num_nodes**2), dtype=torch.int64)
        for seq_t in range(seq_len):
            batch_edge_attr_end_idx = x_edge_index_list[seq_t].shape[1] + batch_edge_attr_start_idx
            x_edge_index = x_edge_index_list[seq_t]
            x_edge_attr = edge_attr[batch_edge_attr_start_idx: batch_edge_attr_end_idx]
            x_graph_adj = torch.sparse_coo_tensor(x_edge_index, x_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
            seq_edge_attr[seq_t] = x_graph_adj.reshape(-1)
            batch_edge_attr_start_idx = batch_edge_attr_end_idx
            seq_batch_strong_connect_edge_index[::, seq_t*num_nodes**2:(seq_t+1)*num_nodes**2] = torch.tensor(list(product(range(seq_t*num_nodes, (seq_t+1)*num_nodes), repeat=2)), dtype=torch.int64).t().contiguous()

        # Temporal Modeling
        gru_edge_attr, _ = self.gru1_edges(seq_edge_attr)
        temporal_edge_attr = gru_edge_attr.reshape(-1, self.model_cfg['num_edge_features'])
        gru_x, _ = self.gru1_x(x.reshape(seq_len, -1))
        temporal_x = gru_x.reshape(-1, self.model_cfg['num_node_features'])

        # Inter-series modeling
        if type(self.graph_encoder).__name__ == "GinEncoder":
            graph_embeds = self.graph_encoder(temporal_x, seq_batch_strong_connect_edge_index, seq_batch_node_id)
        elif type(self.graph_encoder).__name__ == "GineEncoder":
            graph_embeds = self.graph_encoder(temporal_x, seq_batch_strong_connect_edge_index, seq_batch_node_id, temporal_edge_attr)

        # second Temporal Modeling
        gru_output, _ = self.gru2(graph_embeds)

        # Decoder (Graph Adjacency Reconstruction)
        pred_graph_adj = self.decoder(gru_output[-1])  # gru_output[-1] => only take last time-step

        return pred_graph_adj

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False):
        """
        Training MTSCorrAD2 Model
        """
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self

        self.show_model_config()
        best_model_info = self.init_best_model_info(train_data, loss_fns, epochs)
        best_model = []
        num_nodes = self.model_cfg['num_nodes']
        train_loader = self.create_pyg_data_loaders(graph_adj_mats=train_data['edges'],  graph_nodes_mats=train_data["nodes"], loader_seq_len=self.model_cfg["seq_len"])
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "gra_enc_weight_l2_reg": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1),
                             "gra_enc_grad": torch.zeros(1), "gru_grad": torch.zeros(1), "gra_dec_grad": torch.zeros(1), "lr": torch.zeros(1),
                             "pred_gra_embeds": [], "y_gra_embeds": [], "gra_embeds_disparity": {}}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            for batch_idx, batch_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_loss = torch.zeros(1)
                batch_edge_acc = torch.zeros(1)
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                    pred_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    edge_acc = np.isclose(pred_graph_adj.cpu().detach().numpy(), y_graph_adj.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                    batch_edge_acc += edge_acc/(self.num_tr_batches*self.model_cfg['batch_size'])
                    loss_fns["fn_args"]["MSELoss()"].update({"input": pred_graph_adj, "target":  y_graph_adj})
                    loss_fns["fn_args"]["EdgeAccuracyLoss()"].update({"input": pred_graph_adj, "target":  y_graph_adj})
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        batch_loss += loss/(self.num_tr_batches*self.model_cfg['batch_size'])
                        epoch_metrics[fn_name] += loss/(self.num_tr_batches*self.model_cfg['batch_size'])

                if self.model_cfg['graph_enc_weight_l2_reg_lambda']:
                    gra_enc_weight_l2_penalty = self.model_cfg['graph_enc_weight_l2_reg_lambda']*sum(p.pow(2).mean() for p in self.graph_encoder.parameters())
                    batch_loss += gra_enc_weight_l2_penalty
                else:
                    gra_enc_weight_l2_penalty = 0
                batch_loss.backward()
                self.optimizer.step()
                if hasattr(self, 'scheduler'):
                    self.scheduler.step()
                # compute graph embeds
                pred_graph_embeds = self.get_pred_embeddings(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                y_graph_embeds = self.graph_encoder.get_embeddings(y, y_edge_index, y_seq_batch_node_id, y_edge_attr)
                # record metrics for each batch
                epoch_metrics["tr_loss"] += batch_loss
                epoch_metrics["tr_edge_acc"] += batch_edge_acc
                epoch_metrics["gra_enc_weight_l2_reg"] += gra_enc_weight_l2_penalty
                epoch_metrics["gra_enc_grad"] += sum(p.grad.sum() for p in self.graph_encoder.parameters() if p.grad is not None)/self.num_tr_batches
                epoch_metrics["gru_grad"] += sum(p.grad.sum() for p in self.gru2.parameters() if p.grad is not None)/self.num_tr_batches
                epoch_metrics["gra_dec_grad"] += sum(p.grad.sum() for p in self.decoder.parameters() if p.grad is not None)/self.num_tr_batches
                epoch_metrics["lr"] = torch.tensor(self.optimizer.param_groups[0]['lr'])
                epoch_metrics["pred_gra_embeds"].append(pred_graph_embeds.tolist())
                epoch_metrics["y_gra_embeds"].append(y_graph_embeds.tolist())
                # used in observation model info in console
                log_model_info_data = data
                log_model_info_batch_idx = batch_idx

            # Validation
            epoch_metrics['val_loss'], epoch_metrics['val_edge_acc'] = self.test(val_data, loss_fns=loss_fns, show_loader_log=True if epoch_i == 0 else False)

            # record training history and save best model
            for k, v in epoch_metrics.items():
                history_list = best_model_info.setdefault(k+"_history", [])
                history_list.append(v.item() if isinstance(v, torch.Tensor) else v)
            if epoch_metrics['val_loss'] < best_model_info["min_val_loss"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["min_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["min_val_loss_edge_acc"] = epoch_metrics['val_edge_acc'].item()

            # Check if graph_encoder.parameters() have been updated
            assert sum(map(abs, best_model_info['gra_enc_grad_history'])) > 0, f"Sum of gradient of MTSCorrAD.graph_encoder in epoch_{epoch_i}:{sum(map(abs, best_model_info['gra_enc_grad_history']))}"
            # observe model info in console
            if epoch_i == 0:
                best_model_info["model_structure"] = str(self) + "\n" + "="*100 + "\n" + str(summary(self, log_model_info_data.x, log_model_info_data.edge_index, log_model_info_data.batch, log_model_info_data.edge_attr, max_depth=20))
                if show_model_info:
                    logger.info(f"\nNumber of graphs:{log_model_info_data.num_graphs} in No.{log_model_info_batch_idx} batch, the model structure:\n{best_model_info['model_structure']}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.9f}" for k, v in epoch_metrics.items() if "embeds" not in k])
                logger.info(f"In Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | lr: {self.optimizer.param_groups[0]['lr']:.9f}")
            if epoch_i % 500 == 0:  # show oredictive and real adjacency matrix every 500 epochs
                logger.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:{data_batch_idx} \npred_graph_adj:\n{pred_graph_adj}\ny_graph_adj:\n{y_graph_adj}\n")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None, show_loader_log: bool = False):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        num_nodes = self.model_cfg['num_nodes']
        test_loader = self.create_pyg_data_loaders(graph_adj_mats=test_data["edges"],  graph_nodes_mats=test_data["nodes"], loader_seq_len=self.model_cfg["seq_len"], show_log=show_loader_log)
        with torch.no_grad():
            for batch_data in test_loader:
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                    pred_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    edge_acc = np.isclose(pred_graph_adj.cpu().detach().numpy(), y_graph_adj.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                    test_edge_acc += edge_acc/(self.num_val_batches*self.model_cfg['batch_size'])
                    loss_fns["fn_args"]["MSELoss()"].update({"input": pred_graph_adj, "target":  y_graph_adj})
                    loss_fns["fn_args"]["EdgeAccuracyLoss()"].update({"input": pred_graph_adj, "target":  y_graph_adj})
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        test_loss += loss/(self.num_val_batches*self.model_cfg['batch_size'])

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

    def get_pred_embeddings(self, x, edge_index, seq_batch_node_id, edge_attr, *unused_args):
        """
        get  the predictive graph_embeddings with no_grad by using part of self.forward() process
        """
        with torch.no_grad():
            x_edge_index_list = unbatch_edge_index(edge_index, seq_batch_node_id)
            num_nodes = self.model_cfg['num_nodes']
            seq_len = len(x_edge_index_list)
            batch_edge_attr_start_idx = 0
            seq_edge_attr = torch.zeros((seq_len, num_nodes*num_nodes))
            seq_batch_strong_connect_edge_index = torch.zeros((2, seq_len*num_nodes**2), dtype=torch.int64)
            for seq_t in range(seq_len):
                batch_edge_attr_end_idx = x_edge_index_list[seq_t].shape[1] + batch_edge_attr_start_idx
                x_edge_index = x_edge_index_list[seq_t]
                x_edge_attr = edge_attr[batch_edge_attr_start_idx: batch_edge_attr_end_idx]
                x_graph_adj = torch.sparse_coo_tensor(x_edge_index, x_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                seq_edge_attr[seq_t] = x_graph_adj.reshape(-1)
                batch_edge_attr_start_idx = batch_edge_attr_end_idx
                seq_batch_strong_connect_edge_index[::, seq_t*num_nodes**2:(seq_t+1)*num_nodes**2] = torch.tensor(list(product(range(seq_t*num_nodes, (seq_t+1)*num_nodes), repeat=2)), dtype=torch.int64).t().contiguous()

            # Temporal Modeling
            gru_edge_attr, _ = self.gru1_edges(seq_edge_attr)
            temporal_edge_attr = gru_edge_attr.reshape(-1, self.model_cfg['num_edge_features'])
            gru_x, _ = self.gru1_x(x.reshape(seq_len, -1))
            temporal_x = gru_x.reshape(-1, self.model_cfg['num_node_features'])

            # Inter-series modeling
            if type(self.graph_encoder).__name__ == "GinEncoder":
                graph_embeds = self.graph_encoder(temporal_x, seq_batch_strong_connect_edge_index, seq_batch_node_id)
            elif type(self.graph_encoder).__name__ == "GineEncoder":
                graph_embeds = self.graph_encoder(temporal_x, seq_batch_strong_connect_edge_index, seq_batch_node_id, temporal_edge_attr)

            # second Temporal Modeling
            pred_graph_embeds, _ = self.gru2(graph_embeds)


        return pred_graph_embeds[-1]

    def create_pyg_data_loaders(self, graph_adj_mats: np.ndarray, graph_nodes_mats: np.ndarray, loader_seq_len : int,  show_log: bool = True, show_debug_info: bool = False):
        """
        Create Pytorch Geometric DataLoaders
        """
        # Create an instance of the GraphTimeSeriesDataset
        dataset = GraphTimeSeriesDataset(graph_adj_mats,  graph_nodes_mats, model_cfg=self.model_cfg, show_log=show_log)
        # Create mini-batches
        data_loader = DataLoader(dataset, batch_size=loader_seq_len, shuffle=False)

        if show_log:
            logger.info(f'Number of batches in this data_loader: {len(data_loader)}')
            logger.info("="*30)

        if show_debug_info:
            logger.debug('Peeking info of subgraph:')
            for batch_idx, subgraph in enumerate(data_loader):
                if 2 < batch_idx < (len(data_loader)-2):  # only peek the first 2 and last 2 batches
                    continue
                for data_batch_idx in range(len(subgraph)):
                    if 1 < data_batch_idx < (self.model_cfg['batch_size'] - 1):  # only peek the first and last data instances in the batch_data
                        continue
                    logger.debug(f' - Subgraph of data_batch_idx-{data_batch_idx} in batch-{batch_idx}: {subgraph[data_batch_idx]} ; num_graphs:{subgraph[data_batch_idx].num_graphs} ; edge_index[::, 10:15]: {subgraph[data_batch_idx].edge_index[::, 10:15]}')

                    x_nodes_list = unbatch(subgraph[data_batch_idx].x, subgraph[data_batch_idx].batch)
                    x_edge_index_list = unbatch_edge_index(subgraph[data_batch_idx].edge_index, subgraph[data_batch_idx].batch)
                    num_nodes = self.model_cfg['num_nodes']
                    batch_edge_attr_start_idx = 0
                    for seq_t in range(loader_seq_len):
                        if 3 < seq_t < (loader_seq_len-2):  # only peek the first 3 and last 2 seq_t
                            continue
                        batch_edge_attr_end_idx = x_edge_index_list[seq_t].shape[1] + batch_edge_attr_start_idx
                        x_nodes = x_nodes_list[seq_t]
                        x_edge_index = x_edge_index_list[seq_t]
                        x_edge_attr = subgraph[data_batch_idx].edge_attr[batch_edge_attr_start_idx: batch_edge_attr_end_idx]
                        x_graph_adj = torch.sparse_coo_tensor(x_edge_index, x_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                        data_y = subgraph[data_batch_idx].y[seq_t]
                        y_nodes = data_y.x
                        y_edge_index = data_y.edge_index
                        y_edge_attr = data_y.edge_attr
                        y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                        batch_edge_attr_start_idx = batch_edge_attr_end_idx

                        logger.debug((f"\n---------------------------At batch{batch_idx} and data{data_batch_idx} and seq_t{seq_t}---------------------------\n"
                                      f"x.shape: {x_nodes.shape}, x_edge_attr.shape: {x_edge_attr.shape}, x_edge_idx.shape: {x_edge_index.shape}, x_graph_adj.shape:{x_graph_adj.shape}\n"
                                      f"x:\n{x_nodes}\n"
                                      f"x_edges_idx[:5]:\n{x_edge_index[::, :5]}\n"
                                      f"x_graph_adj:\n{x_graph_adj}\n"
                                      f"y.shape: {y_nodes.shape}, y_edge_attr.shape: {y_edge_attr.shape}, y_edge_idx.shape: {y_edge_index.shape}, y_graph_adj.shape:{y_graph_adj.shape}\n"
                                      f"y:\n{y_nodes}\n"
                                      f"y_edges_idx[:5]:\n{y_edge_index[::, :5]}\n"
                                      f"y_graph_adj:\n{y_graph_adj}\n"
                                      f"\n---------------------------At batch{batch_idx} and data{data_batch_idx} and seq_t{seq_t}---------------------------\n"))

        return data_loader


if __name__ == "__main__":
    mts_corr_ad_args_parser = argparse.ArgumentParser()
    mts_corr_ad_args_parser.add_argument("--data_implement", type=str, nargs='?', default="SP500_20082017_CORR_SER_REG_STD_CORR_MAT_HRCHY_10_CLUSTER_LABEL_HALF_MIX",
                                         help="input the data implement name, watch options by operate: logger.info(data_cfg['DATASETS'].keys())")
    mts_corr_ad_args_parser.add_argument("--batch_size", type=int, nargs='?', default=10,
                                         help="input the number of batch size")
    mts_corr_ad_args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=300,
                                         help="input the number of training epochs")
    mts_corr_ad_args_parser.add_argument("--seq_len", type=int, nargs='?', default=30,
                                         help="input the number of sequence length")
    mts_corr_ad_args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                         help="input --save_model to save model weight and model info")
    mts_corr_ad_args_parser.add_argument("--corr_type", type=str, nargs='?', default="pearson",
                                         choices=["pearson", "cross_corr"],
                                         help="input the type of correlation computing, the choices are [pearson, cross_corr]")
    mts_corr_ad_args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                                         help="input the number of stride length of correlation computing")
    mts_corr_ad_args_parser.add_argument("--corr_window", type=int, nargs='?', default=10,
                                         help="input the number of window length of correlation computing")
    mts_corr_ad_args_parser.add_argument("--filt_mode", type=str, nargs='?', default=None,
                                         help="input the filtered mode of graph edges, look up the options by execute python ywt_library/data_module.py -h")
    mts_corr_ad_args_parser.add_argument("--filt_quan", type=float, nargs='?', default=None,
                                         help="input the filtered quantile of graph edges")
    mts_corr_ad_args_parser.add_argument("--discrete_bin", type=int, nargs='?', default=None,
                                         help="input the number of discrete bins of graph edges")
    mts_corr_ad_args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default=None,
                                         help="Decide mode of nodes' vaules of graph_nodes_matrix, look up the options by execute python ywt_library/data_module.py -h")
    mts_corr_ad_args_parser.add_argument("--pretrain_encoder", type=str, nargs='?', default="",
                                         help="input the path of pretrain encoder weights")
    mts_corr_ad_args_parser.add_argument("--pretrain_decoder", type=str, nargs='?', default="",
                                         help="input the path of pretrain decoder weights")
    mts_corr_ad_args_parser.add_argument("--learning_rate", type=float, nargs='?', default=0.0001,
                                         help="input the learning rate of training")
    mts_corr_ad_args_parser.add_argument("--weight_decay", type=float, nargs='?', default=0.01,
                                         help="input the weight decay of training")
    mts_corr_ad_args_parser.add_argument("--graph_enc_weight_l2_reg_lambda", type=float, nargs='?', default=0,
                                         help="input the weight of graph encoder weight l2 norm loss")
    mts_corr_ad_args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                                         help="input [gru] | [gru decoder] | [decoder gru graph_encoder] to decide the position of drop layers")
    mts_corr_ad_args_parser.add_argument("--drop_p", type=float, default=0,
                                         help="input 0~1 to decide the probality of drop layers")
    mts_corr_ad_args_parser.add_argument("--gra_enc", type=str, nargs='?', default="gine",
                                         help="input the type of graph encoder")
    mts_corr_ad_args_parser.add_argument("--gra_enc_aggr", type=str, nargs='?', default="add",
                                         help="input the type of aggregator of graph encoder")
    mts_corr_ad_args_parser.add_argument("--gra_enc_l", type=int, nargs='?', default=1,  # range:1~n, for graph encoder after the second layer,
                                         help="input the number of graph laryers of graph_encoder")
    mts_corr_ad_args_parser.add_argument("--gra_enc_h", type=int, nargs='?', default=4,
                                         help="input the number of graph embedding hidden size of graph_encoder")
    mts_corr_ad_args_parser.add_argument("--gru_l", type=int, nargs='?', default=3,  # range:1~n, for gru
                                         help="input the number of stacked-layers of gru")
    mts_corr_ad_args_parser.add_argument("--gru_h", type=int, nargs='?', default=None,
                                         help="input the number of gru hidden size")
    ARGS = mts_corr_ad_args_parser.parse_args()
    assert bool(ARGS.drop_pos) == bool(ARGS.drop_p), "drop_pos and drop_p must be both input or not input"
    assert bool(ARGS.filt_mode) == bool(ARGS.filt_quan), "filt_mode and filt_quan must be both input or not input"
    assert (bool(ARGS.filt_mode) != bool(ARGS.discrete_bin)) or (ARGS.filt_mode is None and ARGS.discrete_bin is None), "filt_mode and discrete_bin must be both not input or one input"
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor)
    torch.autograd.set_detect_anomaly(True)  # for debug grad
    logger.info(f"===== file_name basis:{output_file_name} =====")
    logger.info(f"===== pytorch running on:{device} =====")

    s_l, w_l = ARGS.corr_stride, ARGS.corr_window
    if ARGS.filt_mode:
        graph_adj_mode_dir = f"filtered_graph_adj_mat/{ARGS.filt_mode}-quan{str(ARGS.filt_quan).replace('.', '')}"
    elif ARGS.discrete_bin:
        graph_adj_mode_dir = f"discretize_graph_adj_mat/discrete_bin{ARGS.discrete_bin}"
    else:
        graph_adj_mode_dir = "graph_adj_mat"
    graph_adj_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.corr_type}/{graph_adj_mode_dir}"
    graph_node_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/graph_node_mat"
    g_model_dir = current_dir / f'save_models/mts_corr_ad_model_2/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    g_model_log_dir = current_dir / f'save_models/mts_corr_ad_model_2/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    g_model_dir.mkdir(parents=True, exist_ok=True)
    g_model_log_dir.mkdir(parents=True, exist_ok=True)

    # model configuration
    is_training, train_count = True, 0
    gra_edges_data_mats = np.load(graph_adj_mat_dir / f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    gra_nodes_data_mats = np.load(graph_node_mat_dir / f"{ARGS.graph_nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy") if ARGS.graph_nodes_v_mode else np.ones((gra_edges_data_mats.shape[0], 1, gra_edges_data_mats.shape[2]))
    norm_train_dataset, norm_val_dataset, norm_test_dataset, scaler = split_and_norm_data(gra_edges_data_mats, gra_nodes_data_mats)
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

    mts_corr_ad_cfg = {"filt_mode": ARGS.filt_mode,
                       "filt_quan": ARGS.filt_quan,
                       "discrete_bin": ARGS.discrete_bin,
                       "graph_nodes_v_mode": ARGS.graph_nodes_v_mode,
                       "tr_epochs": ARGS.tr_epochs,
                       "batch_size": ARGS.batch_size,
                       "seq_len": ARGS.seq_len,
                       "num_batches": {"train": ((len(norm_train_dataset["edges"])-1)//ARGS.batch_size),
                                       "val": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size),
                                       "test": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size)},
                       "pretrain_encoder": ARGS.pretrain_encoder,
                       "pretrain_decoder": ARGS.pretrain_decoder,
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
                       "decoder": MLPDecoder}

    model = MTSCorrAD2(mts_corr_ad_cfg)
    loss_fns_dict = {"fns": [MSELoss(), EdgeAccuracyLoss()],
                     "fn_args": {"MSELoss()": {}, "EdgeAccuracyLoss()": {}}}
    while (is_training is True) and (train_count < 100):
        try:
            train_count += 1
            g_best_model, g_best_model_info = model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
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
                model.save_model(g_best_model, g_best_model_info, model_dir=g_model_dir, model_log_dir=g_model_log_dir)
