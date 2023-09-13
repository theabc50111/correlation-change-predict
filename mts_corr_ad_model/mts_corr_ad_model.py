#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import functools
import json
import logging
import sys
import traceback
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import overload

import dynamic_yaml
import matplotlib as mpl
import numpy as np
import torch
import yaml
from torch.nn import GRU
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR, SequentialLR
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import summary
from torch_geometric.utils import unbatch, unbatch_edge_index
from tqdm import tqdm

sys.path.append("/workspace/correlation-change-predict/utils")
from encoder_decoder import (GineEncoder, GinEncoder, MLPDecoder,
                             ModifiedInnerProductDecoder)

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


class GraphTimeSeriesDataset(Dataset):
    def __init__(self, graph_adj_mats: np.ndarray, graph_nodes_mats: np.ndarray, target_mats: np.ndarray, model_cfg: dict, show_log: bool = True):
        """
        Create list of graph data:
        In order to make the `DataLoader` combine sequencial graphs into a `Data` object, the arrange of self.data_list will not be sequencial.
        For example, in the case of batch_size=3 and seq_len=5, the arrange of self.data_list is:
            |<----batch_size---->|
            |                    |
        [[graph_t0, graph_t1, graph_t2],  ----------
         [graph_t1, graph_t2, graph_t3],      ↓
         [graph_t2, graph_t3, graph_t4],   seq_len
         [graph_t3, graph_t4, graph_t5],      ↑
         [graph_t4, graph_t5, graph_t6],  ----------
         [graph_t3, graph_t4, graph_t5],  ----------
         [graph_t4, graph_t5, graph_t6],      ↓
         [graph_t5, graph_t6, graph_t7],   seq_len
         [graph_t6, graph_t7, graph_t8],      ↑
         [graph_t7, graph_t8, graph_t9],  ----------
                        .
                        .
                        .
         [graph_tn-2, graph_tn-1, graph_tn]]
        """
        self.batch_size = model_cfg['batch_size']
        self.seq_len = model_cfg['seq_len']
        graph_nodes_mats = graph_nodes_mats.transpose(0, 2, 1)
        graph_time_len = graph_adj_mats.shape[0] - 1  # the graph of last "t" can't be used as train data
        y_graph_adj_mats = target_mats
        final_batch_head = ((graph_time_len//self.batch_size)-1)*self.batch_size
        last_seq_len = graph_time_len-final_batch_head-self.batch_size
        data_list = []
        for batch_head in range(0, final_batch_head+1, self.batch_size):
            seq_data_list = []
            for seq_t in range(self.seq_len):
                if batch_head == final_batch_head and seq_t > last_seq_len:
                    break
                batch_data_list = []
                for data_batch_idx in range(self.batch_size):
                    g_t = batch_head + seq_t + data_batch_idx
                    edge_index_next_t = torch.tensor(np.stack(np.where(~np.isnan(y_graph_adj_mats[g_t + 1])), axis=1))
                    edge_attr_next_t = torch.tensor(y_graph_adj_mats[g_t + 1][~np.isnan(y_graph_adj_mats[g_t + 1])].reshape(-1, 1), dtype=torch.float64)
                    node_attr_next_t = torch.tensor(graph_nodes_mats[g_t + 1], dtype=torch.float64)
                    data_y = Data(x=node_attr_next_t, edge_index=edge_index_next_t.t().contiguous(), edge_attr=edge_attr_next_t)
                    edge_index = torch.tensor(np.stack(np.where(~np.isnan(graph_adj_mats[g_t])), axis=1))
                    edge_attr = torch.tensor(graph_adj_mats[g_t][~np.isnan(graph_adj_mats[g_t])].reshape(-1, 1), dtype=torch.float64)
                    node_attr = torch.tensor(graph_nodes_mats[g_t], dtype=torch.float64)
                    data = Data(x=node_attr, y=data_y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
                    batch_data_list.append(data)
                seq_data_list.append(batch_data_list)
            data_list.extend(seq_data_list)
        self.data_list = data_list

        if show_log:
            logger.info(f"Time length of graphs: {graph_time_len}")
            logger.info(f"The length of Dataset is: {len(data_list)}")
            logger.info(f"data.num_node_features: {data.num_node_features}; data.num_edges: {data.num_edges}; data.num_edge_features: {data.num_edge_features}; data.is_undirected: {data.is_undirected()}")
            logger.info(f"data.x.shape: {data.x.shape}; data.y.x.shape: {data.y.x.shape}; data.edge_index.shape: {data.edge_index.shape}; data.edge_attr.shape: {data.edge_attr.shape}")

    def __len__(self):
        """
        Return the length of dataset
        Computation of length of dataset:
        graph_time_len = graph_adj_mats.shape[0] - 1  # the graph of last "t" can't be used as train data
        len(self.data_list) = (graph_time_len//self.model_cfg['batch_size']) * self.model_cfg['seq_len']
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Return a batch_data. For example, when idx is "0", `__getitem__()` will return a list that contains batch_size graphs:
        | __get_item__ idx |         batch_size         |
             idx=0 ----->  [graph_t0, graph_t1, graph_t2]

        In order to make the `DataLoader` combine sequencial graphs into a `Data` objects, the arrange of self.data_list will not be sequencial.
        !!!So the return of `__getitem__()` will not be sequencial either!!!
        For example, when batch_size is "3" and seq_len is 5, `__getitem__()` will return:
        | __get_item__ idx |         batch_size         |
        |                  |                            |
             idx=0 ----->  [graph_t0, graph_t1, graph_t2]----------
             idx=1 ----->  [graph_t1, graph_t2, graph_t3]    ↓
             idx=2 ----->  [graph_t2, graph_t3, graph_t4] seq_len
             idx=3 ----->  [graph_t3, graph_t4, graph_t5]    ↑
             idx=4 ----->  [graph_t4, graph_t5, graph_t6]----------
             idx=5 ----->  [graph_t3, graph_t4, graph_t5]----------
             idx=6 ----->  [graph_t4, graph_t5, graph_t6]    ↓
             idx=7 ----->  [graph_t5, graph_t6, graph_t7] seq_len
             idx=8 ----->  [graph_t6, graph_t7, graph_t8]    ↑
             idx=9 ----->  [graph_t7, graph_t8, graph_t9]----------
            idx=10 ----->  [graph_t6, graph_t7, graph_t8]------------
            idx=11 ----->  [graph_t7, graph_t8, graph_t9]      ↓
            idx=12 ----->  [graph_t8, graph_t9, graph_t10]  seq_len
            idx=13 ----->  [graph_t9, graph_t10, graph_t11]    ↑
            idx=14 ----->  [graph_t10, graph_t11, graph_t12]---------
        """
        batch_data = self.data_list[idx]

        return batch_data

    def len(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Implement this to match abstract base class.
        """
        return self.__len__()

    def get(self, idx: int) -> list:
        r"""
        Gets the data object at index :obj:`idx`.
        Implement this to match abstract base class.
        """
        return self.__getitem__(idx)


class MTSCorrAD(torch.nn.Module):
    """
    Multi-Time Series Correlation Anomaly Detection (MTSCorrAD)
    """
    def __init__(self, model_cfg: dict):
        super(MTSCorrAD, self).__init__()
        self.model_cfg = model_cfg
        # create data loader
        self.num_tr_batches = max(1, self.model_cfg["num_batches"]['train']-1)  # the number of batches in training data loader, -1 because the last batch is not full
        self.num_val_batches = max(1, self.model_cfg["num_batches"]['val']-1)
        self.num_test_batches = max(1, self.model_cfg["num_batches"]['test']-1)

        # set model components
        self.graph_encoder = self.model_cfg['graph_encoder'](**self.model_cfg)
        graph_enc_emb_size = self.graph_encoder.gra_enc_l * self.graph_encoder.gra_enc_h  # the input size of GRU depend on the number of layers of GINconv
        self.gru1 = GRU(graph_enc_emb_size, self.model_cfg["gru_h"], self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0)
        self.decoder = self.model_cfg['decoder'](self.model_cfg['gru_h'], self.model_cfg["num_nodes"], drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        if self.model_cfg["pretrain_encoder"]:
            self.graph_encoder.load_state_dict(torch.load(self.model_cfg["pretrain_encoder"]))
        if self.model_cfg["pretrain_decoder"]:
            self.decoder.load_state_dict(torch.load(self.model_cfg["pretrain_decoder"]))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.model_cfg['learning_rate'], weight_decay=self.model_cfg['weight_decay'])
        schedulers = [ConstantLR(self.optimizer, factor=0.1, total_iters=self.num_tr_batches*6), MultiStepLR(self.optimizer, milestones=list(range(self.num_tr_batches*5, self.num_tr_batches*600, self.num_tr_batches*50))+list(range(self.num_tr_batches*600, self.num_tr_batches*self.model_cfg['tr_epochs'], self.num_tr_batches*100)), gamma=0.9)]
        self.scheduler = SequentialLR(self.optimizer, schedulers=schedulers, milestones=[self.num_tr_batches*6])

    def show_model_config(self):
        """
        Show model information
        """
        observe_model_cfg = {item[0]: item[1] for item in self.model_cfg.items() if item[0] != 'dataset'}
        observe_model_cfg['optimizer'] = str(self.optimizer)
        if hasattr(self.scheduler, '_milestones'):
            observe_model_cfg['scheduler'] = {"scheduler_name": str(self.scheduler.__class__.__name__), "milestones": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones), "gamma": self.scheduler._schedulers[1].gamma}
        else:
            observe_model_cfg['scheduler'] = {"scheduler_name": str(self.scheduler.__class__.__name__)}

        logger.info(f"\nModel Configuration: \n{observe_model_cfg}")
        logger.info("="*80)

    def forward(self, x, edge_index, seq_batch_node_id, edge_attr, output_type, *unused_args):
        """
        Operate when model called
        """
        # Inter-series modeling
        if type(self.graph_encoder).__name__ == "GinEncoder":
            graph_embeds = self.graph_encoder(x, edge_index, seq_batch_node_id)
        elif type(self.graph_encoder).__name__ == "GineEncoder":
            graph_embeds = self.graph_encoder(x, edge_index, seq_batch_node_id, edge_attr)

        # Temporal Modeling
        gru_output, _ = self.gru1(graph_embeds)

        # Decoder (Graph Adjacency Reconstruction)
        pred_graph_adj = self.decoder(gru_output[-1])  # gru_output[-1] => only take last time-step

        if output_type == "discretize":
            bins = torch.tensor(self.model_cfg['output_bins']).reshape(-1, 1)
            num_bins = len(bins)-1
            bins = torch.concat((bins[:-1], bins[1:]), dim=1)
            discretize_values = np.linspace(-1, 1, num_bins)
            for lower, upper, discretize_value in zip(bins[:, 0], bins[:, 1], discretize_values):
                pred_graph_adj = torch.where((pred_graph_adj <= upper) & (pred_graph_adj > lower), discretize_value, pred_graph_adj)
            pred_graph_adj = torch.where(pred_graph_adj < bins.min(), bins.min(), pred_graph_adj)

        return pred_graph_adj

    def init_best_model_info(self, train_data: dict, loss_fns: dict, epochs: int):
        """
        Initialize best_model_info
        """
        best_model_info = {"num_training_graphs": len(train_data['edges']),
                           "filt_mode": self.model_cfg['filt_mode'],
                           "filt_quan": self.model_cfg['filt_quan'],
                           "quan_discrete_bins": self.model_cfg['quan_discrete_bins'],
                           "custom_discrete_bins": self.model_cfg['custom_discrete_bins'],
                           "graph_nodes_v_mode": self.model_cfg['graph_nodes_v_mode'],
                           "batches_per_epoch": self.num_tr_batches,
                           "epochs": epochs,
                           "batch_size": self.model_cfg['batch_size'],
                           "seq_len": self.model_cfg['seq_len'],
                           "opt_lr": self.model_cfg['learning_rate'],
                           "opt_weight_decay": self.model_cfg['weight_decay'],
                           "optimizer": str(self.optimizer),
                           "opt_scheduler": {"gamma": self.scheduler._schedulers[1].gamma, "milestoines": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones)} if hasattr(self.scheduler, '_milestones') else str(self.scheduler.__class__.__name__),
                           "gru_l": self.model_cfg['gru_l'],
                           "gru_h": self.model_cfg['gru_h'],
                           "gra_enc_l": self.model_cfg['gra_enc_l'],
                           "gra_enc_h": self.model_cfg['gra_enc_h'],
                           "gra_enc_mlp_l": self.model_cfg['gra_enc_mlp_l'],
                           "decoder": self.model_cfg['decoder'].__name__,
                           "loss_fns": [fn.__name__ if hasattr(fn, '__name__') else str(fn) for fn in loss_fns["fns"]],
                           "loss_weight": [{fn.__name__ if hasattr(fn, '__name__') else str(fn): str(getattr(fn, "weight", None))} for fn in loss_fns["fns"]],
                           "gra_enc_weight_l2_reg_lambda": self.model_cfg['graph_enc_weight_l2_reg_lambda'],
                           "drop_pos": self.model_cfg["drop_pos"],
                           "drop_p": self.model_cfg["drop_p"],
                           "graph_enc": type(self.graph_encoder).__name__ if hasattr(self, 'graph_encoder') else None,
                           "gra_enc_aggr": self.model_cfg['gra_enc_aggr'],
                           "min_val_loss": float('inf'),
                           "output_type": self.model_cfg['output_type'],
                           "output_bins": '_'.join((str(f) for f in self.model_cfg['output_bins'])).replace('.', '') if self.model_cfg['output_bins'] else None,
                           "target_mats_bins": self.model_cfg['target_mats_bins'],
                           "edge_acc_loss_atol": self.model_cfg['edge_acc_loss_atol'],
                           "two_ord_pred_prob_edge_accu_thres": self.model_cfg['two_ord_pred_prob_edge_accu_thres'],
                           "use_bin_edge_acc_loss": self.model_cfg['use_bin_edge_acc_loss']}

        return best_model_info

    def calc_loss_fn(self, loss_fns: dict, loss_fn_input: torch.Tensor, loss_fn_target: torch.Tensor, num_batches: int,
                     preds: torch.Tensor = None, y_labels: torch.Tensor = None, epoch_metrics: dict = None):
    ###def calc_loss_fn(self, loss_fns: dict, loss_fn_input: torch.Tensor, loss_fn_target: torch.Tensor, batch_loss: torch.Tensor,
    ###                 batch_edge_acc: torch.Tensor, preds: torch.Tensor = None, y_labels: torch.Tensor = None,
    ###                 epoch_metrics: dict = None):
        """
        Calculate loss function
        """
        has_calc_edge_acc = False
        batch_loss = torch.zeros(1)
        batch_edge_acc = torch.zeros(1)
        for fn in loss_fns["fns"]:
            fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
            loss_fns["fn_args"][fn_name].update({"input": loss_fn_input, "target": loss_fn_target})
            partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
            loss = partial_fn()
            batch_loss += loss
            if epoch_metrics is not None:
                epoch_metrics[fn_name] += loss/num_batches  # we don't reset epoch[fn_name] to 0 in each batch, so we need to divide by total number of batches
            if has_calc_edge_acc:
                continue
            if "EdgeAcc" in fn_name:
                edge_acc = 1-loss
            elif self.model_cfg.get("edge_acc_metric_fn"):
                edge_acc = self.model_cfg.get("edge_acc_metric_fn")(loss_fn_input, loss_fn_target)
            elif preds is not None and y_labels is not None:
                edge_acc = (preds == y_labels).to(torch.float).mean()
            else:
                edge_acc = torch.zeros(1)
            has_calc_edge_acc = True
            batch_edge_acc += edge_acc

        return batch_loss, batch_edge_acc
    ###def calc_loss_fn(self, loss_fns: dict, loss_fn_input: torch.Tensor, loss_fn_target: torch.Tensor, batch_loss: torch.Tensor,
    ###                 batch_edge_acc: torch.Tensor, num_batches: int, preds: torch.Tensor = None, y_labels: torch.Tensor = None,
    ###                 epoch_metrics: dict = None):
    ###    """
    ###    Calculate loss function
    ###    """
    ###    has_calc_edge_acc = False
    ###    for fn in loss_fns["fns"]:
    ###        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
    ###        loss_fns["fn_args"][fn_name].update({"input": loss_fn_input, "target": loss_fn_target})
    ###        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
    ###        loss = partial_fn()
    ###        batch_loss += loss/(num_batches*self.model_cfg['batch_size'])
    ###        if epoch_metrics:
    ###            epoch_metrics[fn_name] += loss/(num_batches*self.model_cfg['batch_size'])  # we don't reset epoch[fn_name] to 0 in each batch, so we need to divide by total number of batches
    ###        if has_calc_edge_acc:
    ###            continue
    ###        if "EdgeAcc" in fn_name:
    ###            edge_acc = (1-loss)/num_batches
    ###        elif self.model_cfg.get("edge_acc_metric_fn"):
    ###            edge_acc = (self.model_cfg.get("edge_acc_metric_fn")(loss_fn_input, loss_fn_target))/num_batches
    ###        elif preds is not None and y_labels is not None:
    ###            edge_acc = torch.mean((preds == y_labels).to(torch.float32))/num_batches
    ###        else:
    ###            edge_acc = torch.zeros(1)
    ###        has_calc_edge_acc = True
    ###        batch_edge_acc += edge_acc/self.model_cfg['batch_size']

    ###    return batch_loss, batch_edge_acc
    def infer_batch_data(self, batch_data: list, is_return_graph_embeds: bool = False):
        """
        Calculate batch data
        """
        num_nodes = self.model_cfg["num_nodes"]
        for data_batch_idx in range(self.model_cfg['batch_size']):
            data = batch_data[data_batch_idx]
            x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
            pred_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"]).unsqueeze(0)
            y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense().unsqueeze(0)
            batch_pred_graph_adj = pred_graph_adj if data_batch_idx == 0 else torch.cat((batch_pred_graph_adj, pred_graph_adj), dim=0)
            batch_y_graph_adj = y_graph_adj if data_batch_idx == 0 else torch.cat((batch_y_graph_adj, y_graph_adj), dim=0)

        # compute graph embeds
        pred_graph_embeds = self.get_pred_embeddings(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
        y_graph_embeds = self.graph_encoder.get_embeddings(y, y_edge_index, y_seq_batch_node_id, y_edge_attr)

        # used in observation model info in console
        log_model_info_data = data

        if is_return_graph_embeds:
            return batch_pred_graph_adj, batch_y_graph_adj, log_model_info_data, pred_graph_embeds, y_graph_embeds
        else:
            return batch_pred_graph_adj, batch_y_graph_adj, log_model_info_data

    @overload
    def train(self, mode: bool = True) -> torch.nn.Module:
        ...

    @overload
    def train(self, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False) -> tuple:
        ...

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False):
        """
        Training MTSCorrAD Model
        """
        # In order to make original function of nn.Module.train() work, we need to override it

        super().train(mode=mode)
        if train_data is None:
            return self

        self.show_model_config()
        best_model_info = self.init_best_model_info(train_data, loss_fns, epochs)
        best_model = []
        ###num_nodes = self.model_cfg["num_nodes"]
        train_loader = self.create_pyg_data_loaders(graph_adj_mats=train_data['edges'],  graph_nodes_mats=train_data["nodes"], target_mats=train_data["target"], loader_seq_len=self.model_cfg["seq_len"])
        num_batches = len(train_loader)
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "gra_enc_weight_l2_reg": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1),
                             "gra_enc_grad": torch.zeros(1), "gru_grad": torch.zeros(1), "gra_dec_grad": torch.zeros(1), "lr": torch.zeros(1),
                             "pred_gra_embeds": [], "y_gra_embeds": [], "gra_embeds_disparity": {}}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            for batch_idx, batch_data in enumerate(train_loader):
                ###batch_loss = torch.zeros(1)
                ###batch_edge_acc = torch.zeros(1)
                batch_pred_graph_adj, batch_y_graph_adj, log_model_info_data, pred_graph_embeds, y_graph_embeds = self.infer_batch_data(batch_data, is_return_graph_embeds=True)
                ###for data_batch_idx in range(self.model_cfg['batch_size']):
                ###    data = batch_data[data_batch_idx]
                ###    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                ###    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                ###    pred_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"]).unsqueeze(0)
                ###    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense().unsqueeze(0)
                ###    batch_pred_graph_adj = pred_graph_adj if data_batch_idx == 0 else torch.cat((batch_pred_graph_adj, pred_graph_adj), dim=0)
                ###    batch_y_graph_adj = y_graph_adj if data_batch_idx == 0 else torch.cat((batch_y_graph_adj, y_graph_adj), dim=0)
                ###calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_graph_adj, "loss_fn_target": batch_y_graph_adj,
                ###                       "batch_loss": batch_loss, "batch_edge_acc": batch_edge_acc, "epoch_metrics": epoch_metrics}
                calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_graph_adj, "loss_fn_target": batch_y_graph_adj,
                                       "num_batches": num_batches, "epoch_metrics": epoch_metrics}
                batch_loss, batch_edge_acc = self.calc_loss_fn(**calc_loss_fn_kwargs)

                if self.model_cfg['graph_enc_weight_l2_reg_lambda']:
                    gra_enc_weight_l2_penalty = self.model_cfg['graph_enc_weight_l2_reg_lambda']*sum(p.pow(2).mean() for p in self.graph_encoder.parameters())
                    batch_loss += gra_enc_weight_l2_penalty
                else:
                    gra_enc_weight_l2_penalty = 0
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                #### compute graph embeds
                ###pred_graph_embeds = self.get_pred_embeddings(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                ###y_graph_embeds = self.graph_encoder.get_embeddings(y, y_edge_index, y_seq_batch_node_id, y_edge_attr)
                # record metrics for each batch
                epoch_metrics["tr_loss"] += batch_loss/num_batches
                epoch_metrics["tr_edge_acc"] += batch_edge_acc/num_batches
                epoch_metrics["gra_enc_weight_l2_reg"] += gra_enc_weight_l2_penalty/num_batches
                epoch_metrics["gra_enc_grad"] += sum(p.grad.sum() for p in self.graph_encoder.parameters() if p.grad is not None)/num_batches
                epoch_metrics["gru_grad"] += sum(p.grad.sum() for layer in self.modules() if isinstance(layer, GRU) for p in layer.parameters() if p.grad is not None)/num_batches
                epoch_metrics["gra_dec_grad"] += sum(p.grad.sum() for p in self.decoder.parameters() if p.grad is not None)/num_batches
                epoch_metrics["lr"] = torch.tensor(self.optimizer.param_groups[0]['lr'])
                epoch_metrics["pred_gra_embeds"].append(pred_graph_embeds.tolist())
                epoch_metrics["y_gra_embeds"].append(y_graph_embeds.tolist())

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
                best_model_info["model_structure"] = str(self) + "\n" + "="*100 + "\n" + str(summary(self, log_model_info_data.x, log_model_info_data.edge_index, log_model_info_data.batch, log_model_info_data.edge_attr, self.model_cfg["output_type"], max_depth=20))
                if show_model_info:
                    logger.info(f"\nNumber of graphs:{log_model_info_data.num_graphs} in No.{batch_idx} batch, the model structure:\n{best_model_info['model_structure']}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.9f}" for k, v in epoch_metrics.items() if "embeds" not in k])
                logger.info(f"In Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | lr: {self.optimizer.param_groups[0]['lr']:.9f}")
            if epoch_i % 100 == 0:  # show oredictive and real adjacency matrix every 500 epochs
                obs_data_batch_idx = self.model_cfg['batch_size']-1
                logger.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:{obs_data_batch_idx} \ninput_graph_adj[:5]:\n{log_model_info_data.edge_attr[:5]}\npred_graph_adj:\n{batch_pred_graph_adj[obs_data_batch_idx]}\ny_graph_adj:\n{batch_y_graph_adj[obs_data_batch_idx]}\n")
                ###logger.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:{data_batch_idx} \ninput_graph_adj[:5]:\n{x_edge_attr[:5]}\npred_graph_adj:\n{pred_graph_adj}\ny_graph_adj:\n{y_graph_adj}\n")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None, show_loader_log: bool = False):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        ###num_nodes = self.model_cfg["num_nodes"]
        test_loader = self.create_pyg_data_loaders(graph_adj_mats=test_data["edges"],  graph_nodes_mats=test_data["nodes"], target_mats=test_data["target"], loader_seq_len=self.model_cfg["seq_len"], show_log=show_loader_log)
        num_batches = len(test_loader)
        with torch.no_grad():
            for batch_data in test_loader:
                ###batch_loss = torch.zeros(1)
                ###batch_edge_acc = torch.zeros(1)
                batch_pred_graph_adj, batch_y_graph_adj, _ = self.infer_batch_data(batch_data)
                ###for data_batch_idx in range(self.model_cfg['batch_size']):
                ###    data = batch_data[data_batch_idx]
                ###    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                ###    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                ###    pred_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"]).unsqueeze(0)
                ###    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense().unsqueeze(0)
                ###    batch_pred_graph_adj = pred_graph_adj if data_batch_idx == 0 else torch.cat((batch_pred_graph_adj, pred_graph_adj), dim=0)
                ###    batch_y_graph_adj = y_graph_adj if data_batch_idx == 0 else torch.cat((batch_y_graph_adj, y_graph_adj), dim=0)
                calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_graph_adj, "loss_fn_target": batch_y_graph_adj,
                                       "num_batches": num_batches}
                batch_loss, batch_edge_acc = self.calc_loss_fn(**calc_loss_fn_kwargs)

                test_loss += batch_loss/num_batches
                test_edge_acc += batch_edge_acc/num_batches

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
            # Inter-series modeling
            if type(self.graph_encoder).__name__ == "GinEncoder":
                graph_embeds = self.graph_encoder(x, edge_index, seq_batch_node_id)
            elif type(self.graph_encoder).__name__ == "GineEncoder":
                graph_embeds = self.graph_encoder(x, edge_index, seq_batch_node_id, edge_attr)

            # Temporal Modeling
            pred_graph_embeds, _ = self.gru1(graph_embeds)

        return pred_graph_embeds[-1]

    def create_pyg_data_loaders(self, graph_adj_mats: np.ndarray, graph_nodes_mats: np.ndarray, target_mats: np.ndarray, loader_seq_len : int,  show_log: bool = True, show_debug_info: bool = False):
        """
        Create Pytorch Geometric DataLoaders
        """
        # Create an instance of the GraphTimeSeriesDataset
        dataset = GraphTimeSeriesDataset(graph_adj_mats,  graph_nodes_mats, target_mats, model_cfg=self.model_cfg, show_log=show_log)
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
