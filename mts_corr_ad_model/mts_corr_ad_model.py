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
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, summary
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch_geometric.utils import unbatch, unbatch_edge_index
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
    def __init__(self, graph_adj_mats: np.ndarray, graph_nodes_mats: np.ndarray, model_cfg: dict, show_log: bool = True):
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
                    edge_index_next_t = torch.tensor(np.stack(np.where(~np.isnan(graph_adj_mats[g_t + 1])), axis=1))
                    edge_attr_next_t = torch.tensor(graph_adj_mats[g_t + 1][~np.isnan(graph_adj_mats[g_t + 1])].reshape(-1, 1), dtype=torch.float64)
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

# ## Multi-Dimension Time-Series Correlation Anomly Detection Model
class GineEncoder(torch.nn.Module):
    """
    num_node_features: number of features per node in the graph, in this model every node has same size of features
    gra_enc_l: Number of layers of GINEconv
    gra_enc_h: output size of hidden layer of GINEconv
    """
    def __init__(self, num_node_features: int, gra_enc_l: int, gra_enc_h: int, gra_enc_aggr: str, num_edge_features: int, **unused_kwargs):
        super(GineEncoder, self).__init__()
        self.gra_enc_l = gra_enc_l
        self.gine_convs = torch.nn.ModuleList()
        self.gra_enc_h = gra_enc_h

        for i in range(self.gra_enc_l):
            if i:
                nn = Sequential(Linear(gra_enc_h, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU(),
                                Linear(gra_enc_h, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU())
            else:
                nn = Sequential(Linear(num_node_features, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU(),
                                Linear(gra_enc_h, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU())
            self.gine_convs.append(GINEConv(nn, edge_dim=num_edge_features, aggr=gra_enc_aggr))

    def forward(self, x, edge_index, seq_batch_node_id, edge_attr):
        """
        x: node attributes
        edge_index: existence of edges between nodes
        seq_batch_node_id: assigns each node to a specific example.
        edge_attr: edge attributes
        """
        # Node embeddings
        nodes_emb_layers = []
        for i in range(self.gra_enc_l):
            if i:
                nodes_emb = self.gine_convs[i](nodes_emb, edge_index, edge_attr)
            else:
                nodes_emb = self.gine_convs[i](x, edge_index, edge_attr)  # the shape of nodes_embeds: [seq_len*num_nodes, gra_enc_h]
                                                                          # (each graph represent for a time step, so seq_len means the number of graphs and number of time steps)
            nodes_emb_layers.append(nodes_emb)

        # Graph-level readout
        nodes_emb_pools = [global_add_pool(nodes_emb, seq_batch_node_id) for nodes_emb in nodes_emb_layers]  # global_add_pool : make a super-node to represent the graph
                                                                                                             # the shape of global_add_pool(nodes_emb, seq_batch_node_id): [seq_len, gra_enc_h]
                                                                                                             # seq_len means the number of graphs and number of time steps, so we regard batch_size as time-steps(seq_len))
        # Concatenate and form the graph embeddings
        graph_embeds = torch.cat(nodes_emb_pools, dim=1)  # the shape of graph_embeds: [seq_len, num_layers*gra_enc_h]
                                                          # (each graph represent for a time step, so seq_len means the number of graphs and number of time steps)
        return graph_embeds

    def get_embeddings(self, x, edge_index, seq_batch_node_id, edge_attr):
        """
        get graph_embeddings with no_grad))
        """
        with torch.no_grad():
            graph_embeds = self.forward(x, edge_index, seq_batch_node_id, edge_attr).reshape(-1)  # seq_len means the number of graphs and number of time steps, so we regard batch_size as time-steps(seq_len))

        return graph_embeds


class GinEncoder(torch.nn.Module):
    """
    num_node_features: number of features per node in the graph, in this model every node has same size of features 
    gra_enc_l: Number of layers of GINconv
    gra_enc_h: output size of hidden layer of GINconv
    """
    def __init__(self, num_node_features: int, gra_enc_l: int, gra_enc_h: int, gra_enc_aggr: str, **unused_kwargs):
        super(GinEncoder, self).__init__()
        self.gra_enc_l = gra_enc_l
        self.gin_convs = torch.nn.ModuleList()
        self.gra_enc_h = gra_enc_h

        for i in range(gra_enc_l):
            if i:
                nn = Sequential(Linear(gra_enc_h, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU(),
                                Linear(gra_enc_h, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU())
            else:
                nn = Sequential(Linear(num_node_features, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU(),
                                Linear(gra_enc_h, gra_enc_h),
                                BatchNorm1d(gra_enc_h), ReLU())
            self.gin_convs.append(GINConv(nn, aggr=gra_enc_aggr))


    def forward(self, x, edge_index, seq_batch_node_id, *unused_args):
        """
        x: node attributes
        edge_index: existence of edges between nodes
        seq_batch_node_id: assigns each node to a specific example.
        edge_attr: edge attributes
        """
        # Node embeddings
        nodes_emb_layers = []
        for i in range(self.gra_enc_l):
            if i:
                nodes_emb = self.gin_convs[i](nodes_emb, edge_index)
            else:
                nodes_emb = self.gin_convs[i](x, edge_index)  # the shape of nodes_embeds: [seq_len*num_nodes, gra_enc_h]
                                                              # (each graph represent for a time step, so seq_len means the number of graphs and number of time steps)
            nodes_emb_layers.append(nodes_emb)

        # Graph-level readout
        nodes_emb_pools = [global_add_pool(nodes_emb, seq_batch_node_id) for nodes_emb in nodes_emb_layers]  # global_add_pool : make a super-node to represent the graph
                                                                                                             # the shape of global_add_pool(nodes_emb, seq_batch_node_id): [seq_len, gra_enc_h]
                                                                                                             # seq_len means the number of graphs and number of time steps, so we regard batch_size as time-steps(seq_len))

        # Concatenate and form the graph embeddings
        graph_embeds = torch.cat(nodes_emb_pools, dim=1)  # the shape of graph_embeds: [seq_len, num_layers*gra_enc_h]
                                                          # (each graph represent for a time step, so seq_len means the number of graphs and number of time steps)
        return graph_embeds


    def get_embeddings(self, x, edge_index, seq_batch_node_id, *unused_args):
        """
        get graph_embeddings with no_grad))
        """
        with torch.no_grad():
            graph_embeds = self.forward(x, edge_index, seq_batch_node_id).reshape(-1)  # seq_len means the number of graphs and number of time steps, so we regard batch_size as time-steps(seq_len))

        return graph_embeds


class MTSCorrAD(torch.nn.Module):
    """
    Multi-Time Series Correlation Anomaly Detection (MTSCorrAD)
    """
    def __init__(self, model_cfg: dict):
        super(MTSCorrAD, self).__init__()
        self.model_cfg = model_cfg
        # create data loader
        self.num_tr_batches = self.model_cfg["num_batches"]['train']
        self.num_val_batches = self.model_cfg["num_batches"]['val']

        # set model components
        self.graph_encoder = self.model_cfg['graph_encoder'](**self.model_cfg)
        graph_enc_emb_size = self.graph_encoder.gra_enc_l * self.graph_encoder.gra_enc_h  # the input size of GRU depend on the number of layers of GINconv
        self.gru1 = GRU(graph_enc_emb_size, graph_enc_emb_size, self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"]) if "gru" in self.model_cfg["drop_pos"] else GRU(graph_enc_emb_size, graph_enc_emb_size, self.model_cfg["gru_l"])
        self.fc = Linear(graph_enc_emb_size, self.model_cfg["fc_out_dim"])
        self.decoder = self.model_cfg['decoder']()
        self.dropout = Dropout(p=self.model_cfg["drop_p"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.num_tr_batches*50, gamma=0.9)
        observe_model_cfg = {item[0]: item[1] for item in self.model_cfg.items() if item[0] != 'dataset'}
        self.graph_enc_num_layers = sum(1 for _ in self.graph_encoder.parameters())

        logger.info(f"\nModel Configuration: \n{observe_model_cfg}")

    def forward(self, x, edge_index, seq_batch_node_id, edge_attr, *unused_args):
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
        fc_output = self.fc(gru_output[-1])  # gru_output[-1] => only take last time-step
        fc_output = self.dropout(fc_output) if 'fc' in self.model_cfg['drop_pos'] else fc_output
        z = fc_output.reshape(-1, 1)
        pred_graph_adj = self.decoder.forward_all(z, sigmoid=False)

        return pred_graph_adj

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False):
        """
        Training MTSCorrAD Model
        """
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self

        best_model_info = {"num_training_graphs": len(train_data),
                           "filt_mode": self.model_cfg['filt_mode'],
                           "filt_quan": self.model_cfg['filt_quan'],
                           "graph_nodes_v_mode": self.model_cfg['graph_nodes_v_mode'],
                           "batches_per_epoch": self.num_tr_batches,
                           "epochs": epochs,
                           "batch_size": self.model_cfg['batch_size'],
                           "seq_len": self.model_cfg['seq_len'],
                           "optimizer": str(self.optimizer),
                           "loss_fns": str([fn.__name__ if hasattr(fn, '__name__') else str(fn) for fn in loss_fns["fns"]]),
                           "drop_pos": self.model_cfg["drop_pos"],
                           "graph_enc": type(self.graph_encoder).__name__,
                           "gra_enc_aggr": self.model_cfg['gra_enc_aggr'],
                           "graph_enc_w_grad_history": [],
                           "graph_embeds_history": {"pred_graph_embeds": [],
                                                    "y_graph_embeds": [],
                                                    "graph_embeds_disparity": {}},
                           "min_val_loss": float('inf')}
        graph_enc_w_grad_after = 0
        best_model = []

        train_loader = self.create_pyg_data_loaders(graph_adj_mats=train_data['edges'],  graph_nodes_mats=train_data["nodes"], loader_seq_len=self.model_cfg["seq_len"])
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1)}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            for batch_idx, batch_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                total_loss = torch.zeros(1)
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                    num_nodes = y.shape[0]
                    pred_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    edge_acc = np.isclose(pred_graph_adj.cpu().detach().numpy(), y_graph_adj.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                    epoch_metrics["tr_edge_acc"] += edge_acc / (self.num_tr_batches*self.model_cfg['batch_size'])
                    loss_fns["fn_args"]["MSELoss()"].update({"input": pred_graph_adj, "target":  y_graph_adj})
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        total_loss += loss / (self.num_tr_batches*self.model_cfg['batch_size'])
                        epoch_metrics[fn_name] += loss / (self.num_tr_batches*self.model_cfg['batch_size'])
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_metrics["tr_loss"] += total_loss
                log_model_info_data = data
                log_model_info_batch_idx = batch_idx
            pred_graph_embeds = self.get_pred_embeddings(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
            y_graph_embeds = self.graph_encoder.get_embeddings(y, y_edge_index, y_seq_batch_node_id, y_edge_attr)
            best_model_info["graph_embeds_history"]["pred_graph_embeds"].append(pred_graph_embeds.tolist())
            best_model_info["graph_embeds_history"]["y_graph_embeds"].append(y_graph_embeds.tolist())

            # Check if graph_encoder.parameters() have been updated
            graph_enc_w_grad_after += sum(sum(torch.abs(torch.reshape(p.grad if p.grad is not None else torch.zeros((1,)), (-1,)))) for p in islice(self.graph_encoder.parameters(), 0, self.graph_enc_num_layers))  # sums up in each batch
            assert graph_enc_w_grad_after > 0, f"After loss.backward(), Sum of MainModel.graph_encoder weights in epoch_{epoch_i}:{graph_enc_w_grad_after}"
            # Validation
            epoch_metrics['val_loss'], epoch_metrics['val_edge_acc'] = self.test(val_data, loss_fns=loss_fns, show_loader_log=True if epoch_i == 0 else False)

            # record training history and save best model
            for k, v in epoch_metrics.items():
                history_list = best_model_info.setdefault(k+"_history", [])
                history_list.append(v.item())
            best_model_info["graph_enc_w_grad_history"].append(graph_enc_w_grad_after.item())
            if epoch_metrics['val_loss'] < best_model_info["min_val_loss"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["min_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["min_val_loss_edge_acc"] = epoch_metrics['val_edge_acc'].item()

            # observe model info in console
            if epoch_i == 0:
                best_model_info["model_structure"] = str(self) + "\n" + "="*100 + "\n" + str(summary(self, log_model_info_data.x, log_model_info_data.edge_index, log_model_info_data.batch, log_model_info_data.edge_attr, max_depth=20))
                if show_model_info:
                    logger.info(f"\nNumber of graphs:{log_model_info_data.num_graphs} in No.{log_model_info_batch_idx} batch, the model structure:\n{best_model_info['model_structure']}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.5f}" for k, v in epoch_metrics.items()])
                logger.info(f"Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | Graph encoder weights grad: {graph_enc_w_grad_after.item():.5f}")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None, show_loader_log: bool = False):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        test_loader = self.create_pyg_data_loaders(graph_adj_mats=test_data["edges"],  graph_nodes_mats=test_data["nodes"], loader_seq_len=self.model_cfg["seq_len"], show_log=show_loader_log)
        with torch.no_grad():
            for batch_data in test_loader:
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                    num_nodes = y.shape[0]
                    pred_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    edge_acc = np.isclose(pred_graph_adj.cpu().detach().numpy(), y_graph_adj.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                    test_edge_acc += edge_acc / (self.num_val_batches*self.model_cfg['batch_size'])
                    loss_fns["fn_args"]["MSELoss()"].update({"input": pred_graph_adj, "target":  y_graph_adj})
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        test_loss += loss / (self.num_val_batches*self.model_cfg['batch_size'])

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

        return pred_graph_embeds

    def create_pyg_data_loaders(self, graph_adj_mats: np.ndarray, graph_nodes_mats: np.ndarray, loader_seq_len : int,  show_log: bool = True, show_debug_info: bool = False):
        """
        Create Pytorch Geometric DataLoaders
        """
        # Create an instance of the GraphTimeSeriesDataset
        dataset = GraphTimeSeriesDataset(graph_adj_mats,  graph_nodes_mats, model_cfg=self.model_cfg)
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
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    if 1 < data_batch_idx < (self.model_cfg['batch_size'] - 1):  # only peek the first and last data instances in the batch_data
                        continue
                    logger.debug(f' - Subgraph of data_batch_idx-{data_batch_idx} in batch-{batch_idx}: {subgraph[data_batch_idx]} ; num_graphs:{subgraph[data_batch_idx].num_graphs} ; edge_index[::, 10:15]: {subgraph[data_batch_idx].edge_index[::, 10:15]}')

            logger.debug('Peeking info of data instance:')
            for batch_idx, batch_data in enumerate(data_loader):
                if 2 < batch_idx < (len(data_loader)-1):  # only peek the first 2 and last batches
                    continue
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    if data_batch_idx > 0:  # only peek the first data instance in the batch_data
                        continue
                    data_x_nodes_list = unbatch(batch_data[data_batch_idx].x, batch_data[data_batch_idx].batch)
                    data_x_edges_idx_list = unbatch_edge_index(batch_data[data_batch_idx].edge_index, batch_data[data_batch_idx].batch)
                    batch_edge_attr_start_idx = 0
                    for seq_t in range(loader_seq_len):
                        if 3 < seq_t < (loader_seq_len-2):  # only peek the first 3 and last 2 seq_t
                            continue
                        batch_edge_attr_end_idx = data_x_edges_idx_list[seq_t].shape[1] + batch_edge_attr_start_idx
                        data_x_nodes = data_x_nodes_list[seq_t]
                        data_x_edges_idx = data_x_edges_idx_list[seq_t]
                        data_x_edges = batch_data[data_batch_idx].edge_attr[batch_edge_attr_start_idx: batch_edge_attr_end_idx]
                        data_y = batch_data[data_batch_idx].y[seq_t]
                        data_y_nodes = data_y.x
                        data_y_edges = data_y.edge_attr
                        data_y_edges_idx = data_y.edge_index
                        batch_edge_attr_start_idx = batch_edge_attr_end_idx

                        logger.debug(f"\n batch{batch_idx}_data{data_batch_idx}_x.shape at t{seq_t}: {data_x_nodes.shape} \n batch{batch_idx}_data{data_batch_idx}_x[:5] at t{seq_t}:\n{data_x_nodes[:5]}")
                        logger.debug(f"\n batch{batch_idx}_data{data_batch_idx}_x_edges.shape at t{seq_t}: {data_x_edges.shape} \n batch{batch_idx}_data{data_batch_idx}_x_edges[:5] at t{seq_t}:\n{data_x_edges[:5]}")
                        logger.debug(f"\n batch{batch_idx}_data{data_batch_idx}_x_edges_idx.shape at t{seq_t}: {data_x_edges_idx.shape} \n batch{batch_idx}_data{data_batch_idx}_x_edges_idx[:5] at t{seq_t}:\n{data_x_edges_idx[::, :5]}")
                        logger.debug(f"\n batch{batch_idx}_data{data_batch_idx}_y.shape at t{seq_t}: {data_y_nodes.shape} \n batch{batch_idx}_data{data_batch_idx}_y[:5] at t{seq_t}:\n{data_y_nodes[:5]}")
                        logger.debug(f"\n batch{batch_idx}_data{data_batch_idx}_y_edges.shape at t{seq_t}: {data_y_edges.shape} \n batch{batch_idx}_data{data_batch_idx}_y_edges[:5] at t{seq_t}:\n{data_y_edges[:5]}")
                        logger.debug(f"\n batch{batch_idx}_data{data_batch_idx}_y_edges_idx.shape at t{seq_t}: {data_y_edges_idx.shape} \n batch{batch_idx}_data{data_batch_idx}_y_edges_idx[:5] at t{seq_t}:\n{data_y_edges_idx[::, :5]}")

        return data_loader


if __name__ == "__main__":
    mts_corr_ad_args_parser = argparse.ArgumentParser()
    mts_corr_ad_args_parser.add_argument("--batch_size", type=int, nargs='?', default=10,
                                         help="input the number of batch size")
    mts_corr_ad_args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=300,
                                         help="input the number of training epochs")
    mts_corr_ad_args_parser.add_argument("--seq_len", type=int, nargs='?', default=30,
                                         help="input the number of sequence length")
    mts_corr_ad_args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                         help="input --save_model to save model weight and model info")
    mts_corr_ad_args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                                         help="input the number of stride length of correlation computing")
    mts_corr_ad_args_parser.add_argument("--corr_window", type=int, nargs='?', default=10,
                                         help="input the number of window length of correlation computing")
    mts_corr_ad_args_parser.add_argument("--filt_mode", type=str, nargs='?', default=None,
                                         help="input the filtered mode of graph edges, look up the options by execute python ywt_library/data_module.py -h")
    mts_corr_ad_args_parser.add_argument("--filt_quan", type=float, nargs='?', default=0.5,
                                         help="input the filtered quantile of graph edges")
    mts_corr_ad_args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default=None,
                                         help="Decide mode of nodes' vaules of graph_nodes_matrix, look up the options by execute python ywt_library/data_module.py -h")
    mts_corr_ad_args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                                         help="input [gru] | [gru fc] | [fc gru graph_encoder] to decide the position of drop layers")
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
    ARGS = mts_corr_ad_args_parser.parse_args()
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
    torch.set_default_tensor_type(torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor)
    torch.autograd.set_detect_anomaly(True)  # for debug grad
    logger.info(f"===== file_name basis:{output_file_name} =====")
    logger.info(f"===== pytorch running on:{device} =====")

    s_l, w_l = ARGS.corr_stride, ARGS.corr_window
    graph_adj_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/filtered_graph_adj_mat/{ARGS.filt_mode}-quan{str(ARGS.filt_quan).replace('.', '')}" if ARGS.filt_mode else Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_adj_mat"
    graph_nodes_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_node_mat"
    g_model_dir = current_dir / f'save_models/mts_corr_ad_model/{output_file_name}/corr_s{s_l}_w{w_l}'
    g_model_log_dir = current_dir / f'save_models/mts_corr_ad_model/{output_file_name}/corr_s{s_l}_w{w_l}/train_logs/'
    g_model_dir.mkdir(parents=True, exist_ok=True)
    g_model_log_dir.mkdir(parents=True, exist_ok=True)

    # ## model configuration
    is_training, train_count = True, 0
    graph_adj_data = np.load(graph_adj_data_dir / f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    graph_nodes_data = np.load(graph_nodes_data_dir / f"{ARGS.graph_nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy") if ARGS.graph_nodes_v_mode else np.ones((graph_adj_data.shape[0], 1, graph_adj_data.shape[2]))
    norm_train_dataset, norm_val_dataset, norm_test_dataset, scaler = split_and_norm_data(graph_adj_data, graph_nodes_data)
    # show info
    logger.info(f"graph_adj_data.shape:{graph_adj_data.shape}, graph_nodes_data.shape:{graph_nodes_data.shape}")
    logger.info(f"graph_adj_data.max:{np.nanmax(graph_adj_data)}, graph_adj_data.min:{np.nanmin(graph_adj_data)}")
    logger.info(f"graph_nodes_data.max:{np.nanmax(graph_nodes_data)}, graph_nodes_data.min:{np.nanmin(graph_nodes_data)}")
    logger.info(f"norm_train_nodes_data_mats.max:{np.nanmax(norm_train_dataset['nodes'])}, norm_train_nodes_data_mats.min:{np.nanmin(norm_train_dataset['nodes'])}")
    logger.info(f"norm_val_nodes_data_mats.max:{np.nanmax(norm_val_dataset['nodes'])}, norm_val_nodes_data_mats.min:{np.nanmin(norm_val_dataset['nodes'])}")
    logger.info(f"norm_test_nodes_data_mats.max:{np.nanmax(norm_test_dataset['nodes'])}, norm_test_nodes_data_mats.min:{np.nanmin(norm_test_dataset['nodes'])}")
    logger.info(f'Training set   = {len(norm_train_dataset["edges"])} graphs')
    logger.info(f'Validation set = {len(norm_val_dataset["edges"])} graphs')
    logger.info(f'Test set       = {len(norm_test_dataset["edges"])} graphs')
    logger.info("="*80)

    mts_corr_ad_cfg = {"filt_mode": ARGS.filt_mode,
                       "filt_quan": ARGS.filt_quan,
                       "graph_nodes_v_mode": ARGS.graph_nodes_v_mode,
                       "batch_size": ARGS.batch_size,
                       "seq_len": ARGS.seq_len,
                       "num_batches": {"train": ((len(norm_train_dataset["edges"])-1)//ARGS.batch_size),
                                       "val": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size),
                                       "test": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size)},
                       "drop_pos": ARGS.drop_pos,
                       "drop_p": ARGS.drop_p,
                       "gra_enc_aggr": ARGS.gra_enc_aggr,
                       "gra_enc_l": ARGS.gra_enc_l,
                       "gra_enc_h": ARGS.gra_enc_h,
                       "gru_l": ARGS.gru_l,
                       "fc_out_dim": norm_train_dataset["edges"].shape[1],
                       "num_node_features": norm_train_dataset["nodes"].shape[2],
                       "num_edge_features": 1,
                       "graph_encoder": GineEncoder if ARGS.gra_enc == "gine" else GinEncoder,
                       "decoder": InnerProductDecoder}

    model = MTSCorrAD(mts_corr_ad_cfg)
    loss_fns_dict = {"fns": [MSELoss()],
                     "fn_args": {"MSELoss()": {}}}
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
