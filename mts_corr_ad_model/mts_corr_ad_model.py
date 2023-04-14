#!/usr/bin/env python
# coding: utf-8
import argparse
import functools
import json
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
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, summary
from torch_geometric.utils import unbatch, unbatch_edge_index
from tqdm import tqdm

sys.path.append("/workspace/correlation-change-predict/ywt_library")
from discriminate import DiscriminationTester

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
discriminate_logger = logging.getLogger("discriminate")
matplotlib_logger = logging.getLogger("matplotlib")
logger.setLevel(logging.INFO)
discriminate_logger.setLevel(logging.INFO)
matplotlib_logger.setLevel(logging.ERROR)
mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.simplefilter("ignore")

# ## Load Graph Data
def create_data_loaders(data_loader_cfg: dict, model_cfg: dict, graph_adj_arr: np.ndarray, graph_node_arr: np.ndarray = None):
    logger.info(f"graph_adj_arr.shape:{graph_adj_arr.shape}")
    graph_time_step = graph_adj_arr.shape[0] - 1  # the graph of last "t" can't be used as train data
    dataset = []
    for g_t in range(graph_time_step):
        edge_index = torch.tensor(np.stack(np.where(~np.isnan(graph_adj_arr[g_t])), axis=1))
        edge_index_next_t = torch.tensor(np.stack(np.where(~np.isnan(graph_adj_arr[g_t + 1])), axis=1))
        edge_attr = torch.tensor(graph_adj_arr[g_t][~np.isnan(graph_adj_arr[g_t])].reshape(-1, 1), dtype=torch.float32)
        edge_attr_next_t = torch.tensor(graph_adj_arr[g_t + 1][~np.isnan(graph_adj_arr[g_t + 1])].reshape(-1, 1), dtype=torch.float32)
        node_attr = torch.tensor(np.ones((graph_adj_arr.shape[1], 1)), dtype=torch.float32)  # each node has only one attribute
        data_y = Data(x=node_attr, edge_index=edge_index_next_t.t().contiguous(), edge_attr=edge_attr_next_t)
        data = Data(x=node_attr, y=data_y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        dataset.append(data)
    #model_cfg["dim_out"] = data.y.shape[0]  # turn on this if the input of loss-function graphs
    model_cfg["num_node_features"] = data.num_node_features
    logger.info(f"data.num_node_features: {data.num_node_features}; data.num_edges: {data.num_edges}; data.num_edge_features: {data.num_edge_features}; data.is_undirected: {data.is_undirected()}")
    logger.info(f"data.x.shape: {data.x.shape}; data.y.x.shape: {data.y.x.shape}; data.edge_index.shape: {data.edge_index.shape}; data.edge_attr.shape: {data.edge_attr.shape}")

    # Create training, validation, and test sets
    train_dataset = dataset[:int(len(dataset) * 0.833333333334)]
    val_dataset = dataset[int(len(dataset) * 0.833333333334):int(len(dataset) * 0.91666667)]
    test_dataset = dataset[int(len(dataset) * 0.91666667):]

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=data_loader_cfg["tr_batch"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=data_loader_cfg["val_batch"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=data_loader_cfg["test_batch"], shuffle=False)

    # show info
    logger.info(f'Training set   = {len(train_dataset)} graphs')
    logger.info(f'Validation set = {len(val_dataset)} graphs')
    logger.info(f'Test set       = {len(test_dataset)} graphs')
    logger.debug('Train loader:')
    for i, subgraph in enumerate(train_loader):
        logger.debug(f' - Subgraph {i}: {subgraph} ; Subgraph {i}.num_graphs:{subgraph.num_graphs} ; Subgraph {i}.edge_index[::, 10:15]: {subgraph.edge_index[::, 10:15]}')

    logger.debug('Validation loader:')
    for i, subgraph in enumerate(val_loader):
        logger.debug(f' - Subgraph {i}: {subgraph} ; Subgraph{i}.num_graphs:{subgraph.num_graphs} ; Subgraph {i}.edge_index[::, 10:15]: {subgraph.edge_index[::, 10:15]}')

    logger.debug('Test loader:')
    for i, subgraph in enumerate(test_loader):
        logger.debug(f' - Subgraph {i}: {subgraph} ; Subgraph{i}.num_graphs:{subgraph.num_graphs} ; Subgraph {i}.edge_index[::, 10:15]: {subgraph.edge_index[::, 10:15]}')

    logger.debug('Peeking Train data:')
    first_batch_data = next(iter(train_loader))
    data_x_nodes_list = unbatch(first_batch_data.x, first_batch_data.batch)
    data_x_edges_ind_list = unbatch_edge_index(first_batch_data.edge_index, first_batch_data.batch)
    batch_edge_attr_start_ind = 0

    for i in range(data_loader_cfg["tr_batch"]):
        batch_edge_attr_end_ind = data_x_edges_ind_list[i].shape[1] + batch_edge_attr_start_ind
        data_x_nodes = data_x_nodes_list[i]
        data_x_edges_ind = data_x_edges_ind_list[i]
        data_x_edges = first_batch_data.edge_attr[batch_edge_attr_start_ind: batch_edge_attr_end_ind]
        data_y = first_batch_data.y[i]
        data_y_nodes = data_y.x
        data_y_edges = data_y.edge_attr
        data_y_edges_ind = data_y.edge_index
        logger.debug(f"\n batch0_x{i}.shape: {data_x_nodes.shape} \n batch0_x{i}[:5]:{data_x_nodes[:5]}")
        logger.debug(f"\n batch0_x{i}_edges.shape: {data_x_edges.shape} \n batch0_x{i}_edges[:5]:{data_x_edges[:5]}")
        logger.debug(f"\n batch0_x{i}_edges_ind.shape: {data_x_edges_ind.shape} \n batch0_x{i}_edges_ind[:5]:{data_x_edges_ind[::, :5]}")
        logger.debug(f"\n batch0_y{i}.shape: {data_y_nodes.shape} \n batch0_y{i}[:5]:{data_y_nodes[:5]}")
        logger.debug(f"\n batch0_y{i}_edges.shape: {data_y_edges.shape} \n batch0_y{i}_edges[:5]:{data_y_edges[:5]}")
        logger.debug(f"\n batch0_y{i}_edges_ind.shape: {data_y_edges_ind.shape} \n batch0_y{i}_edges_ind[:5]:{data_y_edges_ind[::, :5]}")
        batch_edge_attr_start_ind = batch_edge_attr_end_ind

    return train_loader, val_loader, test_loader


# ## Multi-Dimension Time-Series Correlation Anomly Detection Model
class GineEncoder(torch.nn.Module):
    """
    num_node_features: number of features per node in the graph, in this model every node has same size of features
    gra_enc_l: Number of layers of GINEconv
    gra_enc_h: output size of hidden layer of GINEconv
    """
    def __init__(self, num_node_features: int, gra_enc_l: int, gra_enc_h: int, gra_enc_edge_dim: int, **unused_kwargs):
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
            self.gine_convs.append(GINEConv(nn, edge_dim=gra_enc_edge_dim))


    def forward(self, x, edge_index, batch_node_id, edge_attr):
        """
        x: node attributes
        edge_index: existence of edges between nodes
        batch_node_id: assigns each node to a specific example.
        edge_attr: edge attributes
        """
        # Node embeddings
        nodes_emb_layers = []
        for i in range(self.gra_enc_l):
            if i:
                nodes_emb = self.gine_convs[i](nodes_emb, edge_index, edge_attr)
            else:
                nodes_emb = self.gine_convs[i](x, edge_index, edge_attr)  # the shape of nodes_embeds: [batch_size*num_nodes, gra_enc_h]
            nodes_emb_layers.append(nodes_emb)

        # Graph-level readout
        nodes_emb_pools = [global_add_pool(nodes_emb, batch_node_id) for nodes_emb in nodes_emb_layers]  # the shape of global_add_pool(nodes_emb, batch_node_id): [batch_size, gra_enc_h]
                                                                                                         # global_add_pool : make a super-node to represent the graph
        # Concatenate and form the graph embeddings
        graph_embeds = torch.cat(nodes_emb_pools, dim=1)  # the shape of graph_embeds: [batch_size, num_layers*gra_enc_h]

        return graph_embeds


    def get_embeddings(self, x, edge_index, batch_node_id, edge_attr):
        """
        get graph_embeddings with no_grad))
        """
        with torch.no_grad():
            graph_embeds = self.forward(x, edge_index, batch_node_id, edge_attr).reshape(-1)

        return graph_embeds


class GinEncoder(torch.nn.Module):
    """
    num_node_features: number of features per node in the graph, in this model every node has same size of features 
    gra_enc_l: Number of layers of GINconv
    gra_enc_h: output size of hidden layer of GINconv
    """
    def __init__(self, num_node_features: int, gra_enc_l: int, gra_enc_h: int, **unused_kwargs):
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
            self.gin_convs.append(GINConv(nn))


    def forward(self, x, edge_index, batch_node_id, *unused_args):
        """
        x: node attributes
        edge_index: existence of edges between nodes
        batch_node_id: assigns each node to a specific example.
        edge_attr: edge attributes
        """
        # Node embeddings
        nodes_emb_layers = []
        for i in range(self.gra_enc_l):
            if i:
                nodes_emb = self.gin_convs[i](nodes_emb, edge_index)
            else:
                nodes_emb = self.gin_convs[i](x, edge_index)  # the shape of nodes_embeds: [batch_size*num_nodes, gra_enc_h]
            nodes_emb_layers.append(nodes_emb)

        # Graph-level readout
        nodes_emb_pools = [global_add_pool(nodes_emb, batch_node_id) for nodes_emb in nodes_emb_layers]  # the shape of global_add_pool(nodes_emb, batch_node_id): [batch_size, gra_enc_h]
                                                                                                         # global_add_pool : make a super-node to represent the graph
        # Concatenate and form the graph embeddings
        graph_embeds = torch.cat(nodes_emb_pools, dim=1)  # the shape of graph_embeds: [batch_size, num_layers*gra_enc_h]

        return graph_embeds


    def get_embeddings(self, x, edge_index, batch_node_id, *unused_args):
        """
        get graph_embeddings with no_grad))
        """
        with torch.no_grad():
            graph_embeds = self.forward(x, edge_index, batch_node_id).reshape(-1)

        return graph_embeds


class MTSCorrAD(torch.nn.Module):
    """
    gru_h: The number of output size of GRU and features in the hidden state h of GRU
    dim_out: The number of output size of MTSCorrAD model
    """
    def __init__(self, graph_encoder: torch.nn.Module, gru_l: int, gru_h: int, dim_out: int, drop_pos: list, drop_p: float, **unused_kwargs):
        super(MTSCorrAD, self).__init__()
        self.drop_pos = drop_pos
        self.graph_encoder = graph_encoder
        gru_input_size = self.graph_encoder.gra_enc_l * self.graph_encoder.gra_enc_h  # the input size of GRU depend on the number of layers of GINconv
        self.gru1 = GRU(gru_input_size, gru_h, gru_l, dropout=drop_p) if "gru" in drop_pos else GRU(gru_input_size, gru_h, gru_l)
        self.lin1 = Linear(gru_h, dim_out)
        self.dropout = Dropout(p=drop_p)


    def forward(self, x, edge_index, batch_node_id, edge_attr, *unused_args):
        # Inter-series modeling
        if type(self.graph_encoder).__name__ == "GinEncoder":
            graph_embeds = self.graph_encoder(x, edge_index, batch_node_id)
        elif type(self.graph_encoder).__name__ == "GineEncoder":
            graph_embeds = self.graph_encoder(x, edge_index, batch_node_id, edge_attr)

        # Temporal Modeling
        gru_output, gru_hn = self.gru1(graph_embeds)  # regarding batch_size as time-steps(sequence length) by using "unbatched" input
        lin_output = self.lin1(gru_output[-1])  # gru_output[-1] => only take last time-step
        graph_embed_pred = self.dropout(lin_output)  if 'fc' in self.drop_pos else lin_output
        #graph_embed_pred = self.dropout(lin_output)

        return graph_embed_pred


# ## Loss function
def barlo_twins_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    loss function 
    """
    assert pred.shape == target.shape, "The shape of prediction and target aren't match"  # TODO


def discr_loss(graphs_info, test_model, criterion: torch.nn.modules.loss = MSELoss(), disp_r: float = 1, loss_r: float = 10):
    real_gra_embeds_dispiraty, real_pred_embeds_dispiraty  = 0, 0
    emb_r_list = [0] + np.linspace(0.01, 1, len(graphs_info)-1).tolist()
    for i, (emb_r, g_info) in enumerate(zip(emb_r_list, graphs_info)):
        if i == 0:
            comp_gra_embeds = test_model.graph_encoder.get_embeddings(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
            comp_pred_embeds = test_model(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
        else:
            gra_embeds = test_model.graph_encoder.get_embeddings(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
            pred_embeds = test_model(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
            real_gra_embeds_dispiraty += emb_r * criterion(comp_gra_embeds, gra_embeds)
            real_pred_embeds_dispiraty += emb_r * criterion(comp_pred_embeds, pred_embeds)
            disp_loss = -1 * (real_gra_embeds_dispiraty + disp_r * real_pred_embeds_dispiraty)
    ret_loss = loss_r * disp_loss

    return ret_loss


# ## Training Model
def train(train_model: torch.nn.Module, train_loader: torch_geometric.loader.dataloader.DataLoader,
          val_loader: torch_geometric.loader.dataloader.DataLoader, optim: torch.optim,
          loss_fns: dict, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False):
    best_model_info = {"num_training_graphs": len(train_loader.dataset),
                       "filt_mode": args.filt_mode,
                       "filt_quan": args.filt_quan,
                       "batchs_per_epoch": len(train_loader),
                       "epochs": epochs,
                       "train_batch": train_loader.batch_size,
                       "val_batch": val_loader.batch_size,
                       "optimizer": str(optim),
                       "loss_fns": str([fn.__name__ if hasattr(fn, '__name__') else str(fn) for fn in loss_fns["fns"]]),
                       "discr_loss_r": loss_fns["fn_args"]["discr_loss"]["loss_r"],
                       "discr_loss_disp_r": loss_fns["fn_args"]["discr_loss"]["disp_r"],
                       "drop_pos": train_model.drop_pos,
                       "graph_enc": type(train_model.graph_encoder).__name__,
                       "graph_enc_w_grad_history": [],
                       "graph_embeds_history": {"graph_embeds_pred": [], 
                                                "y_graph_embeds":[],
                                                "graph_embeds_disparity": dict()},
                       "min_val_loss": float('inf'),
                       "train_loss_history": [],
                       "tr_l2_loss_history": [],
                       "tr_discr_loss_history": [],
                       "val_loss_history": [],
                      }
    gra_embeds_disp_rec_keys = [str(i/(num_diff_graphs-1))+"_disp" for i in range(num_diff_graphs)]
    best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["train_gra_enc"] = {k: [] for k in gra_embeds_disp_rec_keys}
    best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["train_pred"] = {k: [] for k in gra_embeds_disp_rec_keys}
    best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["val_gra_enc"] = {k: [] for k in gra_embeds_disp_rec_keys}
    best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["val_pred"] = {k: [] for k in gra_embeds_disp_rec_keys}
    graph_enc_num_layers = sum(1 for _ in train_model.graph_encoder.parameters())
    graph_enc_w_grad_after = 0
    best_model = []
    train_disc_tester = DiscriminationTester(num_diff_graphs=num_diff_graphs, data_loader=train_loader, x_edge_attr_mats=gra_edges_data_mats[:int((len(gra_edges_data_mats) - 1) * 0.9)])  # the graph of last "t" can't be used as train data
    val_disc_tester = DiscriminationTester(num_diff_graphs=num_diff_graphs, data_loader=val_loader, x_edge_attr_mats=gra_edges_data_mats[int((len(gra_edges_data_mats) - 1) * 0.9):int((len(gra_edges_data_mats) - 1) * 0.95)])  # the graph of last "t" can't be used as train data
    for epoch_i in tqdm(range(epochs)):
        train_model.train()
        #epoch_discr_loss, epoch_l2_loss, epoch_tr_loss = 0, 0, 0
        epoch_loss = {"tr": torch.zeros(1), "val": torch.zeros(1), "MSELoss()": torch.zeros(1), "discr_loss": torch.zeros(1)}
        # Train on batches
        for batch_i, batched_data in enumerate(train_loader):
            torch.autograd.set_detect_anomaly(True)
            x, x_edge_index, x_batch_node_id, x_edge_attr = batched_data.x, batched_data.edge_index, batched_data.batch, batched_data.edge_attr
            y, y_edge_index, y_batch_node_id, y_edge_attr = batched_data.y[-1].x, batched_data.y[-1].edge_index, torch.zeros(batched_data.y[-1].x.shape[0], dtype=torch.int64), batched_data.y[-1].edge_attr  # only take y of x with last time-step on training
            graph_embeds_pred = train_model(x, x_edge_index, x_batch_node_id, x_edge_attr)
            y_graph_embeds = train_model.graph_encoder.get_embeddings(y, y_edge_index, y_batch_node_id, y_edge_attr)
            loss_fns["fn_args"]["MSELoss()"].update({"input": graph_embeds_pred, "target":  y_graph_embeds})
            loss_fns["fn_args"]["discr_loss"].update({"graphs_info": train_disc_tester.graphs_info, "test_model": train_model})
            optim.zero_grad()
            for fn in loss_fns["fns"]:
                fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                loss = partial_fn()
                epoch_loss[fn_name] += loss / len(train_loader)
                epoch_loss["tr"] +=  loss / len(train_loader)
                loss.backward(retain_graph=True)
            graph_enc_w_grad_after += sum(sum(torch.abs(torch.reshape(p.grad if p.grad is not None else torch.zeros((1,)), (-1,)))) for p in islice(train_model.graph_encoder.parameters(), 0, graph_enc_num_layers))  # sums up in each batch
            optim.step()
            best_model_info["graph_embeds_history"]["graph_embeds_pred"].append(graph_embeds_pred.tolist())
            best_model_info["graph_embeds_history"]["y_graph_embeds"].append(y_graph_embeds.tolist())
        else:
            log_model_info_data = batched_data
            log_model_info_batch_i = batch_i

        # Check if graph_encoder.parameters() have been updated
        assert graph_enc_w_grad_after>0, f"After loss.backward(), Sum of MainModel.graph_encoder weights in epoch_{epoch_i}:{graph_enc_w_grad_after}"

        # Validation
        epoch_loss['val'] = test(train_model, val_loader)

        # record training history
        best_model_info["train_loss_history"].append(epoch_loss["tr"].item())
        best_model_info["tr_l2_loss_history"].append(epoch_loss["MSELoss()"].item())
        best_model_info["tr_discr_loss_history"].append(epoch_loss["discr_loss"].item())
        best_model_info["val_loss_history"].append(epoch_loss['val'].item())
        best_model_info["graph_enc_w_grad_history"].append(graph_enc_w_grad_after.item())
        tr_gra_embeds_disp_ylds = train_disc_tester.yield_real_disc(train_model)
        val_gra_embeds_disp_ylds = val_disc_tester.yield_real_disc(train_model)
        for k, tr_gra_embeds_disp, val_gra_embeds_disp in zip(gra_embeds_disp_rec_keys, tr_gra_embeds_disp_ylds, val_gra_embeds_disp_ylds):
            best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["train_gra_enc"][k].append(tr_gra_embeds_disp["gra_enc_emb"])
            best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["train_pred"][k].append(tr_gra_embeds_disp["pred_emb"])
            best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["val_gra_enc"][k].append(val_gra_embeds_disp["gra_enc_emb"])
            best_model_info["graph_embeds_history"]["graph_embeds_disparity"]["val_pred"][k].append(val_gra_embeds_disp["pred_emb"])
            logger.debug(f"{k} of train graph: {tr_gra_embeds_disp}, {k} of val graph: {val_gra_embeds_disp}")

        # record training history and save best model
        if epoch_loss['val']<best_model_info["min_val_loss"]:
            best_model = train_model
            best_model_info["best_val_epoch"] = epoch_i
            best_model_info["min_val_loss"] = epoch_loss['val'].item()

        # observe model info in console
        if show_model_info and epoch_i==0:
            best_model_info["model_structure"] = str(train_model) + "\n" + "="*100 + "\n" + str(summary(train_model, log_model_info_data.x, log_model_info_data.edge_index, log_model_info_data.batch, log_model_info_data.edge_attr, max_depth=20))
            logger.info(f"\nNumber of graphs:{log_model_info_data.num_graphs} in No.{log_model_info_batch_i} batch, the model structure:\n{best_model_info['model_structure']}")
        if epoch_i % 10 == 0:  # show metrics every 10 epochs
            logger.info(f"Epoch {epoch_i:>3} | Train Loss: {epoch_loss['tr'].item():.5f} | Train L2 Loss: {epoch_loss['MSELoss()'].item():.5f} | Train graph emb disparity Loss: {epoch_loss['discr_loss'].item():.5f} | Val Loss: {epoch_loss['val'].item():.5f} ")

    return best_model, best_model_info


def test(test_model:torch.nn.Module, loader:torch_geometric.loader.dataloader.DataLoader, criterion: torch.nn.modules.loss = MSELoss()):
    test_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batched_data in loader:
            x, x_edge_index, x_batch_node_id, x_edge_attr = batched_data.x, batched_data.edge_index, batched_data.batch, batched_data.edge_attr
            y, y_edge_index, y_batch_node_id, y_edge_attr = batched_data.y[-1].x, batched_data.y[-1].edge_index, torch.zeros(batched_data.y[-1].x.shape[0], dtype=torch.int64), batched_data.y[-1].edge_attr  # only take y of x with last time-step on training
            graph_embeds_pred = test_model(x, x_edge_index, x_batch_node_id, x_edge_attr)
            y_graph_embeds = test_model.graph_encoder.get_embeddings(y, y_edge_index, y_batch_node_id, y_edge_attr)
            loss = criterion(graph_embeds_pred, y_graph_embeds)
            test_loss += loss / len(loader)

    return test_loss


def save_model(unsaved_model:torch.nn.Module, model_info:dict):
    e_i = model_info.get("best_val_epoch")
    t = datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    torch.save(unsaved_model, model_dir/f"epoch_{e_i}-{t}.pt")
    with open(model_log_dir/f"epoch_{e_i}-{t}.json","w") as f:
        json_str = json.dumps(model_info)
        f.write(json_str)
    logger.info(f"model has been saved in:{model_dir}")


if __name__ == "__main__":
    mts_corr_ad_args_parser = argparse.ArgumentParser()
    mts_corr_ad_args_parser.add_argument("--tr_batch", type=int, nargs='?', default=12,  # each graph contains 5 days correlation, so 4 graphs means a month, 12 graphs means a quarter
                                         help="input the number of training batch")
    mts_corr_ad_args_parser.add_argument("--val_batch", type=int, nargs='?', default=1,  # each graph contains 5 days correlation, so 4 graphs means a month, 12 graphs means a quarter
                                         help="input the number of training batch")
    mts_corr_ad_args_parser.add_argument("--test_batch", type=int, nargs='?', default=1,  # each graph contains 5 days correlation, so 4 graphs means a month, 12 graphs means a quarter
                                         help="input the number of training batch")
    mts_corr_ad_args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=1000,
                                         help="input the number of training epochs")
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
    mts_corr_ad_args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default="all_values",
                                         help="Decide mode of nodes' vaules of graph_nodes_matrix, look up the options by execute python ywt_library/data_module.py -h")
    mts_corr_ad_args_parser.add_argument("--discr_loss", type=bool, default=False, action=argparse.BooleanOptionalAction,
                                         help="input --discr_loss to add discrimination loss during training")
    mts_corr_ad_args_parser.add_argument("--discr_loss_r", type=float, nargs='?', default=0,
                                         help="Enter the ratio by which the discrimination loss should be multiplied")
    mts_corr_ad_args_parser.add_argument("--discr_pred_disp_r", type=float, nargs='?', default=1,
                                         help="Enter the ratio by which to multiply the predicted graph embedding disparity for the discrimination loss.")
    mts_corr_ad_args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                                         help="input [gru] | [gru fc] | [fc gru graph_encoder] to decide the position of drop layers")
    mts_corr_ad_args_parser.add_argument("--drop_p", type=float, default=0,
                                         help="input 0~1 to decide the probality of drop layers")
    mts_corr_ad_args_parser.add_argument("--gra_enc", type=str, nargs='?', default="gine",
                                         help="input the type of graph encoder")
    mts_corr_ad_args_parser.add_argument("--gra_enc_l", type=int, nargs='?', default=1,  # range:1~n, for graph encoder after the second layer,
                                         help="input the number of graph laryers of graph_encoder")
    mts_corr_ad_args_parser.add_argument("--gra_enc_h", type=int, nargs='?', default=4,
                                         help="input the number of graph embedding hidden size of graph_encoder")
    mts_corr_ad_args_parser.add_argument("--gru_l", type=int, nargs='?', default=1,  # range:1~n, for gru
                                         help="input the number of stacked-layers of gru")
    mts_corr_ad_args_parser.add_argument("--gru_h", type=int, nargs='?', default=8,
                                         help="input the number of gru hidden size")
    args = mts_corr_ad_args_parser.parse_args()
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

    # ## model configuration
    loader_cfg = {"tr_batch": args.tr_batch,
                  "val_batch": args.val_batch,
                  "test_batch": args.test_batch}
    mts_corr_ad_cfg = {"drop_pos": args.drop_pos,
                       "drop_p": args.drop_p,
                       "gra_enc_l": args.gra_enc_l,
                       "gra_enc_h": args.gra_enc_h,
                       "gru_l": args.gru_l,
                       "gru_h": args.gru_h}
    is_training, train_count = True, 0
    gra_edges_data_mats = np.load(graph_adj_mat_dir / f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    gra_nodes_data_mats = np.load(graph_node_mat_dir / f"{args.node}_s{s_l}_w{w_l}_nodes_mat.npy")
    train_graphs_loader, val_graphs_loader, test_graphs_loader = create_data_loaders(data_loader_cfg=loader_cfg, model_cfg=mts_corr_ad_cfg, graph_adj_arr=gra_edges_data_mats)
    mts_corr_ad_cfg["gra_enc_edge_dim"] = next(iter(train_graphs_loader)).edge_attr.shape[1]
    mts_corr_ad_cfg["dim_out"] = mts_corr_ad_cfg["gra_enc_l"] * mts_corr_ad_cfg["gra_enc_h"]
    gin_encoder = GinEncoder(**mts_corr_ad_cfg)
    gine_encoder = GineEncoder(**mts_corr_ad_cfg)
    mts_corr_ad_cfg["graph_encoder"] = gine_encoder if args.gra_enc == "gine" else gin_encoder
    model =  MTSCorrAD(**mts_corr_ad_cfg)
    loss_fns_dict = {"fns": [MSELoss()],
                     "fn_args": {"MSELoss()": {}, "discr_loss": {"disp_r": args.discr_loss_r, "loss_r": args.discr_pred_disp_r}}}
    loss_fns_dict["fns"] = loss_fns_dict["fns"] + [discr_loss] if args.discr_loss else loss_fns_dict["fns"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    while (is_training is True) and (train_count<100):
        try:
            train_count += 1
            model, model_info = train(model, train_graphs_loader, val_graphs_loader, optimizer, loss_fns_dict, epochs=args.tr_epochs, show_model_info=True)
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
                save_model(model, model_info)
