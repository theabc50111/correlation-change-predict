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
from torch.nn import GRU, Dropout, Linear, MSELoss, Sequential, Softmax
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
from mts_corr_ad_model import GraphTimeSeriesDataset

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


class ClassMTSCorrAD(torch.nn.Module):
    """
    Multi-Time Series Correlation Anomaly Detection (MTSCorrAD)
    """
    def __init__(self, model_cfg: dict):
        super(ClassMTSCorrAD, self).__init__()
        self.model_cfg = model_cfg
        # create data loader
        self.num_tr_batches = self.model_cfg["num_batches"]['train']
        self.num_val_batches = self.model_cfg["num_batches"]['val']

        # set model components
        self.graph_encoder = self.model_cfg['graph_encoder'](**self.model_cfg)
        graph_enc_emb_size = self.graph_encoder.gra_enc_l * self.graph_encoder.gra_enc_h  # the input size of GRU depend on the number of layers of GINconv
        self.gru1 = GRU(graph_enc_emb_size, self.model_cfg["gru_h"], self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0)
        self.decoder = self.model_cfg['decoder'](self.model_cfg['gru_h'], self.model_cfg["num_nodes"], drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        self.fc1 = Sequential(OrderedDict([
                                          ("class_fc1", Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2)),
                                          ("class_fc1_drop", Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
                                          ]))
        self.fc2 = Sequential(OrderedDict([
                                          ("class_fc2", Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2)),
                                          ("class_fc2_drop", Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
                                          ]))
        self.fc3 = Sequential(OrderedDict([
                                          ("class_fc3", Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2)),
                                          ("class_fc3_drop", Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
                                          ]))
        self.softmax = Softmax(dim=1)
        if self.model_cfg["pretrain_encoder"]:
            self.graph_encoder.load_state_dict(torch.load(self.model_cfg["pretrain_encoder"]))
        if self.model_cfg["pretrain_decoder"]:
            self.decoder.load_state_dict(torch.load(self.model_cfg["pretrain_decoder"]))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.model_cfg['learning_rate'], weight_decay=self.model_cfg['weight_decay'])
        schedulers = [ConstantLR(self.optimizer, factor=0.1, total_iters=self.num_tr_batches*6), MultiStepLR(self.optimizer, milestones=list(range(self.num_tr_batches*5, self.num_tr_batches*600, self.num_tr_batches*50))+list(range(self.num_tr_batches*600, self.num_tr_batches*self.model_cfg['tr_epochs'], self.num_tr_batches*100)), gamma=0.9)]
        self.scheduler = SequentialLR(self.optimizer, schedulers=schedulers, milestones=[self.num_tr_batches*6])
        observe_model_cfg = {item[0]: item[1] for item in self.model_cfg.items() if item[0] != 'dataset'}
        observe_model_cfg['optimizer'] = str(self.optimizer)
        observe_model_cfg['scheduler'] = {"scheduler_name": str(self.scheduler.__class__.__name__), "milestones": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones), "gamma": self.scheduler._schedulers[1].gamma}

        logger.info(f"\nModel Configuration: \n{observe_model_cfg}")

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
        flatten_pred_graph_adj = pred_graph_adj.view(1, -1)
        fc1_output = self.fc1(flatten_pred_graph_adj)
        fc2_output = self.fc2(flatten_pred_graph_adj)
        fc3_output = self.fc3(flatten_pred_graph_adj)
        logits = torch.cat([fc1_output, fc2_output, fc3_output], dim=0).t()
        outputs = self.softmax(logits)

        return outputs

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False):
        """
        Training MTSCorrAD Model
        """
        # In order to make original function of nn.Module.train() work, we need to override it

        super().train(mode=mode)
        if train_data is None:
            return self

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
                           "optimizer": str(self.optimizer),
                           "opt_scheduler": {"gamma": self.scheduler._schedulers[1].gamma, "milestoines": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones)},
                           "loss_fns": str([fn.__name__ if hasattr(fn, '__name__') else str(fn) for fn in loss_fns["fns"]]),
                           "gra_enc_weight_l2_reg_lambda": self.model_cfg['graph_enc_weight_l2_reg_lambda'],
                           "drop_pos": self.model_cfg["drop_pos"],
                           "drop_p": self.model_cfg["drop_p"],
                           "graph_enc": type(self.graph_encoder).__name__,
                           "gra_enc_aggr": self.model_cfg['gra_enc_aggr'],
                           "min_val_loss": float('inf'),
                           "output_type": self.model_cfg['output_type'],
                           "output_bins": '_'.join((str(f) for f in self.model_cfg['output_bins'])).replace('.', '') if self.model_cfg['output_bins'] else None,
                           "target_mats_bins": self.model_cfg['target_mats_bins'],
                           "edge_acc_loss_atol": self.model_cfg['edge_acc_loss_atol'],
                           "use_bin_edge_acc_loss": self.model_cfg['use_bin_edge_acc_loss']}
        best_model = []

        num_nodes = self.model_cfg["num_nodes"]
        train_loader = self.create_pyg_data_loaders(graph_adj_mats=train_data['edges'],  graph_nodes_mats=train_data["nodes"], target_mats=train_data["target"], loader_seq_len=self.model_cfg["seq_len"])
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "gra_enc_weight_l2_reg": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1),
                             "gra_enc_grad": torch.zeros(1), "gru_grad": torch.zeros(1), "gra_dec_grad": torch.zeros(1), "lr": torch.zeros(1),
                             "pred_gra_embeds": [], "y_gra_embeds": [], "gra_embeds_disparity": {}}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            for batch_idx, batch_data in enumerate(train_loader):
                batch_loss = torch.zeros(1)
                batch_edge_acc = torch.zeros(1)
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                    pred_prob = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"])
                    preds = torch.argmax(pred_prob, dim=1)
                    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    y_labels = (y_graph_adj.view(-1)+1).to(torch.long)
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        loss_fns["fn_args"][fn_name].update({"input": pred_prob, "target": y_labels})
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        batch_loss += loss/self.model_cfg['batch_size']
                        epoch_metrics[fn_name] += loss/(self.num_tr_batches*self.model_cfg['batch_size'])  # we don't reset epoch[fn_name] to 0 in each batch, so we need to divide by total number of batches
                        if "EdgeAcc" in fn_name:
                            edge_acc = 1-loss
                            batch_edge_acc += edge_acc/self.model_cfg['batch_size']
                        else:
                            edge_acc = torch.mean((preds == y_labels).to(torch.float32))
                            batch_edge_acc += edge_acc/self.model_cfg['batch_size']

                if self.model_cfg['graph_enc_weight_l2_reg_lambda']:
                    gra_enc_weight_l2_penalty = self.model_cfg['graph_enc_weight_l2_reg_lambda']*sum(p.pow(2).mean() for p in self.graph_encoder.parameters())
                    batch_loss += gra_enc_weight_l2_penalty
                else:
                    gra_enc_weight_l2_penalty = 0
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # compute graph embeds
                pred_graph_embeds = self.get_pred_embeddings(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                y_graph_embeds = self.graph_encoder.get_embeddings(y, y_edge_index, y_seq_batch_node_id, y_edge_attr)
                # record metrics for each batch
                epoch_metrics["tr_loss"] += batch_loss/self.num_tr_batches
                epoch_metrics["tr_edge_acc"] += batch_edge_acc/self.num_tr_batches
                epoch_metrics["gra_enc_weight_l2_reg"] += gra_enc_weight_l2_penalty/self.num_tr_batches
                epoch_metrics["gra_enc_grad"] += sum(p.grad.sum() for p in self.graph_encoder.parameters() if p.grad is not None)/self.num_tr_batches
                epoch_metrics["gru_grad"] += sum(p.grad.sum() for p in self.gru1.parameters() if p.grad is not None)/self.num_tr_batches
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
                best_model_info["model_structure"] = str(self) + "\n" + "="*100 + "\n" + str(summary(self, log_model_info_data.x, log_model_info_data.edge_index, log_model_info_data.batch, log_model_info_data.edge_attr, self.model_cfg["output_type"], max_depth=20))
                if show_model_info:
                    logger.info(f"\nNumber of graphs:{log_model_info_data.num_graphs} in No.{log_model_info_batch_idx} batch, the model structure:\n{best_model_info['model_structure']}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.9f}" for k, v in epoch_metrics.items() if "embeds" not in k])
                logger.info(f"In Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | lr: {self.optimizer.param_groups[0]['lr']:.9f}")
            if epoch_i % 100 == 0:  # show oredictive and real adjacency matrix every 500 epochs
                logger.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:{data_batch_idx} \ninput_graph_adj[:5]:\n{x_edge_attr[:5]}\npreds:\n{preds}\ny_graph_adj:\n{y_graph_adj}\n")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None, show_loader_log: bool = False):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        num_nodes = self.model_cfg["num_nodes"]
        test_loader = self.create_pyg_data_loaders(graph_adj_mats=test_data["edges"],  graph_nodes_mats=test_data["nodes"], target_mats=test_data["target"], loader_seq_len=self.model_cfg["seq_len"], show_log=show_loader_log)
        with torch.no_grad():
            for batch_data in test_loader:
                batch_loss = torch.zeros(1)
                batch_edge_acc = torch.zeros(1)
                for data_batch_idx in range(self.model_cfg['batch_size']):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                    pred_prob = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"])
                    preds = torch.argmax(pred_prob, dim=1)
                    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    y_labels = (y_graph_adj.view(-1)+1).to(torch.long)
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        loss_fns["fn_args"][fn_name].update({"input": pred_prob, "target":  y_labels})
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        batch_loss += loss/self.model_cfg['batch_size']
                        if "EdgeAcc" in fn_name:
                            edge_acc = 1-loss
                            batch_edge_acc += edge_acc/self.model_cfg['batch_size']
                        else:
                            edge_acc = torch.mean((preds == y_labels).to(torch.float32))
                            batch_edge_acc += edge_acc/self.model_cfg['batch_size']

                test_loss += batch_loss/self.num_val_batches
                test_edge_acc += batch_edge_acc/self.num_val_batches

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
