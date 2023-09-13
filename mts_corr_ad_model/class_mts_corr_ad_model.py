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
from typing import overload

import dynamic_yaml
import matplotlib as mpl
import numpy as np
import torch
import yaml
from torch.nn import GRU, Dropout, Linear, MSELoss, ReLU, Sequential, Softmax
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


class ClassMTSCorrAD(MTSCorrAD):
    """
    Classification-Multi-Time Series Correlation Anomaly Detection (MTSCorrAD)
    Structure of MTSCorrAD3:
                                        ↗ --> FC1 --↘
        GraphEncoder -> GRU -> Decoder -----> FC2 --> Softmax
                                        ↘ --> FC3 --↗
    """
    def __init__(self, model_cfg: dict):
        super(ClassMTSCorrAD, self).__init__(model_cfg)
        self.model_cfg = model_cfg

        # set model components
        self.fc1 = Sequential(OrderedDict([
                                          ("class_fc1", Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2)),
                                          ("class_fc1_relu", ReLU()),
                                          ("class_fc1_drop", Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
                                          ]))
        self.fc2 = Sequential(OrderedDict([
                                          ("class_fc2", Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2)),
                                          ("class_fc2_relu", ReLU()),
                                          ("class_fc2_drop", Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
                                          ]))
        self.fc3 = Sequential(OrderedDict([
                                          ("class_fc3", Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2)),
                                          ("class_fc3_relu", ReLU()),
                                          ("class_fc3_drop", Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
                                          ]))
        self.softmax = Softmax(dim=0)
        ###self.softmax = Softmax(dim=1)

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

        # Classification layers
        flatten_pred_graph_adj = pred_graph_adj.view(1, -1)
        fc1_output = self.fc1(flatten_pred_graph_adj)
        fc2_output = self.fc2(flatten_pred_graph_adj)
        fc3_output = self.fc3(flatten_pred_graph_adj)
        logits = torch.cat([fc1_output, fc2_output, fc3_output], dim=0)
        ###logits = torch.cat([fc1_output, fc2_output, fc3_output], dim=0).t()
        outputs = self.softmax(logits)

        return outputs

    def infer_batch_data(self, batch_data: list, is_return_graph_embeds: bool = False):
        """
        Calculate batch data
        """
        num_nodes = self.model_cfg["num_nodes"]
        for data_batch_idx in range(self.model_cfg['batch_size']):
            data = batch_data[data_batch_idx]
            x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
            pred_prob = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"]).unsqueeze(0)
            preds = torch.argmax(pred_prob, dim=1)
            y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
            y_labels = (y_graph_adj.view(-1)+1).to(torch.long).unsqueeze(0)
            batch_pred_prob = pred_prob if data_batch_idx == 0 else torch.cat([batch_pred_prob, pred_prob], dim=0)
            batch_preds = preds if data_batch_idx == 0 else torch.cat([batch_preds, preds], dim=0)
            batch_y_labels = y_labels if data_batch_idx == 0 else torch.cat([batch_y_labels, y_labels], dim=0)

        # compute graph embeds
        pred_graph_embeds = self.get_pred_embeddings(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
        y_graph_embeds = self.graph_encoder.get_embeddings(y, y_edge_index, y_seq_batch_node_id, y_edge_attr)
        # used in observation model info in console
        log_model_info_data = data

        if is_return_graph_embeds:
            return batch_pred_prob, batch_preds, batch_y_labels, pred_graph_embeds, y_graph_embeds, log_model_info_data
        else:
            return batch_pred_prob, batch_preds, batch_y_labels, log_model_info_data

    @overload
    def train(self, mode: bool = True) -> torch.nn.Module:
        ...

    @overload
    def train(self, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False) -> tuple:
        ...

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, num_diff_graphs: int = 5, show_model_info: bool = False):
        """
        Training ClassMTSCorrAD Model
        """
        # In order to make original function of nn.Module.train() work, we need to override it

        super().train(mode=mode)
        if train_data is None:
            return self

        self.show_model_config()
        best_model_info = self.init_best_model_info(train_data, loss_fns, epochs)
        best_model_info.update({"max_val_edge_acc": 0})
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
                ###for data_batch_idx in range(self.model_cfg['batch_size']):
                ###    data = batch_data[data_batch_idx]
                ###    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                ###    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                ###    pred_prob = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"]).unsqueeze(0)
                ###    preds = torch.argmax(pred_prob, dim=1)
                ###    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                ###    y_labels = (y_graph_adj.view(-1)+1).to(torch.long).unsqueeze(0)
                ###    batch_pred_prob = pred_prob if data_batch_idx == 0 else torch.cat([batch_pred_prob, pred_prob], dim=0)
                ###    batch_preds = preds if data_batch_idx == 0 else torch.cat([batch_preds, preds], dim=0)
                ###    batch_y_labels = y_labels if data_batch_idx == 0 else torch.cat([batch_y_labels, y_labels], dim=0)

                ###    calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": pred_prob, "loss_fn_target": y_labels,
                ###                           "preds": preds, "y_labels": y_labels, "batch_loss": batch_loss,
                ###                           "batch_edge_acc": batch_edge_acc, "epoch_metrics": epoch_metrics}
                ###    batch_loss, batch_edge_acc = self.calc_loss_fn(**calc_loss_fn_kwargs)
                ###calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_prob, "loss_fn_target": batch_y_labels,
                ###                       "preds": batch_preds, "y_labels": batch_y_labels, "batch_loss": batch_loss,
                ###                       "batch_edge_acc": batch_edge_acc, "epoch_metrics": epoch_metrics}
                batch_pred_prob, batch_preds, batch_y_labels, pred_graph_embeds, y_graph_embeds, log_model_info_data = self.infer_batch_data(batch_data, is_return_graph_embeds=True)
                calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_prob, "loss_fn_target": batch_y_labels,
                                       "preds": batch_preds, "y_labels": batch_y_labels, "num_batches": num_batches, "epoch_metrics": epoch_metrics}
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

                ### compute graph embeds
                ##pred_graph_embeds = self.get_pred_embeddings(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                ##y_graph_embeds = self.graph_encoder.get_embeddings(y, y_edge_index, y_seq_batch_node_id, y_edge_attr)
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
                #### used in observation model info in console
                ###log_model_info_data = data
            # Validation
            epoch_metrics['val_loss'], epoch_metrics['val_edge_acc'], batch_val_preds, batch_val_y_labels = self.test(val_data, loss_fns=loss_fns, show_loader_log=True if epoch_i == 0 else False)

            # record training history and save best model
            epoch_metrics["tr_preds"] = batch_preds  # only record the last batch
            epoch_metrics["tr_labels"] = batch_y_labels
            epoch_metrics["val_preds"] = batch_val_preds
            epoch_metrics["val_labels"] = batch_val_y_labels
            for k, v in epoch_metrics.items():
                history_list = best_model_info.setdefault(k+"_history", [])
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0 or (v.dim() == 1 and v.shape[0] == 1):
                        history_list.append(v.item())
                    elif v.dim() >= 2 or v.shape[0] > 1:
                        history_list.append(v.cpu().detach().numpy().tolist())
                else:
                    history_list.append(v)
            if epoch_metrics['val_edge_acc'] > best_model_info["max_val_edge_acc"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["max_val_edge_acc_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["max_val_edge_acc"] = epoch_metrics['val_edge_acc'].item()

            # Check if graph_encoder.parameters() have been updated
            assert sum(map(abs, best_model_info['gra_enc_grad_history'])) > 0, f"Sum of gradient of MTSCorrAD.graph_encoder in epoch_{epoch_i}:{sum(map(abs, best_model_info['gra_enc_grad_history']))}"
            # observe model info in console
            if epoch_i == 0:
                best_model_info["model_structure"] = str(self) + "\n" + "="*100 + "\n" + str(summary(self, log_model_info_data.x, log_model_info_data.edge_index, log_model_info_data.batch, log_model_info_data.edge_attr, self.model_cfg["output_type"], max_depth=20))
                if show_model_info:
                    logger.info(f"\nNumber of graphs:{log_model_info_data.num_graphs} in No.{batch_idx} batch, the model structure:\n{best_model_info['model_structure']}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                no_show_metrics = ["pred_gra_embeds", "y_gra_embeds", "gra_embeds_disparity", "tr_preds", "tr_labels", "val_preds", "val_labels"]
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.9f}" for k, v in epoch_metrics.items() if k not in no_show_metrics])
                logger.info(f"In Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | lr: {self.optimizer.param_groups[0]['lr']:.9f}")
            if epoch_i % 100 == 0:  # show oredictive and real adjacency matrix every 500 epochs
                obs_data_batch_idx = self.model_cfg['batch_size']-1
                logger.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:{obs_data_batch_idx} \ninput_graph_adj[:5]:\n{log_model_info_data.edge_attr[:5]}\npreds:\n{batch_preds[obs_data_batch_idx]}\ny_labels:\n{batch_y_labels[obs_data_batch_idx]}\n")
            ###if epoch_i % 100 == 0:  # show oredictive and real adjacency matrix every 500 epochs
            ###    logger.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:{data_batch_idx} \ninput_graph_adj[:5]:\n{x_edge_attr[:5]}\npreds:\n{preds}\ny_graph_adj:\n{y_graph_adj}\n")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None, show_loader_log: bool = False):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        ###num_nodes = self.model_cfg["num_nodes"]
        test_loader = self.create_pyg_data_loaders(graph_adj_mats=test_data["edges"],  graph_nodes_mats=test_data["nodes"], target_mats=test_data["target"], loader_seq_len=self.model_cfg["seq_len"], show_log=show_loader_log)
        num_batches = len(test_loader)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                ###batch_loss = torch.zeros(1)
                ###batch_edge_acc = torch.zeros(1)
                ###for data_batch_idx in range(self.model_cfg['batch_size']):
                ###    data = batch_data[data_batch_idx]
                ###    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                ###    y, y_edge_index, y_seq_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64), data.y[-1].edge_attr  # only take y of x with last time-step on training
                ###    pred_prob = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr, self.model_cfg["output_type"])
                ###    preds = torch.argmax(pred_prob, dim=1)
                ###    y_graph_adj = torch.sparse_coo_tensor(y_edge_index, y_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                ###    y_labels = (y_graph_adj.view(-1)+1).to(torch.long)
                ###    calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": pred_prob, "loss_fn_target": y_labels,
                ###                           "preds": preds, "y_labels": y_labels, "num_batches": num_batches}
                ###    batch_loss, batch_edge_acc = self.calc_loss_fn(**calc_loss_fn_kwargs)
                batch_pred_prob, batch_preds, batch_y_labels, _ = self.infer_batch_data(batch_data)
                calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_prob, "loss_fn_target": batch_y_labels,
                                       "preds": batch_preds, "y_labels": batch_y_labels, "num_batches": num_batches}
                batch_loss, batch_edge_acc = self.calc_loss_fn(**calc_loss_fn_kwargs)

                test_loss += batch_loss/num_batches
                test_edge_acc += batch_edge_acc/num_batches

        return test_loss, test_edge_acc, batch_preds, batch_y_labels
