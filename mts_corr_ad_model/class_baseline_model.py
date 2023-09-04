#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import functools
import json
import logging
import sys
from collections import OrderedDict
from datetime import datetime
from math import ceil, sqrt
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import numpy as np
import torch
import yaml
from torch.nn import (GRU, BatchNorm1d, Dropout, Linear, MSELoss, Sequential,
                      Softmax)
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from tqdm import tqdm

sys.path.append("/workspace/correlation-change-predict/utils")
from utils import split_and_norm_data

from baseline_model import BaselineGRU
from encoder_decoder import MLPDecoder, ModifiedInnerProductDecoder

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


class ClassBaselineGRU(BaselineGRU):
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(ClassBaselineGRU, self).__init__(model_cfg)
        self.model_cfg = model_cfg
        # create data loader
        self.num_tr_batches = self.model_cfg["num_batches"]['train']

        # set model components
        self.gru = GRU(input_size=self.model_cfg['gru_in_dim'], hidden_size=self.model_cfg['gru_h'], num_layers=self.model_cfg['gru_l'], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0, batch_first=True)
        self.decoder = self.model_cfg['decoder'](self.model_cfg['gru_h'], self.model_cfg["num_nodes"], drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        self.fc1 = Sequential(Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.fc2 = Sequential(Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.fc3 = Sequential(Linear(self.model_cfg["num_nodes"]**2, self.model_cfg["num_nodes"]**2), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.softmax = Softmax(dim=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.num_tr_batches*50, gamma=0.5)


    def forward(self, x, output_type, *unused_args, **unused_kwargs):
        batch_pred_prob = torch.empty(x.shape[0], 3, self.model_cfg['num_nodes']**2)
        gru_output, gru_hn = self.gru(x)
        # Decoder (Graph Adjacency Reconstruction)
        for data_batch_idx in range(x.shape[0]):
            pred_graph_adj = self.decoder(gru_output[data_batch_idx, -1, :])  # gru_output[-1] => only take last time-step
            flatten_pred_graph_adj = pred_graph_adj.reshape(1, -1)
            fc1_output = self.fc1(flatten_pred_graph_adj)
            fc2_output = self.fc2(flatten_pred_graph_adj)
            fc3_output = self.fc3(flatten_pred_graph_adj)
            logits = torch.cat([fc1_output, fc2_output, fc3_output], dim=0)
            pred_prob = self.softmax(logits)
            batch_pred_prob[data_batch_idx] = pred_prob

        return batch_pred_prob

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 1000, **unused_kwargs):
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self

        num_batches = ceil(len(train_data['edges'])//self.model_cfg['batch_size'])+1
        self.show_model_struture()
        best_model_info = self.init_best_model_info(train_data, loss_fns, epochs)
        best_model = []
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1), "gru_gradient": torch.zeros(1), "decoder_gradient": torch.zeros(1)}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            batch_data_generator = self.yield_batch_data(graph_adj_mats=train_data['edges'], target_mats=train_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
            for batch_data in batch_data_generator:
                batch_loss = torch.zeros(1)
                batch_edge_acc = torch.zeros(1)
                x, y = batch_data[0], batch_data[1]
                pred_prob = self.forward(x, output_type=self.model_cfg['output_type'])
                preds = torch.argmax(pred_prob, dim=1)
                y_labels = (y+1).to(torch.long)
                for fn in loss_fns["fns"]:
                    fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                    loss_fns["fn_args"][fn_name].update({"input": pred_prob, "target": y_labels})
                    partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                    loss = partial_fn()
                    batch_loss += loss
                    epoch_metrics[fn_name] += loss/num_batches  # we don't reset epoch[fn_name] to 0 in each batch, so we need to divide by total number of batches
                    if "EdgeAcc" in fn_name:
                        edge_acc = 1-loss
                        batch_edge_acc += edge_acc
                    else:
                        edge_acc = (preds == y_labels).to(torch.float).mean()
                        batch_edge_acc += edge_acc

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_metrics["tr_edge_acc"] += edge_acc/num_batches
                epoch_metrics["tr_loss"] += batch_loss/num_batches
                epoch_metrics["gru_gradient"] += sum([p.grad.sum() for p in self.gru.parameters() if p.grad is not None])/num_batches
                epoch_metrics["decoder_gradient"] += sum([p.grad.sum() for p in self.decoder.parameters() if p.grad is not None])/num_batches

            # Validation
            epoch_metrics['val_loss'], epoch_metrics['val_edge_acc'], val_preds, val_y_labels = self.test(val_data, loss_fns=loss_fns)

            # record training history and save best model
            epoch_metrics["tr_last_batch_preds"] = preds
            epoch_metrics["tr_last_batch_labels"] = y_labels
            epoch_metrics["val_preds"] = val_preds
            epoch_metrics["val_labels"] = val_y_labels
            for k, v in epoch_metrics.items():
                history_list = best_model_info.setdefault(k+"_history", [])
                if v.dim() < 2:
                    history_list.append(v.item())
                else:
                    history_list.append(v.cpu().detach().numpy().tolist())
            if epoch_i == 0:
                best_model_info["model_structure"] = str(self)
            if epoch_metrics['val_loss'] < best_model_info["min_val_loss"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["min_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["min_val_loss_edge_acc"] = epoch_metrics['val_edge_acc'].item()

            if epoch_i == 0:
                logger.info(f"\nModel Structure: \n{self}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                no_show_metrics = ["tr_last_batch_preds", "tr_last_batch_labels", "val_preds", "val_labels"]
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.8f}" for k, v in epoch_metrics.items() if k not in no_show_metrics])
                logger.info(f"Epoch {epoch_i:>3} | {epoch_metric_log_msgs}")
            if epoch_i % 500 == 0:  # show oredictive and real adjacency matrix every 500 epochs
                logger.info(f"\nIn Epoch {epoch_i:>3}, data_batch_idx:7 \ninput_graph_adj[7, 0, :5]:\n{x[7, 0, :5]}\npred_graph_adj[7, :5]:\n{preds[7, :5]}\ny_graph_adj[7, :5]:\n{y[7, :5]}\n")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        with torch.no_grad():
            batch_data_generator = self.yield_batch_data(graph_adj_mats=test_data['edges'], target_mats=test_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
            num_batches = ceil(len(test_data['edges'])//self.model_cfg['batch_size'])+1
            for batch_data in batch_data_generator:
                batch_loss = torch.zeros(1)
                batch_edge_acc = torch.zeros(1)
                x, y = batch_data[0], batch_data[1]
                pred_prob = self.forward(x, output_type=self.model_cfg['output_type'])
                preds = torch.argmax(pred_prob, dim=1)
                y_labels = (y+1).to(torch.long)
                for fn in loss_fns["fns"]:
                    fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                    loss_fns["fn_args"][fn_name].update({"input": pred_prob, "target": y_labels})
                    partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                    loss = partial_fn()
                    batch_loss += loss
                    if "EdgeAcc" in fn_name:
                        edge_acc = 1-loss
                        batch_edge_acc += edge_acc
                    else:
                        edge_acc = (preds == y_labels).to(torch.float).mean()
                        batch_edge_acc += edge_acc
                print(f"val_pred.shape: {preds.shape}, val_y.shape: {y_labels.shape}")

                test_edge_acc += edge_acc / num_batches
                test_loss += batch_loss / num_batches

        return test_loss, test_edge_acc, preds, y_labels
