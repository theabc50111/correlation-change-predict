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
from torch.nn import GRU, BatchNorm1d, Dropout, Linear, MSELoss, Softmax
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from tqdm import tqdm

sys.path.append("/workspace/correlation-change-predict/utils")
from utils import split_and_norm_data

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


class ClassBaselineGRU(torch.nn.Module):
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(ClassBaselineGRU, self).__init__()
        self.model_cfg = model_cfg
        # create data loader
        self.num_tr_batches = self.model_cfg["num_batches"]['train']

        # set model components
        self.gru = GRU(input_size=self.model_cfg['gru_in_dim'], hidden_size=self.model_cfg['gru_h'], num_layers=self.model_cfg['gru_l'], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0, batch_first=True)
        self.decoder = self.model_cfg['decoder'](self.model_cfg['gru_h'], self.model_cfg["num_nodes"], drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        self.fc1 = Linear(self.model_cfg['num_nodes']**2, self.model_cfg['num_nodes']**2)
        self.fc2 = Linear(self.model_cfg['num_nodes']**2, self.model_cfg['num_nodes']**2)
        self.fc3 = Linear(self.model_cfg['num_nodes']**2, self.model_cfg['num_nodes']**2)
        self.softmax = Softmax(dim=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.num_tr_batches*50, gamma=0.5)
        observe_model_cfg = {item[0]: item[1] for item in self.model_cfg.items() if item[0] != 'dataset'}
        observe_model_cfg['optimizer'] = str(self.optimizer)
        observe_model_cfg['scheduler'] = {"scheduler_name": str(self.scheduler.__class__.__name__)}

        logger.info(f"\nModel Configuration: \n{observe_model_cfg}")


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

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 1000):
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self

        num_batches = ceil(len(train_data['edges'])//self.model_cfg['batch_size'])+1
        best_model_info = {"num_training_graphs": len(train_data['edges']),
                           "filt_mode": self.model_cfg['filt_mode'],
                           "filt_quan": self.model_cfg['filt_quan'],
                           "quan_discrete_bins": self.model_cfg['quan_discrete_bins'],
                           "custom_discrete_bins": self.model_cfg['custom_discrete_bins'],
                           "graph_nodes_v_mode": self.model_cfg['graph_nodes_v_mode'],
                           "batches_per_epoch": num_batches,
                           "epochs": epochs,
                           "batch_size": self.model_cfg['batch_size'],
                           "seq_len": self.model_cfg['seq_len'],
                           "optimizer": str(self.optimizer),
                           "opt_scheduler": {},
                           "loss_fns": str([fn.__name__ if hasattr(fn, '__name__') else str(fn) for fn in loss_fns["fns"]]),
                           "drop_pos": self.model_cfg["drop_pos"],
                           "drop_p": self.model_cfg["drop_p"],
                           "min_val_loss": float('inf'),
                           "output_type": self.model_cfg['output_type'],
                           "output_bins": '_'.join((str(f) for f in self.model_cfg['output_bins'])).replace('.', '') if self.model_cfg['output_bins'] else None,
                           "target_mats_bins": self.model_cfg['target_mats_bins'],
                           "edge_acc_loss_atol": self.model_cfg['edge_acc_loss_atol'],
                           "use_bin_edge_acc_loss": self.model_cfg['use_bin_edge_acc_loss']}
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
            epoch_metrics['val_loss'], epoch_metrics['val_edge_acc'] = self.test(val_data, loss_fns=loss_fns)

            # record training history and save best model
            for k, v in epoch_metrics.items():
                history_list = best_model_info.setdefault(k+"_history", [])
                history_list.append(v.item())
            if epoch_i == 0:
                best_model_info["model_structure"] = str(self)
            if epoch_metrics['val_loss'] < best_model_info["min_val_loss"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["min_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["min_val_loss_edge_acc"] = epoch_metrics['val_edge_acc'].item()

            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.8f}" for k, v in epoch_metrics.items()])
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

                test_edge_acc += edge_acc / num_batches
                test_loss += batch_loss / num_batches

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
    def yield_batch_data(graph_adj_mats: np.ndarray, target_mats: np.ndarray, seq_len: int = 10, batch_size: int = 5):
        graph_time_step = graph_adj_mats.shape[0] - 1  # the graph of last "t" can't be used as train data
        _, num_nodes, _ = graph_adj_mats.shape
        graph_adj_arr = graph_adj_mats.reshape(graph_time_step+1, -1)
        target_arr = target_mats.reshape(graph_time_step+1, -1)
        for g_t in range(0, graph_time_step, batch_size):
            cur_batch_size = batch_size if g_t+batch_size <= graph_time_step-seq_len else graph_time_step-seq_len-g_t
            if cur_batch_size <= 0: break
            batch_x = torch.empty((cur_batch_size, seq_len, num_nodes**2)).fill_(np.nan)
            batch_y = torch.empty((cur_batch_size, num_nodes**2)).fill_(np.nan)
            for data_batch_idx in range(cur_batch_size):
                begin_t, end_t = g_t+data_batch_idx, g_t+data_batch_idx+seq_len
                batch_x[data_batch_idx] = torch.tensor(np.nan_to_num(graph_adj_arr[begin_t:end_t], nan=0))
                batch_y[data_batch_idx] = torch.tensor(np.nan_to_num(target_arr[end_t], nan=0))

            assert not torch.isnan(batch_x).any() or not torch.isnan(batch_y).any(), "batch_x or batch_y contains nan"

            yield batch_x, batch_y
