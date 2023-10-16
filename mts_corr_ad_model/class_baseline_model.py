#!/usr/bin/env python
# coding: utf-8
import copy
import functools
import logging
from math import ceil, isclose

import numpy as np
import torch
from torch.nn import GRU, Dropout, Linear, Sequential, Softmax
from tqdm import tqdm

from baseline_model import BaselineGRU

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

        # set model components
        self.graph_size = self.model_cfg["num_nodes"]**2
        self.fc_out_dim = self.graph_size
        self.gru = GRU(input_size=self.model_cfg['gru_in_dim'], hidden_size=self.model_cfg['gru_h'], num_layers=self.model_cfg['gru_l'], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0, batch_first=True)
        self.decoder = self.model_cfg['decoder'](self.model_cfg['gru_h'], self.model_cfg["num_nodes"], drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        self.fc1 = Sequential(Linear(self.graph_size, self.fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.fc2 = Sequential(Linear(self.graph_size, self.fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.fc3 = Sequential(Linear(self.graph_size, self.fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.softmax = Softmax(dim=0)
        self.init_optimizer()

    def forward(self, x, output_type, *unused_args, **unused_kwargs):
        batch_pred_prob = torch.empty(x.shape[0], 3, self.fc_out_dim)
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

    def infer_batch_data(self, batch_data: list):
        """
        Infer batch data and return predicted labels and real labels
        """
        x, y = batch_data[0], batch_data[1]
        pred_prob = self.forward(x, output_type=self.model_cfg['output_type'])
        preds = torch.argmax(pred_prob, dim=1)
        y_labels = (y+1).to(torch.long)
        return pred_prob, preds, y_labels

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 1000, **unused_kwargs):
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self

        best_model_info = self.init_best_model_info(train_data, loss_fns, epochs)
        best_model_info.update({"max_val_edge_acc": 0})
        self.show_model_config()
        best_model = []
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1), "gru_gradient": torch.zeros(1), "decoder_gradient": torch.zeros(1)}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            train_loader = self.yield_batch_data(graph_adj_mats=train_data['edges'], target_mats=train_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
            num_batches = ceil((len(train_data['edges'])-self.model_cfg['seq_len'])/self.model_cfg['batch_size'])
            for batch_idx, batch_data in enumerate(train_loader):
                pred_prob, preds, y_labels = self.infer_batch_data(batch_data)
                calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": pred_prob, "loss_fn_target": y_labels,
                                       "preds": preds, "y_labels": y_labels, "num_batches": num_batches, "epoch_metrics": epoch_metrics}
                batch_loss, batch_edge_acc = self.calc_loss_fn(**calc_loss_fn_kwargs)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                if hasattr(self, "scheduler"):
                    self.scheduler.step()
                epoch_metrics["tr_edge_acc"] += batch_edge_acc/num_batches
                epoch_metrics["tr_loss"] += batch_loss/num_batches
                epoch_metrics["gru_gradient"] += sum([p.grad.sum() for p in self.gru.parameters() if p.grad is not None])/num_batches
                epoch_metrics["decoder_gradient"] += sum([p.grad.sum() for p in self.decoder.parameters() if p.grad is not None])/num_batches

            # Validation
            epoch_metrics['val_loss'], epoch_metrics['val_edge_acc'], batch_val_preds, batch_val_y_labels = self.test(val_data, loss_fns=loss_fns)

            # record training history and save best model
            epoch_metrics["tr_preds"] = preds  # only record the last batch
            epoch_metrics["tr_labels"] = y_labels
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
            if epoch_i == 0:
                best_model_info["model_structure"] = str(self)
            if epoch_metrics['val_edge_acc'] > best_model_info["max_val_edge_acc"]:
                best_model = copy.deepcopy(self.state_dict())
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["max_val_edge_acc_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["max_val_edge_acc"] = epoch_metrics['val_edge_acc'].item()

            if epoch_i == 0:
                logger.info(f"\nModel Structure: \n{self}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.8f}" for k, v in epoch_metrics.items() if v.dim() < 2])
                ###logger.info(f"Epoch {epoch_i:>3} | {epoch_metric_log_msgs}")
                logger.info(f"In Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | lr: {self.optimizer.param_groups[0]['lr']:.9f}")
            if epoch_i % 100 == 0:  # show oredictive and real adjacency matrix every 500 epochs
                x, y = batch_data[0], batch_data[1]
                logger.info(f"\nIn Epoch {epoch_i:>3}, batch_idx:{batch_idx}, data_batch_idx:7 \ninput_graph_adj[7, 0, :5]:\n{x[7, 0, :5]}\npred_graph_adj[7, :5]:\n{preds[7, :5]}\ny_graph_adj[7, :5]:\n{y[7, :5]}\n")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        test_loader = self.yield_batch_data(graph_adj_mats=test_data['edges'], target_mats=test_data['target'], batch_size=self.model_cfg['batch_size'], seq_len=self.model_cfg['seq_len'])
        num_batches = ceil((len(test_data['edges'])-self.model_cfg['seq_len'])/self.model_cfg['batch_size'])
        with torch.no_grad():
            for batch_data in test_loader:
                batch_pred_prob, batch_preds, batch_y_labels = self.infer_batch_data(batch_data)
                calc_loss_fn_kwargs = {"loss_fns": loss_fns, "loss_fn_input": batch_pred_prob, "loss_fn_target": batch_y_labels,
                                       "preds": batch_preds, "y_labels": batch_y_labels, "num_batches": num_batches}
                batch_loss, batch_edge_acc = self.calc_loss_fn(**calc_loss_fn_kwargs)
                test_edge_acc += batch_edge_acc/num_batches
                test_loss += batch_loss/num_batches

        return test_loss, test_edge_acc, batch_preds, batch_y_labels


class ClassBaselineGRUWithoutSelfCorr(ClassBaselineGRU):
    def __init__(self, model_cfg: dict, **unused_kwargs):
        super(ClassBaselineGRUWithoutSelfCorr, self).__init__(model_cfg)

        # set model components
        self.fc_out_dim = self.model_cfg["gru_in_dim"]
        self.fc1 = Sequential(Linear(self.graph_size, self.fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.fc2 = Sequential(Linear(self.graph_size, self.fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.fc3 = Sequential(Linear(self.graph_size, self.fc_out_dim), Dropout(self.model_cfg["drop_p"] if "class_fc" in self.model_cfg["drop_pos"] else 0))
        self.init_optimizer()

    def transform_graph_adj_to_only_triu(self, graph_adj_mats: np.ndarray):
        assert graph_adj_mats.shape[1] == graph_adj_mats.shape[2], "graph_adj_mats must be a square matrix"
        _, num_nodes, _ = graph_adj_mats.shape
        triu_idx = np.triu_indices(num_nodes, k=1)
        graph_adj_mats = graph_adj_mats[::, triu_idx[0], triu_idx[1]]
        return graph_adj_mats

    def yield_batch_data(self, graph_adj_mats: np.ndarray, target_mats: np.ndarray, seq_len: int = 10, batch_size: int = 5):
        graph_time_step = graph_adj_mats.shape[0] - 1  # the graph of last "t" can't be used as train data
        graph_adj_arr = self.transform_graph_adj_to_only_triu(graph_adj_mats)
        target_arr = self.transform_graph_adj_to_only_triu(target_mats)
        for g_t in range(0, graph_time_step, batch_size):
            cur_batch_size = batch_size if g_t+batch_size <= graph_time_step-seq_len else graph_time_step-seq_len-g_t
            if cur_batch_size <= 0: break
            batch_x = torch.empty((cur_batch_size, seq_len, self.model_cfg['gru_in_dim'])).fill_(np.nan)
            batch_y = torch.empty((cur_batch_size, self.fc_out_dim)).fill_(np.nan)
            for data_batch_idx in range(cur_batch_size):
                begin_t, end_t = g_t+data_batch_idx, g_t+data_batch_idx+seq_len
                batch_x[data_batch_idx] = torch.tensor(np.nan_to_num(graph_adj_arr[begin_t:end_t], nan=0))
                batch_y[data_batch_idx] = torch.tensor(np.nan_to_num(target_arr[end_t], nan=0))

            assert not torch.isnan(batch_x).any() or not torch.isnan(batch_y).any(), "batch_x or batch_y contains nan"

            yield batch_x, batch_y


class ClassBaselineGRUOneFeature(ClassBaselineGRUWithoutSelfCorr):
    """
    Only use one feature of graph adjacency matrix as input
    """
    def init_best_model_info(self, train_data: np.ndarray, loss_fns: dict, epochs: int):
        """
        Initialize best_model_info for ClassBaselineGRUOneFeature
        """
        best_model_info = super().init_best_model_info(train_data, loss_fns, epochs)
        best_model_info.update({"input_feature_idx": self.model_cfg["input_feature_idx"][0]})
        return best_model_info

    def yield_batch_data(self, graph_adj_mats: np.ndarray, target_mats: np.ndarray, seq_len: int = 10, batch_size: int = 5):
        assert self.model_cfg["input_feature_idx"] is not None, "input_feature_idx must be set"
        assert len(self.model_cfg["input_feature_idx"]) == 1, "input_feature_idx must be a list with only one element"
        selected_feature_idx = self.model_cfg["input_feature_idx"][0]
        graph_time_step = graph_adj_mats.shape[0] - 1  # the graph of last "t" can't be used as train data
        graph_adj_arr = self.transform_graph_adj_to_only_triu(graph_adj_mats)[::, selected_feature_idx].reshape(-1, 1)
        target_arr = self.transform_graph_adj_to_only_triu(target_mats)[::, selected_feature_idx].reshape(-1, 1)
        for g_t in range(0, graph_time_step, batch_size):
            cur_batch_size = batch_size if g_t+batch_size <= graph_time_step-seq_len else graph_time_step-seq_len-g_t
            if cur_batch_size <= 0: break
            batch_x = torch.empty((cur_batch_size, seq_len, self.model_cfg['gru_in_dim'])).fill_(np.nan)
            batch_y = torch.empty((cur_batch_size, self.fc_out_dim)).fill_(np.nan)
            for data_batch_idx in range(cur_batch_size):
                begin_t, end_t = g_t+data_batch_idx, g_t+data_batch_idx+seq_len
                batch_x[data_batch_idx] = torch.tensor(np.nan_to_num(graph_adj_arr[begin_t:end_t], nan=0))
                batch_y[data_batch_idx] = torch.tensor(np.nan_to_num(target_arr[end_t], nan=0))

            assert not torch.isnan(batch_x).any() or not torch.isnan(batch_y).any(), "batch_x or batch_y contains nan"

            yield batch_x, batch_y


class ClassBaselineGRUCustomFeatures(ClassBaselineGRUWithoutSelfCorr):
    """
    Only use selected features of graph adjacency matrix as input
    """
    def init_best_model_info(self, train_data: np.ndarray, loss_fns: dict, epochs: int):
        """
        Initialize best_model_info for ClassBaselineGRUCustomFeatures
        """
        best_model_info = super().init_best_model_info(train_data, loss_fns, epochs)
        best_model_info.update({"input_feature_idx": sorted(self.model_cfg["input_feature_idx"])})
        return best_model_info

    def yield_batch_data(self, graph_adj_mats: np.ndarray, target_mats: np.ndarray, seq_len: int = 10, batch_size: int = 5):
        assert self.model_cfg["input_feature_idx"] is not None, "input_feature_idx must be set"
        assert len(self.model_cfg["input_feature_idx"]) > 1, "input_feature_idx must be a list with more than one element"
        selected_feature_idx = self.model_cfg["input_feature_idx"]
        graph_time_step = graph_adj_mats.shape[0] - 1  # the graph of last "t" can't be used as train data
        graph_adj_arr = self.transform_graph_adj_to_only_triu(graph_adj_mats)[::, selected_feature_idx].reshape(-1, len(selected_feature_idx))
        target_arr = self.transform_graph_adj_to_only_triu(target_mats)[::, selected_feature_idx].reshape(-1, len(selected_feature_idx))
        for g_t in range(0, graph_time_step, batch_size):
            cur_batch_size = batch_size if g_t+batch_size <= graph_time_step-seq_len else graph_time_step-seq_len-g_t
            if cur_batch_size <= 0: break
            batch_x = torch.empty((cur_batch_size, seq_len, self.model_cfg['gru_in_dim'])).fill_(np.nan)
            batch_y = torch.empty((cur_batch_size, self.fc_out_dim)).fill_(np.nan)
            for data_batch_idx in range(cur_batch_size):
                begin_t, end_t = g_t+data_batch_idx, g_t+data_batch_idx+seq_len
                batch_x[data_batch_idx] = torch.tensor(np.nan_to_num(graph_adj_arr[begin_t:end_t], nan=0))
                batch_y[data_batch_idx] = torch.tensor(np.nan_to_num(target_arr[end_t], nan=0))

            assert not torch.isnan(batch_x).any() or not torch.isnan(batch_y).any(), "batch_x or batch_y contains nan"

            yield batch_x, batch_y
