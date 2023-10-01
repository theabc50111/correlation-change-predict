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


class MTSCorrAD3(MTSCorrAD):
    """
    Multi-Time Series Correlation Anomaly Detection3 (MTSCorrAD3)
    Structure of MTSCorrAD3:
        GRU -> GraphEncoder -> Decoder
    """
    def __init__(self, model_cfg: dict):
        super(MTSCorrAD3, self).__init__(model_cfg)
        self.model_cfg = model_cfg
        # set model components
        num_nodes = self.model_cfg['num_nodes']
        num_edges = num_nodes**2
        num_edge_features = self.model_cfg['num_edge_features']
        graph_enc_emb_size = self.graph_encoder.gra_enc_l * self.graph_encoder.gra_enc_h  # the input size of GRU depend on the number of layers of GINconv
        self.gru1_edges = GRU(num_edges*num_edge_features, num_edges*num_edge_features, self.model_cfg["gru_l"], dropout=self.model_cfg["drop_p"] if "gru" in self.model_cfg["drop_pos"] else 0)
        self.decoder = self.model_cfg['decoder'](graph_enc_emb_size, num_nodes, drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        del self.gru1
        self.init_optimizer()

    def forward(self, x, edge_index, seq_batch_node_id, edge_attr, output_type, *unused_args):
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

        # Inter-series modeling
        if type(self.graph_encoder).__name__ == "GinEncoder":
            graph_embeds = self.graph_encoder(x, seq_batch_strong_connect_edge_index, seq_batch_node_id)
        elif type(self.graph_encoder).__name__ == "GineEncoder":
            graph_embeds = self.graph_encoder(x, seq_batch_strong_connect_edge_index, seq_batch_node_id, temporal_edge_attr)

        # Decoder (Graph Adjacency Reconstruction)
        pred_graph_adj = self.decoder(graph_embeds[-1])  # graph_embeds[-1] => only take last time-step

        if output_type == "discretize":
            bins = torch.tensor(self.model_cfg['output_bins']).reshape(-1, 1)
            num_bins = len(bins)-1
            bins = torch.concat((bins[:-1], bins[1:]), dim=1)
            discretize_values = np.linspace(-1, 1, num_bins)
            for lower, upper, discretize_value in zip(bins[:, 0], bins[:, 1], discretize_values):
                pred_graph_adj = torch.where((pred_graph_adj <= upper) & (pred_graph_adj > lower), discretize_value, pred_graph_adj)
            pred_graph_adj = torch.where(pred_graph_adj < bins.min(), bins.min(), pred_graph_adj)

        return pred_graph_adj

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

            # Inter-series modeling
            if type(self.graph_encoder).__name__ == "GinEncoder":
                pred_graph_embeds = self.graph_encoder(x, seq_batch_strong_connect_edge_index, seq_batch_node_id)
            elif type(self.graph_encoder).__name__ == "GineEncoder":
                pred_graph_embeds = self.graph_encoder(x, seq_batch_strong_connect_edge_index, seq_batch_node_id, temporal_edge_attr)


        return pred_graph_embeds[-1]
