#!/usr/bin/env python
# coding: utf-8
from itertools import chain, repeat
from math import sqrt
from typing import List

import torch
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
from torch_geometric.nn.models.autoencoder import InnerProductDecoder


# Multi-Dimension Time-Series Correlation Anomly Detection Model
class GineEncoder(torch.nn.Module):
    """
    num_node_features: number of features per node in the graph, in this model every node has same size of features
    gra_enc_l: Number of layers of GINEconv
    gra_enc_h: output size of hidden layer of GINEconv
    """
    def __init__(self, gra_enc_l: int, gra_enc_h: int, gra_enc_aggr: str, gra_enc_mlp_l: int, num_node_features: int, num_edge_features: int, drop_p: float, drop_pos: List[str], **unused_kwargs):
        super(GineEncoder, self).__init__()
        self.gra_enc_l = gra_enc_l
        self.gine_convs = torch.nn.ModuleList()
        self.gra_enc_h = gra_enc_h

        one_layer_mlp = [Linear(gra_enc_h, gra_enc_h), BatchNorm1d(gra_enc_h), ReLU()]
        first_gin_mlp = [Linear(num_node_features, gra_enc_h), BatchNorm1d(gra_enc_h), ReLU()] + list(chain.from_iterable(repeat(one_layer_mlp, gra_enc_mlp_l-1))) + [Dropout(p=drop_p if "graph_encoder" in drop_pos else 0.0)]
        gin_mlp = list(chain.from_iterable(repeat(one_layer_mlp, gra_enc_mlp_l))) + [Dropout(p=drop_p if "graph_encoder" in drop_pos else 0.0)]
        for i in range(self.gra_enc_l):
            if i:
                nn = Sequential(*gin_mlp)

            else:
                nn = Sequential(*first_gin_mlp)
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
    def __init__(self, num_node_features: int, gra_enc_l: int, gra_enc_h: int, gra_enc_aggr: str, gra_enc_mlp_l: int, drop_p: float, drop_pos: List[str], **unused_kwargs):
        super(GinEncoder, self).__init__()
        self.gra_enc_l = gra_enc_l
        self.gin_convs = torch.nn.ModuleList()
        self.gra_enc_h = gra_enc_h

        one_layer_mlp = [Linear(gra_enc_h, gra_enc_h), BatchNorm1d(gra_enc_h), ReLU()]
        first_gin_mlp = [Linear(num_node_features, gra_enc_h), BatchNorm1d(gra_enc_h), ReLU()] + list(chain.from_iterable(repeat(one_layer_mlp, gra_enc_mlp_l-1))) + [Dropout(p=drop_p if "graph_encoder" in drop_pos else 0.0)]
        gin_mlp = list(chain.from_iterable(repeat(one_layer_mlp, gra_enc_mlp_l))) + [Dropout(p=drop_p if "graph_encoder" in drop_pos else 0.0)]
        for i in range(self.gra_enc_l):
            if i:
                nn = Sequential(*gin_mlp)

            else:
                nn = Sequential(*first_gin_mlp)
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


class ModifiedInnerProductDecoder(InnerProductDecoder):
    """
    Inner product decoder layer. Modified to insert fc layer before forward_all.
    """
    def __init__(self, fc1_in_dim: int, fc1_out_dim: int, drop_p: float = 0.0):
        super(ModifiedInnerProductDecoder, self).__init__()
        self.fc1 = Linear(fc1_in_dim, fc1_out_dim)
        self.dropout = Dropout(p=drop_p)  # Set dropout probability to 0.2
        self.fc1_is_dropout = drop_p > 0.0

    def forward(self, enc_output: torch.Tensor, has_sigmoid: bool = False):
        fc1_output = self.fc1(enc_output)
        fc1_output = self.dropout(fc1_output) if self.fc1_is_dropout else fc1_output
        z = fc1_output.reshape(-1, 1)
        pred_graph_adj = self.forward_all(z, sigmoid=has_sigmoid)
        return pred_graph_adj


class MLPDecoder(torch.nn.Module):
    """
    Multi-layer perceptron decoder layer.
    """
    def __init__(self, fc1_in_dim: int, sqrt_fc1_out_dim: int, drop_p: float = 0.0):
        super(MLPDecoder, self).__init__()
        self.fc1_out_dim = sqrt_fc1_out_dim**2
        self.fc1 = Linear(fc1_in_dim, self.fc1_out_dim)
        self.dropout = Dropout(p=drop_p)
        self.fc1_is_dropout = drop_p > 0.0

    def forward(self, enc_output: torch.Tensor, has_sigmoid: bool = False):
        fc1_output = self.fc1(enc_output)
        fc1_output = self.dropout(fc1_output) if self.fc1_is_dropout else fc1_output
        pred_graph_adj = fc1_output.reshape(int(sqrt(self.fc1_out_dim)), -1)
        return pred_graph_adj
