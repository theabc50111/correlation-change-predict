#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
from pathlib import Path
import warnings
import sys
import logging
from pprint import pformat
from itertools import product, islice
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib as mpl
import torch
from torch.nn import Linear, GRU, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, summary
# from torchsummary import summary
import dynamic_yaml
import yaml

sys.path.append("/workspace/correlation-change-predict/ywt_library")
import data_module
from data_module import data_gen_cfg, gen_corr_mat_thru_t


with open('../config/data_config.yaml') as f:
    data = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data))

warnings.simplefilter("ignore")
logging.basicConfig(format='%(levelname)-8s [%(filename)s] %(message)s',
                    level=logging.INFO)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.ERROR)
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
# logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# loggin.debug(logger_list)

# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501
logging.debug(pformat(data_cfg, indent=1, width=100, compact=True))
logging.info(pformat(data_gen_cfg, indent=1, width=100, compact=True))


# ## Data implement & output setting & testset setting

# In[2]:


# data implement setting
data_implement = "SP500_20082017_CORR_SER_REG_CORR_MAT_HRCHY_11_CLUSTER"  # watch options by operate: logging.info(data_cfg["DATASETS"].keys())
# train set setting
train_items_setting = "-train_train"  # -train_train|-train_all
# setting of name of output files and pictures title
output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
logging.info(f"===== file_name basis:{output_file_name} =====")


# In[3]:


s_l, w_l = data_gen_cfg["CORR_STRIDE"], data_gen_cfg["CORR_WINDOW"]
graph_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-graph_data"
model_dir = Path(f'./save_models/{output_file_name}/corr_s{s_l}_w{w_l}')
model_log_dir = Path(f'./save_models/{output_file_name}/corr_s{s_l}_w{w_l}/train_logs/')
model_dir.mkdir(parents=True, exist_ok=True)
model_log_dir.mkdir(parents=True, exist_ok=True)


# ## model configuration

# In[4]:


gin_enc_cfg = {"num_gin_layers": 1,  # range:1~n, for GIN after the second layer,
               "gin_dim_h": 3,
              }
data_loader_cfg = {"tr_loader_batch_size": 12,  # each graph contains 5 days correlation, so 4 graphs means a month, 12 graphs means a quarter
                   "val_loader_batch_size": 4,  # each graph contains 5 days correlation, so 4 graphs means a month, 12 graphs means a quarter
                   "test_loader_batch_size": 4,  # each graph contains 5 days correlation, so 4 graphs means a month, 12 graphs means a quarter
                  }
mts_corr_ad_cfg = {"gru_layers": 1,  # range:1~n, for gru
                   "gru_dim_out": 8,
                   }
mts_corr_ad_cfg["dim_out"] = gin_enc_cfg["num_gin_layers"] *  gin_enc_cfg["gin_dim_h"]


# ## Load Graph Data

# In[5]:


graph_arr = np.load(graph_data_dir/f"corr_s{s_l}_w{w_l}_graph.npy")  # each graph consist of 66 node & 66^2 edges
logging.info(f"graph_arr.shape:{graph_arr.shape}")
graph_time_step = graph_arr.shape[0] - 1  # the graph of last "t" can't be used as train data
node_attr = torch.tensor(np.zeros((graph_arr.shape[1], 1)), dtype=torch.float32)  # each node has only one attribute
edge_index = torch.tensor(list(product(range(graph_arr.shape[1]), repeat=2)))
dataset = []
for g_t in range(graph_time_step):
    edge_attr = torch.tensor(np.hstack(graph_arr[g_t]).reshape(-1, 1), dtype=torch.float32)
    edge_attr_next_t = torch.tensor(np.hstack(graph_arr[g_t+1]).reshape(-1, 1), dtype=torch.float32)
    data_y = Data(x=node_attr, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr_next_t)
    data = Data(x=node_attr, y=data_y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
    dataset.append(data)
else:
    #mts_corr_ad_cfg["dim_out"] = data.y.shape[0]  # if the input of loss-function graphs instead of graphs' embedding
    gin_enc_cfg["num_node_features"] = data.num_node_features
    logging.info(f"data.num_node_features: {data.num_node_features}; data.num_edges: {data.num_edges}; data.num_edge_features: {data.num_edge_features}; data.is_undirected: {data.is_undirected()}; ")
    logging.info(f"data.x.shape: {data.x.shape}; data.y.x.shape: {data.y.x.shape}; data.edge_index.shape: {data.edge_index.shape}; data.edge_attr.shape: {data.edge_attr.shape}")

# Create training, validation, and test sets
train_dataset = dataset[:int(len(dataset)*0.9)]

import pickle
with open("../../tmp/tmp_torch_graph_dataset.pickle", 'wb') as handle:
    pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
val_dataset   = dataset[int(len(dataset)*0.9):int(len(dataset)*0.95)]
test_dataset  = dataset[int(len(dataset)*0.95):]

# Create mini-batches
train_loader = DataLoader(train_dataset, batch_size = data_loader_cfg["tr_loader_batch_size"], shuffle=False)
val_loader = DataLoader(val_dataset, batch_size = data_loader_cfg["val_loader_batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = data_loader_cfg["test_loader_batch_size"], shuffle=False)

# show info
logging.info(f'Training set   = {len(train_dataset)} graphs')
logging.info(f'Validation set = {len(val_dataset)} graphs')
logging.info(f'Test set       = {len(test_dataset)} graphs')
logging.debug('Train loader:')
for i, subgraph in enumerate(train_loader):
    logging.debug(f' - Subgraph {i}: {subgraph} ; Subgraph {i}.num_graphs:{subgraph.num_graphs}')

logging.debug('Validation loader:')
for i, subgraph in enumerate(val_loader):
    logging.debug(f' - Subgraph {i}: {subgraph} ; Subgraph{i}.num_graphs:{subgraph.num_graphs}')

logging.debug('Test loader:')
for i, subgraph in enumerate(test_loader):
    logging.debug(f' - Subgraph {i}: {subgraph} ; Subgraph{i}.num_graphs:{subgraph.num_graphs}')

logging.debug('Peeking Train data:')
data_x_nodes = next(iter(train_loader)).x.reshape(train_loader.batch_size, -1)
data_x_edges = next(iter(train_loader)).edge_attr.reshape(train_loader.batch_size, -1)
data_y_nodes = torch.cat([y.x for y in next(iter(train_loader)).y]).reshape(train_loader.batch_size, -1)
data_y_edges = torch.cat([y.edge_attr for y in next(iter(train_loader)).y]).reshape(train_loader.batch_size, -1)
for i in range(12):
    logging.debug(f"\n batch0_x{i}.shape: {data_x_nodes[i].shape} \n batch0_x{i}[:5]:{data_x_nodes[i][:5]}")
    logging.debug(f"\n batch0_x{i}_edges.shape: {data_x_edges[i].shape} \n batch0_x{i}_edges[:5]:{data_x_edges[i][:5]}")
    logging.debug(f"\n batch0_y{i}.shape: {data_y_nodes[i].shape} \n batch0_y{i}[:5]:{data_y_nodes[i][:5]}")
    logging.debug(f"\n batch0_y{i}_edges.shape: {data_y_edges[i].shape} \n batch0_y{i}_edges[:5]:{data_y_edges[i][:5]}")


# ## Multi-Dimension Time-Series Correlation Anomly Detection Model

# In[6]:


class GinEncoder(torch.nn.Module):
    """
    num_node_features: number of features per node in the graph, in this model every node has same size of features 
    gin_dim_h: output size of hidden layer of GINconv
    gru_layers: Number of recurrent layers of GRU
    """
    def __init__(self, num_node_features:int, num_gin_layers:int, gin_dim_h:int, **kwargs):
        super(GinEncoder, self).__init__()
        self.num_gin_layers = num_gin_layers
        self.gin_convs = torch.nn.ModuleList()
        self.gin_dim_h = gin_dim_h

        for i in range(num_gin_layers):
            if i:
                nn = Sequential(Linear(gin_dim_h, gin_dim_h),
                                BatchNorm1d(gin_dim_h), ReLU(),
                                Linear(gin_dim_h, gin_dim_h), ReLU())
            else:
                nn = Sequential(Linear(num_node_features, gin_dim_h),
                                BatchNorm1d(gin_dim_h), ReLU(),
                                Linear(gin_dim_h, gin_dim_h), ReLU())
            self.gin_convs.append(GINConv(nn))


    def forward(self, x, edge_index, batch_node_id):
        # Node embeddings
        nodes_emb_layers = []
        for i in range(self.num_gin_layers):
            if i:
                nodes_emb = self.gin_convs[i](nodes_emb, edge_index)
            else:
                nodes_emb = self.gin_convs[i](x, edge_index)  # the shape of nodes_embeds: [batch_size*num_nodes, gin_dim_h] 
            nodes_emb_layers.append(nodes_emb)

        # Graph-level readout
        nodes_emb_pools = [global_add_pool(nodes_emb, batch_node_id) for nodes_emb in nodes_emb_layers]  # the shape of global_add_pool(nodes_emb, batch_node_id): [batch_size, gin_dim_h]
                                                                                                         # global_add_pool : make a super-node to represent the graph
        # Concatenate and form the graph embeddings
        graph_embeds = torch.cat(nodes_emb_pools, dim=1)  # the shape of graph_embeds: [batch_size, num_layers*gin_dim_h]

        return graph_embeds


    def get_embeddings(self, x, edge_index, batch_node_id):
        with torch.no_grad():
            graph_embeds = self.forward(x, edge_index, batch_node_id).reshape(-1)

        return graph_embeds

class MTSCorrAD(torch.nn.Module):
    """
    gru_dim_out: The number of output size of GRU and features in the hidden state h of GRU
    dim_out: The number of output size of MTSCorrAD model
    """
    def __init__(self, graph_encoder:torch.nn.Module, gru_layers:int, gru_dim_out:int, dim_out:int, **kwargs):
        super(MTSCorrAD, self).__init__()
        self.graph_encoder = GinEncoder(**gin_enc_cfg).to("cuda")
        gru_input_size = self.graph_encoder.num_gin_layers * self.graph_encoder.gin_dim_h  # the input size of GRU depend on the number of layers of GINconv
        self.gru1 = GRU(gru_input_size, gru_dim_out, gru_layers)
        self.lin1 = Linear(gru_dim_out, dim_out)


    def forward(self, x, edge_index, batch_node_id):
        # Inter-series modeling
        graph_embeds = self.graph_encoder(x, edge_index, batch_node_id)

        # Temporal Modeling
        gru_output, gru_hn = self.gru1(graph_embeds)  # regarding batch_size as time-steps(sequence length) by using "unbatched" input
        graph_embed_pred = self.lin1(gru_output[-1])  # gru_output[-1] => only take last time-step

        return graph_embed_pred


# In[7]:


def train(model:torch.nn.Module, train_loader:torch_geometric.loader.dataloader.DataLoader,
          val_loader:torch_geometric.loader.dataloader.DataLoader, optimizer, criterion, epochs:int =5, show_model_info=False):
    best_model_info = {"epochs": epochs,
                       "train_batch": train_loader.batch_size,
                       "val_batch": val_loader.batch_size,
                       "optimizer": optimizer.__str__(),
                       "criterion": criterion.__str__(),
                       "min_val_loss": float('inf'),
                       "train_loss_history": [],
                       "val_loss_history": [],
                      }
    model_num_layers = sum(1 for _ in model.parameters())
    graph_enc_num_layers =  sum(1 for _ in model.graph_encoder.parameters())
    graph_enc_w_grad_after = 0
    for epoch_i in tqdm(range(epochs)):
        model.train()
        train_loss = 0

        # Train on batches
        for batch_i, data in enumerate(train_loader):
            data.to("cuda")
            x, x_edge_index, x_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            y, y_edge_index, y_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64).to("cuda"), data.y[-1].edge_attr  # only take y of x with last time-step on training
            optimizer.zero_grad()
            graph_embeds_pred = model(x, x_edge_index, x_batch_node_id)
            y_graph_embeds = model.graph_encoder.get_embeddings(y, y_edge_index, y_batch_node_id)
            loss =  criterion(graph_embeds_pred, y_graph_embeds)
            train_loss += loss / len(train_loader)
            loss.backward()
            graph_enc_w_grad_after += sum(sum(torch.abs(torch.reshape(p.grad if p.grad!=None else torch.zeros((1,)).to('cuda'), (-1,)))) for p in islice(model.graph_encoder.parameters(), 0, graph_enc_num_layers))
            optimizer.step()

        # Check if graph_encoder.parameters() have been updated in each epoch
        assert graph_enc_w_grad_after>0, f"After loss.backward(), Sum of MainModel.graph_encoder weights in epoch_{epoch_i}:{graph_enc_w_grad_after}"

        # Validation
        val_loss = test(model, val_loader, criterion)

        # save best model
        if val_loss<best_model_info["min_val_loss"]:
            best_model = model
            best_model_info["best_val_epoch"] = epoch_i
            best_model_info["min_val_loss"] = val_loss.item()

        best_model_info["train_loss_history"].append(train_loss.item())
        best_model_info["val_loss_history"].append(val_loss.item())

        # observe model info in console
        if show_model_info and epoch_i==0:
            best_model_info["model_structure"] = model.__str__() + "\n" + "="*100 + "\n" + summary(model, data.x, data.edge_index, data.batch, max_depth=20).__str__()
            logging.info(f"\n{best_model_info['model_structure']}")
        if(epoch_i % 10 == 0):  # show metrics every 10 epochs
            logging.info(f"Epoch {epoch_i:>3} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} ")

    return best_model, best_model_info


def test(model:torch.nn.Module, loader:torch_geometric.loader.dataloader.DataLoader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_i, data in enumerate(loader):
            data.to("cuda")
            x, x_edge_index, x_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            y, y_edge_index, y_batch_node_id, y_edge_attr = data.y[-1].x, data.y[-1].edge_index, torch.zeros(data.y[-1].x.shape[0], dtype=torch.int64).to("cuda"), data.y[-1].edge_attr  # only take y of x with last time-step on training
            graph_embeds_pred = model(x, x_edge_index, x_batch_node_id)
            y_graph_embeds = model.graph_encoder.get_embeddings(y, y_edge_index, y_batch_node_id)
            loss = criterion(graph_embeds_pred, y_graph_embeds)
            # test_loss += loss / len(loader)
            test_loss += loss

    return test_loss

def save_model(model:torch.nn.Module, model_info:dict, data_gen_cfg:dict):
    e_i = model_info["best_val_epoch"]
    t = datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    torch.save(model, model_dir/f"epoch_{e_i}-{t}.pt")
    with open(model_log_dir/f"epoch_{e_i}-{t}.json","w") as f:
        json_str = json.dumps(model_info)
        f.write(json_str)
    logging.info(f"model has been saved in:{model_dir}")

gin_encoder = GinEncoder(**gin_enc_cfg).to("cuda")
mts_corr_ad_cfg["graph_encoder"] = gin_encoder
model =  MTSCorrAD(**mts_corr_ad_cfg).to("cuda")
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
model, model_info = train(model, train_loader, val_loader, optimizer, criterion, epochs=5000, show_model_info=True)
save_model(model, model_info, data_gen_cfg)


# In[ ]:




