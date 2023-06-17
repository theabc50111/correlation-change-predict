#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import numpy as np
import torch
import yaml
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR, SequentialLR
from torch_geometric.data import DataLoader
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
logger.setLevel(logging.INFO)
metrics_logger.setLevel(logging.INFO)
utils_logger.setLevel(logging.INFO)
warnings.simplefilter("ignore")


class GAE(torch.nn.Module):
    def __init__(self, model_cfg):
        super(GAE, self).__init__()

        self.model_cfg = model_cfg
        # create data loader
        self.num_tr_batches = self.model_cfg["num_batches"]['train']
        self.num_val_batches = self.model_cfg["num_batches"]['val']

        # set model components
        self.graph_encoder = self.model_cfg['graph_encoder'](**self.model_cfg)
        graph_enc_emb_size = self.graph_encoder.gra_enc_l * self.graph_encoder.gra_enc_h  # the input size of GRU depend on the number of layers of GINconv
        self.decoder = self.model_cfg['decoder'](graph_enc_emb_size, self.model_cfg["num_edges"], drop_p=self.model_cfg["drop_p"] if "decoder" in self.model_cfg["drop_pos"] else 0)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.model_cfg['learning_rate'], weight_decay=self.model_cfg['weight_decay'])
        schedulers = [ConstantLR(self.optimizer, factor=0.1, total_iters=self.num_tr_batches*6), MultiStepLR(self.optimizer, milestones=list(range(self.num_tr_batches*5, self.num_tr_batches*600, self.num_tr_batches*50))+list(range(self.num_tr_batches*600, self.num_tr_batches*self.model_cfg['tr_epochs'], self.num_tr_batches*100)), gamma=0.9)]
        self.scheduler = SequentialLR(self.optimizer, schedulers=schedulers, milestones=[self.num_tr_batches*6])
        observe_model_cfg = {item[0]: item[1] for item in self.model_cfg.items() if item[0] != 'dataset'}
        observe_model_cfg['optimizer'] = str(self.optimizer)
        observe_model_cfg['scheduler'] = {"scheduler_name": str(self.scheduler.__class__.__name__), "milestones": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones), "gamma": self.scheduler._schedulers[1].gamma}
        self.graph_enc_num_layers = sum(1 for _ in self.graph_encoder.parameters())

        logger.info(f"\nModel Configuration: \n{observe_model_cfg}")

    def forward(self, x, edge_index, seq_batch_node_id, edge_attr):
        """
        Operate when model called
        """
        # Inter-series modeling
        if type(self.graph_encoder).__name__ == "GinEncoder":
            graph_embeds = self.graph_encoder(x, edge_index, seq_batch_node_id)
        elif type(self.graph_encoder).__name__ == "GineEncoder":
            graph_embeds = self.graph_encoder(x, edge_index, seq_batch_node_id, edge_attr)

        x_recon = self.decoder(graph_embeds)
        return x_recon

    def train(self, mode: bool = True, train_data: np.ndarray = None, val_data: np.ndarray = None, loss_fns: dict = None, epochs: int = 5, show_model_info: bool = False):
        """
        Training MTSCorrAD Model
        """
        # In order to make original function of nn.Module.train() work, we need to override it
        super().train(mode=mode)
        if train_data is None:
            return self

        best_model = []

        train_loader = self.create_pyg_data_loaders(graph_adj_mats=train_data['edges'],  graph_nodes_mats=train_data["nodes"], loader_seq_len=self.model_cfg["seq_len"])
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "gra_enc_weight_l2_reg": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1),
                             "gra_enc_grad": torch.zeros(1), "gru_grad": torch.zeros(1), "gra_dec_grad": torch.zeros(1), "lr": torch.zeros(1),
                             "pred_gra_embeds": [], "y_gra_embeds": [], "gra_embeds_disparity": {}}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            for batch_idx, batch_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_loss = torch.zeros(1)
                batch_edge_acc = torch.zeros(1)

    def create_pyg_data_loaders(self, graph_adj_mats: np.ndarray, graph_nodes_mats: np.ndarray, loader_seq_len : int, show_log: bool = True, show_debug_info: bool = False):
        """
        Create Pytorch Geometric DataLoaders
        """
        # Create an instance of the GraphTimeSeriesDataset
        dataset = GraphTimeSeriesDataset(graph_adj_mats,  graph_nodes_mats, model_cfg=self.model_cfg, show_log=show_log)
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
    gae_args_parser = argparse.ArgumentParser()
    gae_args_parser.add_argument("--data_implement", type=str, nargs='?', default="SP500_20082017_CORR_SER_REG_STD_CORR_MAT_HRCHY_10_CLUSTER_LABEL_HALF_MIX",
                                 help="input the data implement name, watch options by operate: logger.info(data_cfg['DATASETS'].keys())")
    gae_args_parser.add_argument("--batch_size", type=int, nargs='?', default=10,
                                 help="input the number of batch size")
    gae_args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=300,
                                 help="input the number of training epochs")
    gae_args_parser.add_argument("--seq_len", type=int, nargs='?', default=30,
                                 help="input the number of sequence length")
    gae_args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                 help="input --save_model to save model weight and model info")
    gae_args_parser.add_argument("--corr_stride", type=int, nargs='?', default=1,
                                 help="input the number of stride length of correlation computing")
    gae_args_parser.add_argument("--corr_window", type=int, nargs='?', default=10,
                                 help="input the number of window length of correlation computing")
    gae_args_parser.add_argument("--filt_mode", type=str, nargs='?', default=None,
                                 help="input the filtered mode of graph edges, look up the options by execute python ywt_library/data_module.py -h")
    gae_args_parser.add_argument("--filt_quan", type=float, nargs='?', default=0.5,
                                 help="input the filtered quantile of graph edges")
    gae_args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default=None,
                                 help="Decide mode of nodes' vaules of graph_nodes_matrix, look up the options by execute python ywt_library/data_module.py -h")
    gae_args_parser.add_argument("--learning_rate", type=float, nargs='?', default=0.0001,
                                 help="input the learning rate of training")
    gae_args_parser.add_argument("--weight_decay", type=float, nargs='?', default=0.01,
                                 help="input the weight decay of training")
    gae_args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                                 help="input [gru] | [gru decoder] | [decoder gru graph_encoder] to decide the position of drop layers")
    gae_args_parser.add_argument("--drop_p", type=float, default=0,
                                 help="input 0~1 to decide the probality of drop layers")
    gae_args_parser.add_argument("--gra_enc", type=str, nargs='?', default="gine",
                                 help="input the type of graph encoder")
    gae_args_parser.add_argument("--gra_enc_aggr", type=str, nargs='?', default="add",
                                 help="input the type of aggregator of graph encoder")
    gae_args_parser.add_argument("--gra_enc_l", type=int, nargs='?', default=1,  # range:1~n, for graph encoder after the second layer,
                                 help="input the number of graph laryers of graph_encoder")
    gae_args_parser.add_argument("--gra_enc_h", type=int, nargs='?', default=4,
                                 help="input the number of graph embedding hidden size of graph_encoder")
    ARGS = gae_args_parser.parse_args()
    logger.info(pformat(f"\n{vars(ARGS)}", indent=1, width=40, compact=True))

    # Data implement & output setting & testset setting
    # data implement setting
    data_implement = ARGS.data_implement
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
    graph_adj_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/filtered_graph_adj_mat/{ARGS.filt_mode}-quan{str(ARGS.filt_quan).replace('.', '')}" if ARGS.filt_mode else Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_adj_mat"
    graph_node_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"]) / f"{output_file_name}/graph_node_mat"
    g_model_dir = current_dir / f'save_models/mts_corr_ad_model/{output_file_name}/corr_s{s_l}_w{w_l}'
    g_model_log_dir = current_dir / f'save_models/mts_corr_ad_model/{output_file_name}/corr_s{s_l}_w{w_l}/train_logs/'
    g_model_dir.mkdir(parents=True, exist_ok=True)
    g_model_log_dir.mkdir(parents=True, exist_ok=True)

    # model configuration
    is_training, train_count = True, 0
    gra_edges_data_mats = np.load(graph_adj_mat_dir / f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    gra_nodes_data_mats = np.load(graph_node_mat_dir / f"{ARGS.graph_nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy") if ARGS.graph_nodes_v_mode else np.ones((gra_edges_data_mats.shape[0], 1, gra_edges_data_mats.shape[2]))
    norm_train_dataset, norm_val_dataset, norm_test_dataset, scaler = split_and_norm_data(gra_edges_data_mats, gra_nodes_data_mats)
    # show info
    logger.info(f"gra_edges_data_mats.shape:{gra_edges_data_mats.shape}, gra_nodes_data_mats.shape:{gra_nodes_data_mats.shape}")
    logger.info(f"gra_edges_data_mats.max:{np.nanmax(gra_edges_data_mats)}, gra_edges_data_mats.min:{np.nanmin(gra_edges_data_mats)}")
    logger.info(f"gra_nodes_data_mats.max:{np.nanmax(gra_nodes_data_mats)}, gra_nodes_data_mats.min:{np.nanmin(gra_nodes_data_mats)}")
    logger.info(f"norm_train_nodes_data_mats.max:{np.nanmax(norm_train_dataset['nodes'])}, norm_train_nodes_data_mats.min:{np.nanmin(norm_train_dataset['nodes'])}")
    logger.info(f"norm_val_nodes_data_mats.max:{np.nanmax(norm_val_dataset['nodes'])}, norm_val_nodes_data_mats.min:{np.nanmin(norm_val_dataset['nodes'])}")
    logger.info(f"norm_test_nodes_data_mats.max:{np.nanmax(norm_test_dataset['nodes'])}, norm_test_nodes_data_mats.min:{np.nanmin(norm_test_dataset['nodes'])}")
    logger.info(f'Training set   = {len(norm_train_dataset["edges"])} graphs')
    logger.info(f'Validation set = {len(norm_val_dataset["edges"])} graphs')
    logger.info(f'Test set       = {len(norm_test_dataset["edges"])} graphs')
    logger.info("="*80)

    gae_cfg = {"filt_mode": ARGS.filt_mode,
               "filt_quan": ARGS.filt_quan,
               "graph_nodes_v_mode": ARGS.graph_nodes_v_mode,
               "tr_epochs": ARGS.tr_epochs,
               "batch_size": ARGS.batch_size,
               "seq_len": ARGS.seq_len,
               "num_batches": {"train": ((len(norm_train_dataset["edges"])-1)//ARGS.batch_size),
                               "val": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size),
                               "test": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size)},
               "learning_rate": ARGS.learning_rate,
               "weight_decay": ARGS.weight_decay,
               "drop_pos": ARGS.drop_pos,
               "drop_p": ARGS.drop_p,
               "gra_enc_aggr": ARGS.gra_enc_aggr,
               "gra_enc_l": ARGS.gra_enc_l,
               "gra_enc_h": ARGS.gra_enc_h,
               "num_edges": (norm_train_dataset["edges"].shape[1]),
               "num_node_features": norm_train_dataset["nodes"].shape[1],
               "num_edge_features": 1,
               "graph_encoder": GineEncoder if ARGS.gra_enc == "gine" else GinEncoder,
               "decoder": MLPDecoder}

    model = GAE(gae_cfg)
    loss_fns_dict = {"fns": [MSELoss(), EdgeAccuracyLoss()],
                     "fn_args": {"MSELoss()": {}, "EdgeAccuracyLoss()": {}}}
    while (is_training is True) and (train_count < 100):
        try:
            train_count += 1
            train_loader = model.create_pyg_data_loaders(graph_adj_mats=norm_train_dataset['edges'],  graph_nodes_mats=norm_train_dataset["nodes"], loader_seq_len=gae_cfg["seq_len"], show_log=True, show_debug_info=False)
            #g_best_model, g_best_model_info = model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
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
