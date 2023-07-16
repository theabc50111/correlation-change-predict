#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import functools
import json
import logging
import sys
import traceback
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from pprint import pformat

import dynamic_yaml
import numpy as np
import torch
import yaml
from torch.nn import MSELoss
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

        best_model_info = {"num_training_graphs": len(train_data),
                           "filt_mode": self.model_cfg['filt_mode'],
                           "filt_quan": self.model_cfg['filt_quan'],
                           "graph_nodes_v_mode": self.model_cfg['graph_nodes_v_mode'],
                           "batches_per_epoch": self.num_tr_batches,
                           "epochs": epochs,
                           "batch_size": self.model_cfg['batch_size'],
                           "optimizer": str(self.optimizer),
                           "opt_scheduler": {"gamma": self.scheduler._schedulers[1].gamma, "milestoines": self.scheduler._milestones+list(self.scheduler._schedulers[1].milestones)},
                           "loss_fns": str([fn.__name__ if hasattr(fn, '__name__') else str(fn) for fn in loss_fns["fns"]]),
                           "gra_enc_weight_l2_reg_lambda": self.model_cfg['graph_enc_weight_l2_reg_lambda'],
                           "drop_pos": self.model_cfg["drop_pos"],
                           "drop_p": self.model_cfg["drop_p"],
                           "graph_enc": type(self.graph_encoder).__name__,
                           "gra_enc_aggr": self.model_cfg['gra_enc_aggr'],
                           "min_val_loss": float('inf')}
        best_model = []
        train_loader = self.create_pyg_data_loaders(graph_adj_mats=train_data['edges'],  graph_nodes_mats=train_data["nodes"], batch_size=self.model_cfg["batch_size"])
        for epoch_i in tqdm(range(epochs)):
            self.train()
            epoch_metrics = {"tr_loss": torch.zeros(1), "val_loss": torch.zeros(1), "gra_enc_weight_l2_reg": torch.zeros(1), "tr_edge_acc": torch.zeros(1), "val_edge_acc": torch.zeros(1),
                             "gra_enc_grad": torch.zeros(1), "gra_dec_grad": torch.zeros(1), "lr": torch.zeros(1)}
            epoch_metrics.update({str(fn): torch.zeros(1) for fn in loss_fns["fns"]})
            # Train on batches
            for batch_idx, batch_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_loss = torch.zeros(1)
                batch_edge_acc = torch.zeros(1)
                for data_batch_idx in range(len(batch_data)):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    num_nodes = x.shape[0]
                    recon_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                    x_graph_adj = torch.sparse_coo_tensor(x_edge_index, x_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    edge_acc = np.isclose(recon_graph_adj.cpu().detach().numpy(), x_graph_adj.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                    batch_edge_acc += edge_acc/(self.num_tr_batches*self.model_cfg['batch_size'])
                    #print(f"batch_idx:{batch_idx}, data_batch_idx:{data_batch_idx}, edge_acc:{edge_acc}, batch_edge_acc:{batch_edge_acc}")
                    loss_fns["fn_args"]["MSELoss()"].update({"input": recon_graph_adj, "target":  x_graph_adj})
                    loss_fns["fn_args"]["EdgeAccuracyLoss()"].update({"input": recon_graph_adj, "target":  x_graph_adj})
                    #print(f"self.num_tr_batches:{self.num_tr_batches}, self.model_cfg['batch_size']:{self.model_cfg['batch_size']}, self.num_tr_batches*self.model_cfg['batch_size']: {self.num_tr_batches*self.model_cfg['batch_size']}")
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        batch_loss += loss/(self.num_tr_batches*self.model_cfg['batch_size'])
                        #print(f"fn_name:{fn_name}, loss:{loss}, batch_loss:{batch_loss}")
                        epoch_metrics[fn_name] += loss/(self.num_tr_batches*self.model_cfg['batch_size'])

                if self.model_cfg['graph_enc_weight_l2_reg_lambda']:
                    gra_enc_weight_l2_penalty = self.model_cfg['graph_enc_weight_l2_reg_lambda']*sum(p.pow(2).mean() for p in self.graph_encoder.parameters())
                    batch_loss += gra_enc_weight_l2_penalty
                else:
                    gra_enc_weight_l2_penalty = 0
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # record metrics for each batch
                epoch_metrics["tr_loss"] += batch_loss
                epoch_metrics["tr_edge_acc"] += batch_edge_acc
                epoch_metrics["gra_enc_weight_l2_reg"] += gra_enc_weight_l2_penalty
                epoch_metrics["gra_enc_grad"] += sum(p.grad.sum() for p in self.graph_encoder.parameters() if p.grad is not None)/self.num_tr_batches
                epoch_metrics["gra_dec_grad"] += sum(p.grad.sum() for p in self.decoder.parameters() if p.grad is not None)/self.num_tr_batches
                epoch_metrics["lr"] = torch.tensor(self.optimizer.param_groups[0]['lr'])
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
                best_model = {"encoder_weight": copy.deepcopy(self.graph_encoder.state_dict()), "decoder_weight": copy.deepcopy(self.decoder.state_dict())}
                best_model_info["best_val_epoch"] = epoch_i
                best_model_info["min_val_loss"] = epoch_metrics['val_loss'].item()
                best_model_info["min_val_loss_edge_acc"] = epoch_metrics['val_edge_acc'].item()

            # observe model info in console
            if epoch_i == 0:
                best_model_info["model_structure"] = str(self) + "\n" + "="*100 + "\n" + str(summary(self, log_model_info_data.x, log_model_info_data.edge_index, log_model_info_data.batch, log_model_info_data.edge_attr, max_depth=20))
                if show_model_info:
                    logger.info(f"\nNumber of graphs:{batch_data.num_graphs} in No.{log_model_info_batch_idx} batch, the model structure:\n{best_model_info['model_structure']}")
            if epoch_i % 10 == 0:  # show metrics every 10 epochs
                epoch_metric_log_msgs = " | ".join([f"{k}: {v.item():.9f}" for k, v in epoch_metrics.items() if "embeds" not in k])
                logger.info(f"In Epoch {epoch_i:>3} | {epoch_metric_log_msgs} | lr: {self.optimizer.param_groups[0]['lr']:.9f}")
            if epoch_i % 500 == 0:  # show oredictive and real adjacency matrix every 500 epochs
                logger.info(f"\nIn Epoch {epoch_i:>3} \nrecon_graph_adj:\n{recon_graph_adj}\nx_graph_adj:\n{x_graph_adj}\n")

        return best_model, best_model_info

    def test(self, test_data: np.ndarray = None, loss_fns: dict = None, show_loader_log: bool = False):
        self.eval()
        test_loss = 0
        test_edge_acc = 0
        test_loader = self.create_pyg_data_loaders(graph_adj_mats=test_data["edges"],  graph_nodes_mats=test_data["nodes"], batch_size=self.model_cfg["batch_size"], show_log=show_loader_log)
        with torch.no_grad():
            for batch_data in test_loader:
                for data_batch_idx in range(len(batch_data)):
                    data = batch_data[data_batch_idx]
                    x, x_edge_index, x_seq_batch_node_id, x_edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                    num_nodes = x.shape[0]
                    recon_graph_adj = self(x, x_edge_index, x_seq_batch_node_id, x_edge_attr)
                    x_graph_adj = torch.sparse_coo_tensor(x_edge_index, x_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    edge_acc = np.isclose(recon_graph_adj.cpu().detach().numpy(), x_graph_adj.cpu().detach().numpy(), atol=0.05, rtol=0).mean()
                    test_edge_acc += edge_acc/(self.num_val_batches*self.model_cfg['batch_size'])
                    loss_fns["fn_args"]["MSELoss()"].update({"input": recon_graph_adj, "target":  x_graph_adj})
                    loss_fns["fn_args"]["EdgeAccuracyLoss()"].update({"input": recon_graph_adj, "target":  x_graph_adj})
                    for fn in loss_fns["fns"]:
                        fn_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
                        partial_fn = functools.partial(fn, **loss_fns["fn_args"][fn_name])
                        loss = partial_fn()
                        test_loss += loss/(self.num_val_batches*self.model_cfg['batch_size'])

        return test_loss, test_edge_acc

    @staticmethod
    def save_model(unsaved_model: OrderedDict, model_info: dict, model_dir: Path, model_log_dir: Path):
        e_i = model_info.get("best_val_epoch")
        t_stamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        torch.save(unsaved_model['encoder_weight'], model_dir/f"epoch_{e_i}-{t_stamp}-encoder_weight.pt")
        torch.save(unsaved_model['decoder_weight'], model_dir/f"epoch_{e_i}-{t_stamp}-decoder_weight.pt")
        with open(model_log_dir/f"epoch_{e_i}-{t_stamp}.json", "w") as f:
            json_str = json.dumps(model_info)
            f.write(json_str)
        logger.info(f"model has been saved in:{model_dir}")

    def create_pyg_data_loaders(self, graph_adj_mats: np.ndarray, graph_nodes_mats: np.ndarray, batch_size: int, show_log: bool = True, show_debug_info: bool = False):
        """
        Create Pytorch Geometric DataLoaders
        """
        # Create an instance of the GraphTimeSeriesDataset
        graph_time_len = graph_adj_mats.shape[0]
        graph_nodes_mats = graph_nodes_mats.transpose(0, 2, 1)
        data_list = []
        for g_t in range(graph_time_len):
            edge_index = torch.tensor(np.stack(np.where(~np.isnan(graph_adj_mats[g_t])), axis=1))
            edge_attr = torch.tensor(graph_adj_mats[g_t][~np.isnan(graph_adj_mats[g_t])].reshape(-1, 1), dtype=torch.float64)
            node_attr = torch.tensor(graph_nodes_mats[g_t], dtype=torch.float64)
            data = Data(x=node_attr, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
            data_list.append(data)
        # Create mini-batches
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

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
                    logger.debug(f"subgraph.num_graphs:{subgraph.num_graphs}\n")
                    x_nodes = subgraph[data_batch_idx].x
                    x_edge_index = subgraph[data_batch_idx].edge_index
                    x_edge_attr = subgraph[data_batch_idx].edge_attr
                    num_nodes = x_nodes.shape[0]
                    x_graph_adj = torch.sparse_coo_tensor(x_edge_index, x_edge_attr[:, 0], (num_nodes, num_nodes)).to_dense()
                    logger.debug((f"\n---------------------------At batch{batch_idx} and data{data_batch_idx}---------------------------\n"
                                  f"x.shape: {x_nodes.shape}, x_edge_attr.shape: {x_edge_attr.shape}, x_edge_idx.shape: {x_edge_index.shape}, x_graph_adj.shape:{x_graph_adj.shape}\n"
                                  f"x:\n{x_nodes}\n"
                                  f"x_edges_idx[:5]:\n{x_edge_index[::, :5]}\n"
                                  f"x_graph_adj:\n{x_graph_adj}\n"
                                  f"\n---------------------------At batch{batch_idx} and data{data_batch_idx}---------------------------\n"))

        return data_loader


if __name__ == "__main__":
    gae_args_parser = argparse.ArgumentParser()
    gae_args_parser.add_argument("--data_implement", type=str, nargs='?', default="SP500_20082017_CORR_SER_REG_STD_CORR_MAT_HRCHY_10_CLUSTER_LABEL_HALF_MIX",
                                 help="input the data implement name, watch options by operate: logger.info(data_cfg['DATASETS'].keys())")
    gae_args_parser.add_argument("--batch_size", type=int, nargs='?', default=10,
                                 help="input the number of batch size")
    gae_args_parser.add_argument("--tr_epochs", type=int, nargs='?', default=300,
                                 help="input the number of training epochs")
    gae_args_parser.add_argument("--save_model", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                 help="input --save_model to save model weight and model info")
    gae_args_parser.add_argument("--corr_type", type=str, nargs='?', default="pearson",
                                 choices=["pearson", "cross_corr"],
                                 help="input the type of correlation computing, the choices are [pearson, cross_corr]")
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
    gae_args_parser.add_argument("--graph_enc_weight_l2_reg_lambda", type=float, nargs='?', default=0,
                                 help="input the weight of graph encoder weight l2 norm loss")
    gae_args_parser.add_argument("--drop_pos", type=str, nargs='*', default=[],
                                 help="input [decoder] | [graph_encoder] | [decoder graph_encoder] to decide the position of drop layers")
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
    graph_adj_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.corr_type}/filtered_graph_adj_mat/{ARGS.filt_mode}-quan{str(ARGS.filt_quan).replace('.', '')}" if ARGS.filt_mode else Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{ARGS.corr_type}/graph_adj_mat"
    graph_node_mat_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/graph_node_mat"
    g_model_dir = current_dir / f'save_models/gae_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}'
    g_model_log_dir = current_dir / f'save_models/gae_model/{output_file_name}/{ARGS.corr_type}/corr_s{s_l}_w{w_l}/train_logs/'
    g_model_dir.mkdir(parents=True, exist_ok=True)
    g_model_log_dir.mkdir(parents=True, exist_ok=True)

    # model configuration
    is_training, train_count = True, 0
    gra_edges_data_mats = np.load(graph_adj_mat_dir/f"corr_s{s_l}_w{w_l}_adj_mat.npy")
    gra_nodes_data_mats = np.load(graph_node_mat_dir/f"{ARGS.graph_nodes_v_mode}_s{s_l}_w{w_l}_nodes_mat.npy") if ARGS.graph_nodes_v_mode else np.ones((gra_edges_data_mats.shape[0], 1, gra_edges_data_mats.shape[2]))
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
               "num_batches": {"train": ((len(norm_train_dataset["edges"])-1)//ARGS.batch_size),
                               "val": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size),
                               "test": ((len(norm_val_dataset["edges"])-1)//ARGS.batch_size)},
               "learning_rate": ARGS.learning_rate,
               "weight_decay": ARGS.weight_decay,
               "graph_enc_weight_l2_reg_lambda": ARGS.graph_enc_weight_l2_reg_lambda,
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
            g_best_model, g_best_model_info = model.train(train_data=norm_train_dataset, val_data=norm_val_dataset, loss_fns=loss_fns_dict, epochs=ARGS.tr_epochs, show_model_info=True)
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
