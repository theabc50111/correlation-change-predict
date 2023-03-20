import logging

import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import unbatch, unbatch_edge_index

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)

# set devide of pytorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class DiscriminationTester:
    """
    Use the instance of this class to test the discrimination power of graph encoder and graphs
    """
    def __init__(self, criterion: torch.nn.modules.loss,
                 data_loader: torch_geometric.loader.dataloader.DataLoader,
                 x_edge_attr_mats: np.ndarray, num_diff_graphs: int = 5):
        self.data_loader = data_loader
        self.x_edge_attr_mats = x_edge_attr_mats
        self.criterion = criterion
        self.num_diff_graphs = num_diff_graphs
        self.graphs_info = self.set_diff_graphs(num_intv=num_diff_graphs)


    def set_diff_graphs(self, num_intv):
        """
        Output three graphs based on the data_loader. These graphs should show the greatest difference, median difference, and least difference compared to the first graph.
        """
        x_list,  x_edge_ind_list,  = [], []
        criterion = self.criterion
        for batched_data in self.data_loader:
            x_list.extend(unbatch(batched_data.x, batched_data.batch))
            x_edge_ind_list.extend(unbatch_edge_index(batched_data.edge_index, batched_data.batch))

        x_edge_attr_mats = torch.tensor(np.nan_to_num(self.x_edge_attr_mats, nan=0), dtype=torch.float32)  # To compute the L2-loss between x_edge_attr_mats, fill any null values with 0.
        criterion = self.criterion
        graphs_disparity = np.array(list(map(lambda x: (criterion(x_list[0], x[0]) + criterion(x_edge_attr_mats[0], x[1])).cpu().numpy(), zip(x_list[1:], x_edge_attr_mats[1:]))))
        intv_inds = np.linspace(0, len(graphs_disparity)-1, num=num_intv, endpoint=True).astype(int).tolist()
        graph_disp_min_med_max_idx = np.argsort(graphs_disparity)[intv_inds] + 1  # +1 to make offset, because ignoring the first graph when computing graphs_disparity
        output_graph_idx = [0] + graph_disp_min_med_max_idx.tolist()

        # Since there are not null values in x_edge_attr_mats, use self.x_edge_attr_mats[i] to find the index of non-null values. This can be done by x_edge_attr_mats[i][~np.isnan(self.x_edge_attr_mats[i])].
        return [{"gra_time_pt":i, "graph_disp": criterion(x_list[0], x_list[i]).item() + criterion(x_edge_attr_mats[0], x_edge_attr_mats[i]).item(), "x": x_list[i], "x_edge_attr": x_edge_attr_mats[i][~np.isnan(self.x_edge_attr_mats[i])].reshape(-1, 1), "x_edge_ind": x_edge_ind_list[i]} for i in output_graph_idx]


    def yield_real_disc(self, test_model: torch.nn.Module):
        """
        Use real data to test discirmination power of graph embeds
        """
        logger.debug(list(map(lambda x: {"gra_time_pt": x["gra_time_pt"], "graph_disp": x["graph_disp"]}, self.graphs_info)))

        for i, g_info in enumerate(self.graphs_info):
            if i == 0:
                comp_gra_embeds = test_model.graph_encoder.get_embeddings(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
                comp_pred_embeds = test_model(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
            else:
                gra_embeds = test_model.graph_encoder.get_embeddings(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
                pred_embeds = test_model(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
                real_gra_embeds_dispiraty = self.criterion(comp_gra_embeds, gra_embeds).item()
                real_pred_embeds_dispiraty = self.criterion(comp_pred_embeds, pred_embeds).item()
                logger.debug(f"Time:{g_info['gra_time_pt']}, graph_dispiraty: {g_info['graph_disp']}, graph_embeds_dispiraty: {real_gra_embeds_dispiraty}, pred_embeds_disparity: {real_pred_embeds_dispiraty}")
                yield {"gra_enc_emb": real_gra_embeds_dispiraty, "pred_emb": real_pred_embeds_dispiraty}
