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
                x_edge_attr_mats: np.ndarray):
        self.data_loader = data_loader
        self.x_edge_attr_mats = x_edge_attr_mats
        self.criterion = criterion
        self.__comp_graph_info, self.__min_diff_graph_info, self.__med_diff_graph_info, self.__max_diff_graph_info = self.set_diff_graphs()


    def set_diff_graphs(self):
        """
        Output three graphs based on the data_loader. These graphs should show the greatest difference, median difference, and least difference compared to the first graph.
        """
        x_list,  x_edge_ind_list, x_batch_node_id_list = [], [], []
        criterion = self.criterion
        for batched_data in self.data_loader:
            x_list.extend(unbatch(batched_data.x, batched_data.batch))
            x_edge_ind_list.extend(unbatch_edge_index(batched_data.edge_index, batched_data.batch))

        x_edge_attr_mats = torch.tensor(np.nan_to_num(self.x_edge_attr_mats, nan=0), dtype=torch.float32)
        criterion = self.criterion
        graphs_disparity = np.array(list(map(lambda x: (criterion(x_list[0], x[0]) + criterion(x_edge_attr_mats[0], x[1])).cpu().numpy(), zip(x_list[1:], x_edge_attr_mats[1:]))))
        graph_disp_min_med_max_idx = np.argsort(graphs_disparity)[[0, int(len(graphs_disparity)/2), -1]] + 1  # +1 to make offset, because ignoring the first graph when computing graphs_disparity
        output_graph_idx = [0] + graph_disp_min_med_max_idx.tolist()

        return [{"gra_time_pt":i, "graph_disp": criterion(x_list[0], x_list[i]).item() + criterion(x_edge_attr_mats[0], x_edge_attr_mats[i]).item(), "x": x_list[i], "x_edge_attr": x_edge_attr_mats[i][~np.isnan(self.x_edge_attr_mats[i])].reshape(-1, 1), "x_edge_ind": x_edge_ind_list[i]} for i in output_graph_idx]


    def yield_real_disc(self, test_model: torch.nn.Module):
        """
        Use real data to test discirmination power of graph embeds
        """
        graphs_info = self.__comp_graph_info, self.__min_diff_graph_info, self.__med_diff_graph_info, self.__max_diff_graph_info
        logger.info(list(map(lambda x: {"gra_time_pt": x["gra_time_pt"], "graph_disp": x["graph_disp"]}, graphs_info)))

        for i, g_info in enumerate(graphs_info):
            if i == 0:
                comp_gra_embeds = test_model.graph_encoder.get_embeddings(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])
            else:
                gra_embeds = test_model.graph_encoder.get_embeddings(g_info["x"], g_info["x_edge_ind"], torch.zeros(g_info["x"].shape[0], dtype=torch.int64), g_info["x_edge_attr"])

                real_gra_dispiraty = self.criterion(comp_gra_embeds, gra_embeds).item()
                logger.debug(f"Time:{g_info['gra_time_pt']}, graph_embeds_dispiraty: {real_gra_dispiraty}, graph_dispiraty: {g_info['graph_disp']}")
                yield real_gra_dispiraty
