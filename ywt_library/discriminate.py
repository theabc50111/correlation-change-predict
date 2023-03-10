from itertools import product

import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import unbatch, unbatch_edge_index



# set devide of pytorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class DiscriminationTester:
    """
    Use the instance of this class to test the discrimination power of graph encoder and graphs
    """
    def __init__(self, criterion: torch.nn.modules.loss, data_loader: torch_geometric.loader.dataloader.DataLoader):
        self.data_loader = data_loader
        self.criterion = criterion
        self.__comp_graph_info, self.__min_diff_graph_info, self.__med_diff_graph_info, self.__max_diff_graph_info = self.set_diff_graphs()


    def set_diff_graphs(self):
        """
        Output three graphs based on the data_loader. These graphs should show the greatest difference, median difference, and least difference compared to the first graph.
        """
        x_list,  x_edge_attr_list  = [],  []
        batch_size, criterion = self.data_loader.batch_size, self.criterion
        num_x_edges = int(next(iter(self.data_loader)).num_edges / batch_size)
        for batched_data in self.data_loader:
            x_list.extend(unbatch(batched_data.x, batched_data.batch))
            x_edge_attr_list.extend([batched_data.edge_attr[i * num_x_edges: (i + 1) * num_x_edges] for i in range(batch_size)])

        graphs_disparity = np.array(list(map(lambda x: (criterion(x_list[0], x[0]) + criterion(x_edge_attr_list[0], x[1])).cpu().numpy(), zip(x_list[1:], x_edge_attr_list[1:]))))
        graph_disp_min_med_max_idx = np.argsort(graphs_disparity)[[0, int(len(graphs_disparity)/2), -1]] + 1  # +1 to make offset, because ignoring the first graph when computing graphs_disparity
        output_graph_idx = [0] + graph_disp_min_med_max_idx.tolist()

        return [{"x": x_list[i], "x_edge_attr": x_edge_attr_list[i], "graph_disp": criterion(x_edge_attr_list[0], x_edge_attr_list[i]).item()} for i in output_graph_idx]


    def real_disc_test(self, test_model: torch.nn.Module):
        """
        Use real data to test discirmination power
        """
        first_batch_data = next(iter(self.data_loader))
        num_nodes = unbatch(first_batch_data.x, first_batch_data.batch)[0].shape[0]
        first_edge_index = unbatch_edge_index(first_batch_data.edge_index, first_batch_data.batch)[0]
        first_batch_node_id = first_batch_data.batch[:num_nodes]
        graphs_info = self.__comp_graph_info, self.__min_diff_graph_info, self.__med_diff_graph_info, self.__max_diff_graph_info
        graph_embeds_list = [test_model.graph_encoder.get_embeddings(g_info["x"], first_edge_index, first_batch_node_id, g_info["x_edge_attr"]) for g_info in graphs_info]

        return tuple([self.criterion(gra_emb[0], gra_emb[1]).item() for gra_emb in product(graph_embeds_list[0], graph_embeds_list[1:])])
