import inspect
import logging
from math import sqrt

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
    def __init__(self, data_loader: torch_geometric.loader.dataloader.DataLoader,
                 x_edge_attr_mats: np.ndarray, num_diff_graphs: int = 5):
        self.data_loader = data_loader
        self.x_edge_attr_mats = x_edge_attr_mats
        self.criterion = torch.nn.MSELoss()
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

        ret_list = []
        for i in output_graph_idx:
            # Since there are not null values in x_edge_attr_mats, use self.x_edge_attr_mats[i] to find the index of non-null values. This can be done by x_edge_attr_mats[i][~np.isnan(self.x_edge_attr_mats[i])].
            non_null_idx = torch.tensor(~np.isnan(self.x_edge_attr_mats[i]), dtype=torch.bool)
            graph_dixp_info = {"gra_time_pt": i,
                               "graph_disp": criterion(x_list[0], x_list[i]).item() + criterion(x_edge_attr_mats[0], x_edge_attr_mats[i]).item(),
                               "x": x_list[i],
                               "x_edge_attr": x_edge_attr_mats[i][non_null_idx].reshape(-1, 1),
                               "x_edge_ind": x_edge_ind_list[i]}

            ret_list.append(graph_dixp_info)
        return ret_list


    def yield_real_disc(self, test_model: torch.nn.Module):
        """
        Use real data to test discirmination power of graph embeds
        """
        frame = inspect.currentframe().f_back
        instance_name = [var_name for var_name, var_val in frame.f_locals.items() if var_val is self][0]
        logger.debug(f"For {instance_name}〔called in {inspect.stack()[1][3]}() at [{frame.f_code.co_filename}:{frame.f_lineno}]〕")
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
                logger.debug(f"[{instance_name}]-- Time:{g_info['gra_time_pt']}, graph_dispiraty: {g_info['graph_disp']}, graph_embeds_dispiraty: {real_gra_embeds_dispiraty}, pred_embeds_disparity: {real_pred_embeds_dispiraty}")
                yield {"gra_enc_emb": real_gra_embeds_dispiraty, "pred_emb": real_pred_embeds_dispiraty}


class EdgeAccuracyLoss(torch.nn.Module):
    """ 
    This loss function is used to compute the edge accuracy of the prediction.
    """
    def __init__(self):
        super(EdgeAccuracyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, atol: float = 0.05) -> torch.Tensor:
        raise NotImplementedError("This function is not implemented yet.")
        ###edge_acc = torch.isclose(input, target, atol=atol, rtol=0).to(torch.float64).mean()
        ###edge_acc.requires_grad = True
        ###loss = 1 - edge_acc
        ###return loss


class BinsEdgeAccuracyLoss(torch.nn.Module):
    """
    This loss function is used to compute the edge accuracy of the discretized prediction.
    """
    def __init__(self):
        super(BinsEdgeAccuracyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, bins_list: list) -> torch.Tensor:
        raise NotImplementedError("This function is not implemented yet.")
        ###bins = torch.tensor(bins_list).reshape(-1, 1)
        ###num_bins = len(bins)-1
        ###bins = torch.concat((bins[:-1], bins[1:]), dim=1)
        ###edge_correct = 0
        ###discretize_values = np.linspace(-1, 1, num_bins)
        ###for lower, upper, discretize_value in zip(bins[:, 0], bins[:, 1], discretize_values):
        ###    bin_center = (lower + upper) / 2
        ###    atol = (upper - lower) / 2
        ###    mask_input = torch.where((input > lower) & (input <= upper), input, -12345)
        ###    convert_target = torch.where(target == discretize_value, bin_center, 12345)
        ###    edge_correct += torch.isclose(mask_input, convert_target, atol=atol, rtol=0).sum()
        ###lowest_bin_mask_input = torch.where(input == bins.min(), input, -12345)  # For the case of lowest bin
        ###edge_correct += torch.isclose(lowest_bin_mask_input, target, atol=0, rtol=0).sum()  # For the case of lowest bin
        ###edge_acc = edge_correct / torch.numel(input)
        ###edge_acc.requires_grad = True
        ###loss = 1 - edge_acc
        ###return loss


class TwoOrderPredProbEdgeAccuracyLoss(torch.nn.Module):
    def __init__(self, threshold: float, num_classes: int):
        super(TwoOrderPredProbEdgeAccuracyLoss, self).__init__()
        self.threshold = threshold
        self.num_classes = num_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, graph_size = input.shape
        assert num_classes == self.num_classes, "The number of classes in the input tensor is not equal to the number of classes in the model."
        sorted_pred_prob, sorted_indices = torch.sort(input, dim=1, descending=True)
        sorted_preds = sorted_indices
        first_order_preds = sorted_preds[::, 0]
        second_order_preds = sorted_preds[::, 1]
        first_second_order_pred_prob_diff = torch.diff(sorted_pred_prob, axis=1)[::, 0].abs()
        second_order_preds_mask = first_second_order_pred_prob_diff < self.threshold
        filtered_second_oreder_preds = torch.where(second_order_preds_mask, second_order_preds, torch.nan)
        assert not bool(((first_order_preds == target) & (filtered_second_oreder_preds == target)).sum()), "There are some edges that are both first and second order predictions at the same time, which is not allowed."
        num_correct_first_order_preds = (first_order_preds == target).sum()
        num_correct_second_order_preds = (filtered_second_oreder_preds == target).sum()
        edge_acc = (num_correct_first_order_preds+num_correct_second_order_preds)/target.numel()
        loss = 1 - edge_acc

        return loss


class TwoOrderPredProbEdgeAccuracy(torch.nn.Module):
    def __init__(self, threshold: float, num_classes: int):
        super(TwoOrderPredProbEdgeAccuracy, self).__init__()
        self.threshold = threshold
        self.num_classes = num_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, graph_size = input.shape
        assert num_classes == self.num_classes, "The number of classes in the input tensor is not equal to the number of classes in the model."
        two_order_pred_prob_edge_acc_loss_fn = TwoOrderPredProbEdgeAccuracyLoss(threshold=self.threshold, num_classes=input.shape[1])
        two_order_pred_prob_edge_acc_loss = two_order_pred_prob_edge_acc_loss_fn(input, target)
        edge_acc = 1 - two_order_pred_prob_edge_acc_loss

        return edge_acc

class UpperTriangleEdgeAccuracyLoss(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(UpperTriangleEdgeAccuracyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, graph_size = input.shape
        assert num_classes == self.num_classes, "The number of classes in the input tensor is not equal to the number of classes in the model."
        assert sqrt(graph_size).is_integer(), "The graph size must be a perfect square number."
        num_nodes = int(sqrt(graph_size))
        reshaped_input = input.reshape(-1, num_nodes, num_nodes)
        reshaped_target = target.reshape(-1, num_nodes, num_nodes)
        triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
        for i, pred_prob_along_class in enumerate(reshaped_input):
            pred_prob_triu = pred_prob_along_class[triu_idx[0], triu_idx[1]] if i == 0 else torch.vstack((pred_prob_triu, pred_prob_along_class[triu_idx[0], triu_idx[1]]))
        for i, target_along_graph in enumerate(reshaped_target):
            triu_target = target_along_graph[triu_idx[0], triu_idx[1]] if i == 0 else torch.vstack((triu_target, target_along_graph[triu_idx[0], triu_idx[1]]))
        triu_input = pred_prob_triu.reshape(batch_size, num_classes, -1)
        triu_preds = torch.argmax(triu_input, dim=1)
        edge_acc = (triu_preds == triu_target).to(torch.float64).mean()
        loss = 1 - edge_acc

        return loss

class UpperTriangleEdgeAccuracy(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(UpperTriangleEdgeAccuracy, self).__init__()
        self.num_classes = num_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, graph_size = input.shape
        assert num_classes == self.num_classes, "The number of classes in the input tensor is not equal to the number of classes in the model."
        assert sqrt(graph_size).is_integer(), "The graph size must be a perfect square number."
        upper_triangle_edge_acc_loss_fn = UpperTriangleEdgeAccuracyLoss(num_classes=input.shape[1])
        upper_triangle_edge_acc_loss = upper_triangle_edge_acc_loss_fn(input, target)
        edge_acc = 1 - upper_triangle_edge_acc_loss

        return edge_acc

class CustomIndicesEdgeAccuracyLoss(torch.nn.Module):
    def __init__(self, num_classes: int, selected_indices: list):
        super(CustomIndicesEdgeAccuracyLoss, self).__init__()
        self.num_classes = num_classes
        self.selected_indices = selected_indices

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, graph_size = input.shape
        assert num_classes == self.num_classes, "The number of classes in the input tensor is not equal to the number of classes in the model."
        assert graph_size > 1, "The graph size must be greater than 1."
        selected_input = input[::, ::, self.selected_indices]
        selected_target = target[::, self.selected_indices]
        selected_preds = torch.argmax(selected_input, dim=1)
        edge_acc = (selected_preds == selected_target).to(torch.float64).mean()
        loss = 1 - edge_acc
        print(f"input.shape: {input.shape}, target.shape: {target.shape}")
        print(f"selected_input.shape: {selected_input.shape}, selected_target.shape: {selected_target.shape}, selected_preds.shape: {selected_preds.shape}")


        return loss


class CustomIndicesEdgeAccuracy(torch.nn.Module):
    def __init__(self, num_classes: int, selected_indices: list):
        super(CustomIndicesEdgeAccuracy, self).__init__()
        self.num_classes = num_classes
        self.selected_indices = selected_indices

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        custom_indices_edge_acc_loss_fn = CustomIndicesEdgeAccuracyLoss(num_classes=input.shape[1], selected_indices=self.selected_indices)
        custom_indices_edge_acc_loss = custom_indices_edge_acc_loss_fn(input, target)
        edge_acc = 1 - custom_indices_edge_acc_loss

        return edge_acc
