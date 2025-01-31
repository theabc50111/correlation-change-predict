import json
import logging
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)

def stl_decompn(corr_series: "pd.Series", overview: bool = False) -> (float, float, float):
    output_resid = 100000
    output_trend = None
    output_period = None
    corr_series.name = corr_series.iloc[0]
    corr_series = corr_series.iloc[1:]
    for p in range(2, 11):
        decompose_result_mult = seasonal_decompose(corr_series, period=p)
        resid_sum = np.abs(decompose_result_mult.resid).sum()
        if output_resid > resid_sum:
            output_resid = resid_sum
            output_trend = decompose_result_mult.trend.dropna()
            output_period = p

    reg = LinearRegression().fit(np.arange(len(output_trend)).reshape(-1, 1), output_trend)

    if overview:
        decompose_result_mult = seasonal_decompose(corr_series, period=output_period)
        trend = decompose_result_mult.trend.dropna().reset_index(drop=True)
        plt.figure(figsize=(7, 1))
        plt.plot(trend)
        plt.plot([0, len(trend)], [reg.intercept_, reg.intercept_+ len(trend)*reg.coef_])
        plt.title("trend & regression line")
        plt.show()
        plt.close()
        decompose_result_mult_fig = decompose_result_mult.plot()
        decompose_result_mult_fig .set_size_inches(10, 12, forward=True)
        for ax in decompose_result_mult_fig.axes:
            ax.tick_params(labelrotation=60) # Rotates X-Axis Ticks by 45-degrees
        plt.show()
        plt.close()

    return output_period, output_resid, output_trend.std(), reg.coef_[0]


def calc_corr_ser_property(corr_dataset: pd.DataFrame, corr_property_df_path: Path):
    """
    Produce property of correlation series in form of dataframe
    """
    if corr_property_df_path.exists():
        corr_property_df = pd.read_csv(corr_property_df_path).set_index("items")
    else:
        corr_mean = corr_dataset.mean(axis=1)
        corr_std = corr_dataset.std(axis=1)
        corr_stl_series = corr_dataset.apply(stl_decompn, axis=1)
        corr_stl_array = [[stl_period, stl_resid, stl_trend_std, stl_trend_coef] for stl_period, stl_resid, stl_trend_std, stl_trend_coef in corr_stl_series.values]
        corr_property_df = pd.DataFrame(corr_stl_array, index=corr_dataset.index)
        corr_property_df = pd.concat([corr_property_df, corr_mean, corr_std], axis=1)
        corr_property_df.columns = ["corr_stl_period", "corr_stl_resid", "corr_stl_trend_std", "corr_stl_trend_coef", "corr_ser_mean", "corr_ser_std"]
        corr_property_df.index.name = "items"
        corr_property_df.to_csv(corr_property_df_path)
    return corr_property_df


def split_and_norm_data(edges_mats: np.ndarray, nodes_mats: np.ndarray, target_mats: np.ndarray, batch_size: int, show_info: bool = False):
    """
    split dataset to train, validation, test
    normalize these dataset
    """
    assert len(edges_mats) == len(nodes_mats), "Check the whether gra_edges_data_mats match gra_nodes_data_mats"
    num_graphs = len(nodes_mats)

    # Split to training, validation, and test sets

    for val_test_pct in np.linspace(0.1, 0.3, 21):
        if int(num_graphs * val_test_pct) > 2*batch_size:
            train_pct = 1 - val_test_pct
            val_pct = train_pct + (val_test_pct / 2)
            break
    train_dataset = {"edges": edges_mats[:int(num_graphs * train_pct)], "nodes": nodes_mats[:int(num_graphs * train_pct)]}
    val_dataset = {"edges": edges_mats[int(num_graphs * train_pct):int(num_graphs * val_pct)], "nodes": nodes_mats[int(num_graphs * train_pct):int(num_graphs * val_pct)]}
    test_dataset = {"edges": edges_mats[int(num_graphs * val_pct):], "nodes": nodes_mats[int(num_graphs * val_pct):]}
    if target_mats is not None:
        train_dataset["target"] = target_mats[:int(num_graphs * train_pct)]
        val_dataset["target"] = target_mats[int(num_graphs * train_pct):int(num_graphs * val_pct)]
        test_dataset["target"] = target_mats[int(num_graphs * val_pct):]
    else:
        train_dataset["target"] = train_dataset["edges"]
        val_dataset["target"] = val_dataset["edges"]
        test_dataset["target"] = test_dataset["edges"]
    sc = StandardScaler()

    if (train_dataset["nodes"] == 1).all() or (train_dataset["nodes"] == 0).all():
        return train_dataset, val_dataset, test_dataset, sc

    # normalize graph nodes data
    all_timesteps, num_features, num_nodes = nodes_mats.shape
    val_timesteps = val_dataset['nodes'].shape[0]
    stacked_tr_nodes_arr = train_dataset["nodes"].transpose(0, 2, 1).reshape(-1, num_features)
    stacked_val_nodes_arr = val_dataset["nodes"].transpose(0, 2, 1).reshape(-1, num_features)
    stacked_test_nodes_arr = test_dataset["nodes"].transpose(0, 2, 1).reshape(-1, num_features)

    norm_train_nodes_arr = sc.fit_transform(stacked_tr_nodes_arr)
    norm_val_nodes_arr = sc.transform(stacked_val_nodes_arr)
    norm_test_nodes_arr = sc.transform(stacked_test_nodes_arr)
    revers_norm_val_nodes_arr = sc.inverse_transform(norm_val_nodes_arr).reshape(val_timesteps, num_nodes, num_features).transpose(0, 2, 1)
    assert np.allclose(val_dataset["nodes"], revers_norm_val_nodes_arr), "Check the normalization process"

    train_dataset['nodes'] = norm_train_nodes_arr.reshape(-1, num_nodes, num_features).transpose(0, 2, 1)
    val_dataset['nodes'] = norm_val_nodes_arr.reshape(-1, num_nodes, num_features).transpose(0, 2, 1)
    test_dataset['nodes'] = norm_test_nodes_arr.reshape(-1, num_nodes, num_features).transpose(0, 2, 1)
    logger.info("===== Normalization info =====")
    logger.info(f"Graph_nodes, Number of features seen during fit: {sc.n_features_in_}\nvariance for each feature: {sc.var_}\nmean for each feature: {sc.mean_}\nscale for each feature: {sc.scale_}")

    if show_info:
        logger.info("===== Before normalization =====")
        logger.info(f"\nnodes_mats for first four nodes in t0, t1:\n{nodes_mats[:2, :, :4]}")
        logger.info("===== After normalization =====")
        logger.info(f"\ntrain_dataset['nodes'] for first four nodes in t0, t1:\n{train_dataset['nodes'][:2, :, :4]}")

    logger.info("="*80)

    return train_dataset, val_dataset, test_dataset, sc


def find_abs_max_cross_corr(x):
    """Finds the index of absolute-maximum cross correlation of a signal with itself, then return the correspond cross correlation.

    Args:
      x: The signal.

    Returns:
      The sign*(abs_maximum cross correlation) of the signal with itself.
    """

    cross_correlation = np.correlate(x, x, mode='full')
    lag = np.argmax(np.absolute(cross_correlation))
    return cross_correlation[lag]


def convert_str_bins_list(str_bins: str) -> list:
    """Converts a string of bins to a list of bins.

    Args:
      str_bins: A string of bins.

    Returns:
      A list of bins.
    """

    bins_list = []
    for str_bin in str_bins.replace("bins_", "").split("_"):
        if "-" in str_bin:
            new_str_bin = str_bin[:2]+"."+str_bin[2:]
            bins_list.append(float(new_str_bin))
        else:
            new_str_bin = str_bin[:1]+"."+str_bin[1:]
            bins_list.append(float(new_str_bin))

    return bins_list


def set_plot_log_data(log_path: Path):
    """Sets the data for plotting."""
    with open(log_path, "r") as f:
        log = json.load(f)

    tr_preds_history = log['tr_preds_history']
    tr_labels_history = log['tr_labels_history']
    val_preds_history = log['val_preds_history']
    val_labels_history = log['val_labels_history']
    #last_batch_size = len(tr_preds_history[0])
    graph_adj_mat_size = len(tr_preds_history[0][0])
    num_epochs = len(tr_preds_history)
    num_obs_epochs = 5
    obs_epochs = np.linspace(0, num_epochs-1, num_obs_epochs, dtype="int")
    obs_best_val_epoch = np.argmax(np.array(log['val_edge_acc_history']))
    obs_batch_idx = 10
    best_epoch_val_preds = np.array(val_preds_history[obs_best_val_epoch])
    best_epoch_val_labels = np.array(val_labels_history[obs_best_val_epoch])

    if sqrt(len(tr_preds_history[0][0])).is_integer() and len(tr_preds_history[0][0]) != 1:
        num_nodes = int(sqrt(len(tr_preds_history[0][0])))
        is_square_graph = True
    else:
        num_nodes_minus_one = 1
        while (num_nodes_minus_one**2 + num_nodes_minus_one)/2 != len(tr_preds_history[0][0]):  # arithmetic progression sum formula
            num_nodes_minus_one += 1
        num_nodes = num_nodes_minus_one+1
        is_square_graph = False
    assert is_square_graph == (graph_adj_mat_size == num_nodes**2), "when the graph is square graph, the size of graph should be num_nodes**2"
    assert len(tr_preds_history) == len(tr_labels_history) and len(tr_preds_history) == len(val_preds_history) and len(tr_preds_history) == len(val_labels_history), "length of {tr_preds_history, tr_labels_history, val_preds_history, val_labels_history} should be equal to num_epochs"
    return_dict = {key: value for key, value in locals().items() if key not in ['f', 'log', 'log_path', 'num_epochs', 'obs_best_val_epoch']}

    return return_dict

def plot_heatmap(preds: np.ndarray, labels: np.ndarray, save_fig_path: Path = None, can_show_conf_mat: bool = False):

    total_data_confusion_matrix = pd.DataFrame(confusion_matrix(labels.reshape(-1), preds.reshape(-1), labels=[0, 1, 2]), columns=range(-1, 2), index=range(-1, 2))
    plt.figure(figsize = (10, 10))
    plt.rcParams.update({'font.size': 44})
    ax = plt.gca()
    heatmap(total_data_confusion_matrix, annot=True, ax=ax, fmt='g')
    ax.set(xlabel="Prediction", ylabel="Ground Truth", title="val")
    if can_show_conf_mat:
        logger.info(f"confusion_matrix:\n{total_data_confusion_matrix}")
    if save_fig_path:
        plt.savefig(save_fig_path)
    plt.show()
    plt.close()
