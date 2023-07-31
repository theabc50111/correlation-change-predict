import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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


def split_and_norm_data(edges_mats: np.ndarray, nodes_mats: np.ndarray, target_mats: np.ndarray, show_info: bool = False):
    """
    split dataset to train, validation, test
    normalize these dataset
    """
    assert len(edges_mats) == len(nodes_mats), "Check the whether gra_edges_data_mats match gra_nodes_data_mats"
    num_graphs = len(nodes_mats)

    # Split to training, validation, and test sets
    train_dataset = {"edges": edges_mats[:int(num_graphs * 0.9)], "nodes": nodes_mats[:int(num_graphs * 0.9)]}
    val_dataset = {"edges": edges_mats[int(num_graphs * 0.9):int(num_graphs * 0.95)], "nodes": nodes_mats[int(num_graphs * 0.9):int(num_graphs * 0.95)]}
    test_dataset = {"edges": edges_mats[int(num_graphs * 0.95):], "nodes": nodes_mats[int(num_graphs * 0.95):]}
    if target_mats is not None:
        train_dataset["target"] = target_mats[:int(num_graphs * 0.9)]
        val_dataset["target"] = target_mats[int(num_graphs * 0.9):int(num_graphs * 0.95)]
        test_dataset["target"] = target_mats[int(num_graphs * 0.95):]
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
