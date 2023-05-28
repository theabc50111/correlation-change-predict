# Function: clustering utilities

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.pyplot import MultipleLocator
from scipy.stats import uniform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def calc_silhouette_label_freq_std(estimator: sklearn.base.ClusterMixin, x: pd.DataFrame, silhouette_score_ratio: int = 0.1, silhouette_metric: str = "precomputed") -> float:
    estimator.fit(x)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(x.index)
    labels_symbol, label_freq = np.unique(cluster_labels, return_counts=True)
    if num_labels in (1, num_samples):
        return -1

    return silhouette_score_ratio * silhouette_score(x, cluster_labels, metric=silhouette_metric) + (1 - silhouette_score_ratio) * (1 / np.array(label_freq).std())


def hrchy_clustering_distance_threshold_rs(x: pd.DataFrame, data_mat_mode: str = "precomputed", verbose: int = 0):
    param_dict = {"n_clusters": [None], "affinity": [data_mat_mode],
                  "linkage": ["single", "complete", "average"],
                  "distance_threshold": uniform(loc=0.55, scale=0.6)}
    cv = [(slice(None), slice(None))]
    hrchy_clustering_rs = RandomizedSearchCV(estimator=AgglomerativeClustering(), param_distributions=param_dict,
                                             n_iter=100000, scoring=calc_silhouette_label_freq_std, cv=cv, n_jobs=-1)
    hrchy_clustering_rs.fit(x)

    if verbose == 1:
        print(f"hrchy_clustering_rs.best_estimator_: {hrchy_clustering_rs.best_estimator_}")
        print(f"hrchy_clustering_rs.best_params_: {hrchy_clustering_rs.best_params_}")
        print(f"hrchy_clustering_rs.best_score_: {hrchy_clustering_rs.best_score_}")
        print(f"hrchy_clustering_rs.best_estimator_.n_leaves_: {hrchy_clustering_rs.best_estimator_.n_leaves_}")
        print(f"hrchy_clustering_rs.best_estimator_.n_clusters_: {hrchy_clustering_rs.best_estimator_.n_clusters_}")
        print(f"np.unique(hrchy_clustering_rs.best_estimator_.labels_): {np.unique(hrchy_clustering_rs.best_estimator_.labels_, return_counts=True)}")
        print(f"hrchy_clustering_rs.best_estimator_.labels_: {hrchy_clustering_rs.best_estimator_.labels_}")
        print(f"hrchy_clustering_rs.n_features_in_: {hrchy_clustering_rs.n_features_in_}")
        print(f"hrchy_clustering_rs.feature_names_in_: {hrchy_clustering_rs.feature_names_in_}")
        print("-"*50)

    return hrchy_clustering_rs.best_estimator_


def hrchy_clustering_n_cluster_gs(x: pd.DataFrame, data_mat_mode: str = "precomputed", verbose: int = 0):
    param_dict = {"n_clusters": range(2, 20), "affinity": [data_mat_mode],
                  "linkage": ["single", "complete", "average"]}
    cv = [(slice(None), slice(None))]
    hrchy_clustering_gs = GridSearchCV(estimator=AgglomerativeClustering(), param_grid=param_dict,
                                       scoring=calc_silhouette_label_freq_std, cv=cv, n_jobs=-1)
    hrchy_clustering_gs.fit(x)

    if verbose == 1:
        print(f"hrchy_clustering_gs.best_estimator_: {hrchy_clustering_gs.best_estimator_}")
        print(f"hrchy_clustering_gs.best_params_: {hrchy_clustering_gs.best_params_}")
        print(f"hrchy_clustering_gs.best_score_: {hrchy_clustering_gs.best_score_}")
        print(f"hrchy_clustering_gs.best_estimator_.n_leaves_: {hrchy_clustering_gs.best_estimator_.n_leaves_}")
        print(f"hrchy_clustering_gs.best_estimator_.n_clusters_: {hrchy_clustering_gs.best_estimator_.n_clusters_}")
        print(f"np.unique(hrchy_clustering_gs.best_estimator_.labels_): {np.unique(hrchy_clustering_gs.best_estimator_.labels_, return_counts=True)}")
        print(f"hrchy_clustering_gs.best_estimator_.labels_: {hrchy_clustering_gs.best_estimator_.labels_}")
        print(f"hrchy_clustering_gs.n_features_in_: {hrchy_clustering_gs.n_features_in_}")
        print(f"hrchy_clustering_gs.feature_names_in_: {hrchy_clustering_gs.feature_names_in_}")
        print("-"*50)

    return hrchy_clustering_gs.best_estimator_


def obs_hrchy_cluster_instances(x: pd.DataFrame, data_mat_mode: str = "precomputed", verbose: int = 1):
    for n in range(2, 20):
        hrchy_cluster = AgglomerativeClustering(n_clusters=n, linkage="complete", affinity=data_mat_mode, compute_distances=True)
        hrchy_cluster.fit(x)

        if verbose == 1:
            print(f"hrchy_cluste.n_clusters_: {hrchy_cluster.n_clusters_}")
            print(f"hrchy_cluste.labels and whose number of instances: {np.unique(hrchy_cluster.labels_, return_counts=True)}")
            # print(f"(ticker, cluster label): {list(zip(X.index, hrchy_cluster.labels_))}")
            # print(f"The estimated number of connected components:{hrchy_cluster.n_connected_components_}")
            # print(f"hrchy_cluster.n_leaves_: {hrchy_cluster.n_leaves_}")
            # print(f"hrchy_cluster.n_features_in_: {hrchy_cluster.n_features_in_}")
            print("-"*50)


def hrchy_cluster_fixed_n_cluster(x: pd.DataFrame, n: int, data_mat_mode: str = "precomputed", verbose: int = 1):
    hrchy_cluster = AgglomerativeClustering(n_clusters=n, linkage="complete", affinity=data_mat_mode)
    hrchy_cluster.fit(x)

    if verbose == 1:
        print(f"hrchy_cluste.n_clusters_: {hrchy_cluster.n_clusters_}")
        print(f"hrchy_cluste.labels and whose instances: {np.unique(hrchy_cluster.labels_, return_counts=True)}")
        print(f"hrchy_cluster.n_leaves_: {hrchy_cluster.n_leaves_}")
        print(f"hrchy_cluster.n_features_in_: {hrchy_cluster.n_features_in_}")
        print("-"*50)

    return hrchy_cluster


def plot_cluster_labels_distribution(trained_cluster: sklearn.base.ClusterMixin, cluster_name: str, fig_title: str):
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.bar(np.unique(trained_cluster.labels_, return_counts=True)[0], np.unique(trained_cluster.labels_, return_counts=True)[1])
    plt.grid()
    plt.ylabel("instances in cluster")
    plt.xlabel("cluster label")
    plt.title(f"{cluster_name}\n {fig_title}")
    plt.show()  # findout elbow point
    plt.close()
    logging.info(f"cluster of each point distribution: {np.unique(trained_cluster.labels_, return_counts=True)}")
