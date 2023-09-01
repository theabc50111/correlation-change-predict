#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from itertools import combinations
from pathlib import Path
import os
import sys
import warnings
import logging
from pprint import pformat
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import silhouette_score, make_scorer
from scipy.stats import uniform
from sklearn.metrics import fbeta_score
import networkx as nx
import dynamic_yaml
import yaml

sys.path.append("/workspace/correlation-change-predict/utils")
from utils import calc_corr_ser_property
from gen_corr_graph_data import gen_corr_dist_mat
from cluster_utils import (calc_silhouette_label_freq_std, hrchy_clustering_distance_threshold_rs, hrchy_clustering_n_cluster_gs,
                           obs_hrchy_cluster_instances, hrchy_cluster_fixed_n_cluster, plot_cluster_labels_distribution, plot_dendrogram)


current_dir = Path(os.getcwd())
data_config_path = current_dir / "../config/data_config.yaml"
with open(data_config_path) as f:
    data_cfg_yaml = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data_cfg_yaml))

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.ERROR)
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
# logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# print(logger_list)

# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501
logging.debug(pformat(data_cfg, indent=1, width=100, compact=True))


# # Prepare data

# ## Data implement & output setting & testset setting

# In[ ]:


# data implement setting
data_implement = "SP500_20112015"  # watch options by printing /config/data_config.yaml/["DATASETS"].keys()
# print(data_cfg["DATASETS"].keys())
# etl set setting
etl_items_setting = "-train_all"  # -train_train|-train_all
# data split period setting, only suit for only settings of Korean paper
data_split_setting = "data_sp_test2"
# set correlation type
corr_type = "pearson"  # "pearson" | "cross_corr"
# set CORR_WINDOW and CORR_STRIDE length
w_l=50 ; s_l = 1
# Decide how to calculate corr_ser
corr_ser_clac_method = "corr_ser_calc_regular"  # corr_ser_calc_regular|corr_ser_calc_abs
# Decide correlation series reduction method
corr_ser_reduction_method = "corr_ser_std" # "corr_ser_std" | "corr_ser_mean"
# Decide composition of correlation_matrix
corr_mat_compo = "sim"
# Decide how to filter distanc_mat
filtered_distance_mat_method = "large_corr_ser_mean_filtered"  # no_filtered|large_corr_ser_mean_filtered|small_corr_ser_mean_filtered
# Decide whether save output or not
is_save_output = False


# In[ ]:


dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
dataset_df = dataset_df.set_index('Date')
all_set = list(dataset_df.columns)  # all data
train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
test_set = data_cfg['DATASETS'][data_implement]['TEST_SET'] if data_cfg['DATASETS'][data_implement].get('TEST_SET') else [p for p in all_set if p not in train_set]  # all data - train data
logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

# test items implement settings
items_implement = train_set if etl_items_setting == "-train_train" else all_set
logging.info(f"===== len(etl set): {len(items_implement)} =====")

# setting of name of output files and pictures title
output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + etl_items_setting
fig_title = f"{data_implement}{etl_items_setting}-{data_split_setting}"
logging.info(f"===== file_name basis:{output_file_name}, fig_title basis:{fig_title} =====")
# display(dataset_df)

# input folder settings
corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{corr_type}/corr_data"
assert corr_data_dir.exists(), f"\nOperate data_module.py to generate correlation data first, operate command:\n    python data_module.py --data_implement SP500_20082017 --train_items_setting -train_all --data_split_setting -data_sp_test2 --graph_mat_compo sim --save_corr_data"

# output folder settings
res_dir= Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{corr_type}/corr_property/corr_s{s_l}_w{w_l}/{corr_ser_clac_method}"
cluster_items_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}/{corr_type}/cluster/corr_s{s_l}_w{w_l}/{corr_ser_clac_method}/{corr_ser_reduction_method}/{corr_mat_compo}/{filtered_distance_mat_method}"
res_dir.mkdir(parents=True, exist_ok=True)
cluster_items_dir.mkdir(parents=True, exist_ok=True)


# ## Load or Create Correlation Data

# In[ ]:


train_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_train.csv"
dev_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_dev.csv"
test1_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_test1.csv"
test2_df_path = corr_data_dir/f"corr_s{s_l}_w{w_l}_test2.csv"
if data_split_setting == "data_sp_train":
    df_path = train_df_path
elif data_split_setting == "data_sp_valid":
    df_path = dev_df_path
elif data_split_setting == "data_sp_test1":
    df_path = test1_df_path
elif data_split_setting == "data_sp_test2":
    df_path = test2_df_path
corr_dataset = pd.read_csv(df_path, index_col=["items"])
logging.info(f"corr_dataset.shape:{corr_dataset.shape}")
display(corr_dataset.head())


# In[ ]:


# item_pair = "EXR & RCL_0"  #  set by chooing one of column `items` of `corr_dataset`
# plt.figure(figsize=(12,6))
# plt.plot(corr_dataset.loc[item_pair, ::].T)
# plt.title(f"{item_pair} with corr_s{s_l}_w{w_l}", fontsize=20)
# plt.xticks(np.arange(0, corr_dataset.shape[1], 250), rotation=45)
# plt.show()
# plt.close()


# # Calculate properties of Corrlelation series

# In[ ]:


if corr_ser_clac_method == "corr_ser_calc_regular":
    corr_property_df_path = res_dir/f"{output_file_name}-{data_split_setting}-corr_series_property.csv"
    corr_property_df = calc_corr_ser_property(corr_dataset=corr_dataset, corr_property_df_path=corr_property_df_path)
elif corr_ser_clac_method == "corr_ser_calc_abs":
    # calculate corr_property_df with abs(corr_dataset)
    corr_property_df_path = res_dir/f"{output_file_name}-{data_split_setting}-corr_series_abs_property.csv"
    corr_property_df = calc_corr_ser_property(corr_dataset=corr_dataset.abs(), corr_property_df_path=corr_property_df_path)

corr_property_df_mean_reduct_series = corr_property_df.mean(axis=0)
corr_property_df_mean_reduct_series.name = "corr_property_df_mean_reduct_series"
corr_property_df_std_reduct_series = corr_property_df.std(axis=0)
corr_property_df_std_reduct_series.name = "corr_property_df_std_reduct_series"

logging.info(f"{fig_title} + corr_w{w_l} + corr_s{s_l} + corr_ser_clac_method:{corr_ser_clac_method}")
logging.info(f"Min of corr_ser_mean:{corr_property_df.loc[::,'corr_ser_mean'].min()}")
display(corr_property_df.head())
print("="*50)
display(corr_property_df_mean_reduct_series)
print("="*50)
display(corr_property_df_std_reduct_series)


# ## plot distribution of all correlation of all item_pairs

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(6, 10)
all_item_pair_corrs = np.hstack(corr_dataset.values)
axes[0].hist(all_item_pair_corrs, bins=20)
axes[0].xaxis.set_tick_params(labelsize=18)
axes[1].boxplot(all_item_pair_corrs)
axes[1].yaxis.set_tick_params(labelsize=18)
plt.show()
plt.close()


# # Clustring

# ## calculate distance matrix and
# ## output distance matrix

# In[ ]:


corr_ser_std = corr_property_df.loc[::, "corr_ser_std"]
corr_ser_mean = corr_property_df.loc[::, "corr_ser_mean"]
corr_reduction_ser = corr_ser_std if corr_ser_reduction_method == "corr_ser_std" else corr_ser_mean
selected_dataset_df = dataset_df.loc[::, items_implement]
distance_mat = gen_corr_dist_mat(corr_reduction_ser, selected_dataset_df, out_mat_compo=corr_mat_compo)
corr_ser_mean_q2 = corr_property_df['corr_ser_mean'].quantile(0.5)
filtered_distance_mat_settings = {"large_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) > corr_ser_mean_q2),
                                  "small_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) < corr_ser_mean_q2),
                                  "positive_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) > 0),
                                  "negative_corr_ser_mean_filtered": ~(gen_corr_dist_mat(corr_ser_mean, selected_dataset_df, out_mat_compo=corr_mat_compo) < 0)}

filtered_distance_mat_method = "large_corr_ser_mean_filtered"  # no_filtered|large_corr_ser_mean_filtered|small_corr_ser_mean_filtered
if (filter_mask := filtered_distance_mat_settings.get(filtered_distance_mat_method)) is not None:
    distance_mat[filter_mask] = 0
    G = nx.from_pandas_adjacency(distance_mat)
    all_cliques = list(nx.find_cliques(G))
    max_clique = max(all_cliques, key=len)
    distance_mat = distance_mat.loc[max_clique, max_clique]

display(distance_mat.head())
if is_save_output:
    distance_mat.to_csv(cluster_items_dir/"distance_mat.csv")
    logging.info(f"distance_mat.csv has been save to {cluster_items_dir}")
    with open(cluster_items_dir/"max_clique_nodes.txt", "w") as f:
        f.write(str(max_clique))


# In[ ]:


# test
# test_stock_tickers = ["ED", "BAC", "XEL", "MA"]
# test_distance_mat = distance_mat.loc[test_stock_tickers, test_stock_tickers]
# display(test_distance_mat)  # comlpete: (ED, BAC), (XEL), (MA) -> (ED, BAC), (XEL, MA)  -> (ED, BAC, XEL, MA)
#                             # single: (ED, BAC), (XEL), (MA) -> (ED, BAC, XEL), (MA)  -> (ED, BAC, XEL, MA)


# ## calculate cluster label for each data

# In[ ]:


obs_hrchy_cluster_instances(distance_mat)
num_clusters = 11  # Determin by observe result of obs_hrchy_cluster_instances()
fixed_n_cluster_hrchy_cluster = hrchy_cluster_fixed_n_cluster(distance_mat, n=num_clusters)
# distance_threshold_hrchy_cluster = hrchy_clustering_distance_threshold_rs(dissimilarity_mat, verbose=0)
# n_cluster_hrchy_cluster = hrchy_clustering_n_cluster_gs(distance_mat, verbose=0)


# ## output cluster results

# In[ ]:


if num_clusters:
    output_cluster = fixed_n_cluster_hrchy_cluster  # the value of output_cluster depend on performance which shows in plot cluster label distribution
    output_cluster_name = f"corr_mat_hrchy_{num_clusters}_cluster"

hrchy_cluster_labels_df = pd.DataFrame(output_cluster.labels_, index=distance_mat.index, columns=[f"{output_cluster_name}_label"]).reset_index()
if is_save_output:
    hrchy_cluster_labels_df.to_csv(cluster_items_dir/f"{output_cluster_name}.csv")
    logging.info(f"{output_cluster_name}.csv has been save to {cluster_items_dir}")


# ## plot cluster info

# In[ ]:


# plot cluster label distribution
plot_cluster_labels_distribution(output_cluster, cluster_name=f"Hirerarchy clustering with {num_clusters} n_clusters", fig_title=fig_title, save_dir=cluster_items_dir)
plot_dendrogram(output_cluster, truncate_mode="level", p=4, save_dir=cluster_items_dir)

hrchy_cluster_labels_df = hrchy_cluster_labels_df.set_index("items") if "items" in hrchy_cluster_labels_df.columns else hrchy_cluster_labels_df
for cluster_label in range(output_cluster.n_clusters_):
    same_cluster_items = hrchy_cluster_labels_df.where(hrchy_cluster_labels_df == cluster_label).dropna(thresh=1).index
    same_cluster_distance_mat = distance_mat.loc[same_cluster_items, same_cluster_items]
    print("The distance_mat of cluster_{cluster_label}:")
    display(same_cluster_distance_mat)
    if is_save_output:
        same_cluster_distance_mat.to_csv(cluster_items_dir/f"distance_mat-{output_cluster_name}-cluster_label_{cluster_label}.csv")


# # plot correlation coffecient distribution of data

# In[ ]:


datasets = {'train_data--comb(150,2)': train_corr_series_concat, 'all_data--comb(445,2)': all_corr_series_concat, 'other_data--comb(295,2)': other_corr_series_concat}
etl_types = ["boxplot", "histogram", "qqplot", "Emprical Cumulative Density"]
fig, axes = plt.subplots(figsize=(20, 20),nrows=len(etl_types), ncols=len(datasets), sharex=False, sharey=False, dpi=100)

for row, etl_type in enumerate(etl_types):
    for col,dataset_key in enumerate(datasets):
        # print(row, etl_type, col, dataset_key, datasets[dataset_key])
        s = axes[row, col]
        s.set_title(f"{dataset_key}: \n{etl_type}", fontsize=24)
        if etl_type=="boxplot":
            s.boxplot(datasets[dataset_key], showmeans=True)
        elif etl_type=="histogram":
            s.hist(datasets[dataset_key], bins=[b/10 for b in range(-13,14)])
        elif etl_type=="qqplot":
            percents = [0.001, 0.2, 0.5, 0.8, 0.999]
            #x,y = [norm.ppf(p) for p in percents], [np.quantile(train_corr_series_concat, p) for p in percents]
            x,y = [norm.ppf(p) for p in percents], [np.quantile(datasets[dataset_key], p) for p in percents]
            sm.qqplot(datasets[dataset_key], line='q', ax=s)
            s.scatter(x,y, c='m', marker='x', s=300)
        elif etl_type=="Emprical Cumulative Density":
            pd.Series(datasets[dataset_key]).value_counts().sort_index().cumsum().plot(ax=s)

# 分開, 避免子圖標籤互相重疊
plt.tight_layout()
plt.savefig("./results/dataset_exploration.png")
plt.show()
plt.close()


# In[ ]:


df = pd.DataFrame([[dataset_key, datasets[dataset_key].std()] for dataset_key in datasets], 
                  columns=['Dataset', 'Standard deviation'])
ax = sns.barplot(x='Dataset', y='Standard deviation', data=df)
ax.set_title('std of correlation')
ax.set(ylim=[0.47, 0.475])
ax.bar_label(ax.containers[0])
plt.xticks(rotation=60)
plt.savefig("./results/dataset_exploration_2.png")
plt.show()
plt.close()


# In[ ]:


sns.distplot(train_corr_series_concat)
# plt.hist(train_corr_series, bins=[b/10 for b in range(-13,14)])


# In[ ]:


train_corr_series_df = gen_corr_series(None, "train_dataset.csv", from_file=True, concat_all=False)
all_corr_series_df = gen_corr_series(None, "445_dataset.csv", from_file=True, concat_all=False)
other_corr_series_df = gen_corr_series(None, "295_dataset.csv", from_file=True, concat_all=False)


# In[ ]:


datasets = {'train_data--comb(150,2)': train_corr_series_df, 'all_data--comb(445,2)': all_corr_series_df, 'other_data--comb(295,2)': other_corr_series_df}
etl_types = ["boxplot", "histogram"]
static_types = ["mean", "std"]
fig, axes = plt.subplots(figsize=(30, 30),nrows=len(list(product(etl_types, static_types))), ncols=len(datasets), sharex=False, sharey=False, dpi=100)

for row, (etl_type, static_type) in enumerate(product(etl_types, static_types)):
    for col,dataset_key in enumerate(datasets):
        s = axes[row, col]
        s.set_title(f"{dataset_key}: \n{etl_type}_{static_type}", fontsize=24)
        if etl_type=="boxplot":
            s.boxplot(datasets[dataset_key].iloc[:, ::5].describe().loc[static_type,:], showmeans=True)
        elif etl_type=="histogram":
            s.hist(datasets[dataset_key].iloc[:, ::5].describe().loc[static_type,:], bins=[b/10 for b in range(-13,14)])

fig.suptitle(f"Each correlation_series static property _20220718", fontsize=24)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# 分開, 避免子圖標籤互相重疊
# plt.tight_layout()
plt.savefig("./results/dataset_exploration_3.png")
plt.show()
plt.close()


# In[ ]:


display(train_corr_series_df)
display(train_corr_series_df.iloc[:,::5])
display(train_corr_series_df.iloc[:,::5].describe())
display(train_corr_series_df.iloc[:,::5].describe().loc['std',:])

