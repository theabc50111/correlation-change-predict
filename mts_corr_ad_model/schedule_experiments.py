import argparse
import os
from datetime import datetime, timedelta
from itertools import chain, product, repeat
from pprint import pprint

data_implement_list = ["--data_implement SP500_20112015_CORR_SER_REG_STD_CORR_MAT_LARGE_FILTERED_HRCHY_10_CLUSTER_LABEL_LAST_V2"]  # "--data_implement LINEAR_REG_ONE_CLUSTER_DIM_30_BKPS_0_NOISE_STD_30"
batch_size_list = [""]
train_models_list = ["--train_models CLASSBASELINEONEFEATURE"]  # ["", "--train_models MTSCORRAD", "--train_models MTSCORRAD --train_models BASELINE", "--train_models MTSORRAD --train_models BASELINE --train_models GAE"]
corr_type_list = ["--corr_type pearson"]  # ["--corr_type pearson", "--corr_type cross_corr"]
seq_len_list = ["--seq_len 30"]  # ["--seq_len 5", "--seq_len 10"]
filt_mode_list = [""]  # ["", "--filt_mode keep_strong", "--filt_mode keep_positive", "--filt_mode keep_abs"]
filt_quan_list = [""]  # ["", "--filt_quan 0.25", "--filt_quan 0.5", "--filt_quan 0.75"]
quan_discrete_bins_list = [""]  # ["", "--quan_discrete_bins 2", "--quan_discrete_bins 3", "--quan_discrete_bins 4"]
custom_discrete_bins_list = [""]  # ["", "--custom_discrete_bins -1 --custom_discrete_bins 0 --custom_discrete_bins 1", "--custom_discrete_bins -1 --custom_discrete_bins -0.25 --custom_discrete_bins 0.25 --custom_discrete_bins 1", "--custom_discrete_bins -1 --custom_discrete_bins -0.5 --custom_discrete_bins 0 --custom_discrete_bins 0.5 --custom_discrete_bins 1"]
nodes_v_mode_list = [""]  # ["", "--graph_nodes_v_mode all_values", "--graph_nodes_v_mode mean", "--graph_nodes_v_mode mean_std"]
target_mats_path_list = ["--target_mats_path pearson/custom_discretize_graph_adj_mat/bins_-10_-025_025_10"]  # ["", "--target_mats_path pearson/custom_discretize_graph_adj_mat/bins_-10_-025_025_10", "--target_mats_path pearson/quan_discretize_graph_adj_mat/bin3"]
discr_loss_list = [""]  # ["" , "--discr_loss"]
discr_loss_r_list = [""]  # ["", "--discr_loss_r 0.1", "--discr_loss_r 0.01", "--discr_loss_r 0.001"]
discr_pred_disp_r_list = [""]  # ["", "--discr_pred_disp_r 1", "--discr_pred_disp_r 2", "--discr_pred_disp_r 5"]
learning_rate_list = [""]  # ["--learn_rate 0.0001", "--learn_rate 0.0005", "--learn_rate 0.001", "--learn_rate 0.005", "--learn_rate 0.01", "--learn_rate 0.05", "--learn_rate 0.1"]
weight_decay_list = [""]  # ["--weight_decay 0.0001", "--weight_decay 0.0005", "--weight_decay 0.001", "--weight_decay 0.005", "--weight_decay 0.01", "--weight_decay 0.05", "--weight_decay 0.1"]
graph_enc_weight_l2_reg_lambda_list = [""]  # ["", "--graph_enc_weight_l2_reg_lambda 0.01", "--graph_enc_weight_l2_reg_lambda 0.001"]
drop_pos_list = [""]  # ["", "--drop_pos gru", "--drop_pos decoder --drop_pos gru", "--drop_pos gru --drop_pos decoder --drop_pos graph_encoder"]
drop_p_list = [""]  # ["--drop_p 0.33", "--drop_p 0.5", "--drop_p 0.66"]
gra_enc_list = [""]  # ["", "--gra_enc gin", "--gra_enc gine"]
gra_enc_aggr_list = [""]  # ["", "--gra_enc_aggr mean", "--gra_enc_aggr add", "--gra_enc_aggr max"]
gra_enc_l_list = [""]  # ["--gra_enc_l 1", "--gra_enc_l 2", "--gra_enc_l 3", "--gra_enc_l 4", "--gra_enc_l 5"]
gra_enc_h_list = [""]  # ["--gra_enc_h 32", "--gra_enc_h 64", "--gra_enc_h 128", "--gra_enc_h 256", "--gra_enc_h 512"]
gra_enc_mlp_l_list = [""]  # ["--gra_enc_mlp_l 1", "--gra_enc_mlp_l 2", "--gra_enc_mlp_l 3"]
gru_l_list = [""]  # ["--gru_l 1", "--gru_l 2", "--gru_l 3", "--gru_l 4", "--gru_l 5"]
gru_h_list = [""]  # ["--gru_h 40", "--gru_h 80", "--gru_h 100", "--gru_h 320", "--gru_h 640"]
gru_input_feature_idx_list = ["--gru_input_feature_idx 0", "--gru_input_feature_idx 1", "--gru_input_feature_idx 2", "--gru_input_feature_idx 3", "--gru_input_feature_idx 4", "--gru_input_feature_idx 5"]  # ["", "--gru_input_feature_idx 0", "--gru_input_feature_idx 1", "--gru_input_feature_idx 2", "--gru_input_feature_idx 3", "--gru_input_feature_idx 4"]
use_weighted_loss_list = [""]  # ["", "--use_weighted_loss true"]
edge_acc_loss_atol_list = [""]  # ["", "--edge_acc_loss_atol 0.05", "--edge_acc_loss_atol 0.1", "--edge_acc_loss_atol 0.33"]
two_ord_pred_prob_edge_accu_thres_list = [""]  # ["", "--two_ord_pred_prob_edge_accu_thres 0.1", "--two_ord_pred_prob_edge_accu_thres 0.2"]
use_edge_acc_bins_list = [""]  # ["", "--use_bin_edge_acc_loss true"]
output_type_list = ["--output_type class_probability"]  # ["--output_type discretize", "--output_type class_probability"]
output_bins_list = [""]  # ["--output_bins -1 --output_bins -0.25 --output_bins --output_bins 0.25 --output_bins 1", "--output_bins -1 --output_bins -0.5 --output_bins 0 --output_bins 0.5 --output_bins 1", "--output_bins -1 --output_bins 0 --output_bins 1"]

args_values = list(product(data_implement_list, batch_size_list, train_models_list, corr_type_list, seq_len_list, filt_mode_list, filt_quan_list, quan_discrete_bins_list,
                           custom_discrete_bins_list, nodes_v_mode_list, target_mats_path_list, discr_loss_list, discr_loss_r_list, discr_pred_disp_r_list, learning_rate_list, weight_decay_list,
                           graph_enc_weight_l2_reg_lambda_list, drop_pos_list, drop_p_list, gra_enc_list, gra_enc_aggr_list, gra_enc_l_list, gra_enc_h_list, gra_enc_mlp_l_list, gru_l_list, gru_h_list,
                           gru_input_feature_idx_list, use_weighted_loss_list, edge_acc_loss_atol_list, two_ord_pred_prob_edge_accu_thres_list, use_edge_acc_bins_list, output_type_list, output_bins_list))
args_keys = ["data_implement", "batch_size", "train_models", "corr_type", "seq_len", "filt_mode", "filt_quan", "quan_discrete_bins", "custom_discrete_bins", "nodes_v_mode", "target_mats_path", "discr_loss", "discr_loss_r", "discr_pred_disp_r",
             "learning_rate", "weight_decay", "graph_enc_weight_l2_reg_lambda", "drop_pos", "drop_p", "gra_enc", "gra_enc_aggr", "gra_enc_l", "gra_enc_h", "gra_enc_mlp_l", "gru_l", "gru_h", "gru_input_feature_idx", "use_weighted_loss",
             "edge_acc_loss_atol", "two_ord_pred_prob_edge_accu_thres", "use_edge_acc_bins", "output_type", "output_bins"]
args_list = []
for args_value in args_values:
    args_dict = dict(zip(args_keys, args_value))
    args_list.append(args_dict)

#args_list = list(filter(lambda x: not (x["gra_enc_l"] == "--gra_enc_l 1" and x["gra_enc_h"] == "--gra_enc_h 32"), args_list))
#args_list = list(filter(lambda x: not (x["gra_enc_l"] == "--gra_enc_l 2" and x["gra_enc_h"] == "--gra_enc_h 32"), args_list))
args_list = list(filter(lambda x: not (x["gra_enc_l"] == "--gra_enc_l 5" and x["gra_enc_h"] == "--gra_enc_h 16"), args_list))
args_list = list(filter(lambda x: not ((not x["filt_mode"] and x["filt_quan"]) or (x["filt_mode"] and not x["filt_quan"])), args_list))
args_list = list(filter(lambda x: not ((not x["discr_loss"] and x["discr_loss_r"]) or (x["discr_loss"] and not x["discr_loss_r"])), args_list))
args_list = list(filter(lambda x: not ((not x["discr_loss"] and x["discr_pred_disp_r"]) or (x["discr_loss"] and not x["discr_pred_disp_r"])), args_list))
args_list = list(filter(lambda x: not ((not x["drop_pos"] and x["drop_p"]) or (x["drop_pos"] and not x["drop_p"])), args_list))  # Eliminate cases where either one of {drop_p, drop_pos} is null.
args_list = sorted(args_list, key=lambda x: int(x["gra_enc_h"].replace("--gra_enc_h ", ""))) if set(map(lambda x: x['gra_enc_h'], args_list)) != {""} else args_list
args_list = sorted(args_list, key=lambda x: int(x["gra_enc_l"].replace("--gra_enc_l ", "")), reverse=True) if set(map(lambda x: x['gra_enc_l'], args_list)) != {""} else args_list
args_list = sorted(args_list, key=lambda x: x["discr_loss"])

if set(map(lambda x: x['gra_enc_l'], args_list)) != {""}:
    gra_enc_l_values_set = set(map(lambda x: x['gra_enc_l'], args_list))
    gra_enc_l_values_set.discard("")
    gra_enc_l_pop_value = gra_enc_l_values_set.pop()
    num_models = sum(1 for x in args_list if x["discr_loss"] == "" and x["gra_enc_l"] == gra_enc_l_pop_value)  # the main reasons for model operation time: discr_loss, gra_enc_l
    model_timedelta_list = [timedelta(hours=5, minutes=30)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_lisbwwt
elif set(map(lambda x: x['gru_l'], args_list)) != {""}:
    gru_l_values_set = set(map(lambda x: x['gru_l'], args_list))
    gru_l_values_set.discard("")
    gru_l_pop_value = gru_l_values_set.pop()
    num_models = sum(1 for x in args_list if x['gru_l'] == gru_l_pop_value)  # the main reasons for model operation time: discr_loss, gra_enc_l, gra_enc_h
    model_timedelta_list = [timedelta(hours=1, minutes=20), timedelta(hours=1, minutes=20), timedelta(hours=1, minutes=25)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_list
else:
    num_models = len(args_list)
    model_timedelta_list = [timedelta(hours=1, minutes=20)]

model_timedelta_list = list(chain.from_iterable(repeat(x, num_models) for x in model_timedelta_list))
model_timedelta_list = [0] + model_timedelta_list
model_timedelta_list.pop()
assert len(args_list) == len(model_timedelta_list), f"The order of elements of model_timedelta_list〔length: {len(model_timedelta_list)}〕 should comply with the order of args_list: 〔length: {len(args_list)}〕.\nps. model_timedelta_list is based on num_modelsm num_models: {num_models}"
print(f"# len of experiments: {len(args_list)}")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--script", type=str, nargs='?', default="crontab_main.sh",
                             help="Input the name of operating script")
    args_parser.add_argument("--operating_time", type=str, nargs='?', default="+ 0:03",
                             help=(f"Input the operating time, the format of time: +/- hours:minutes.\n"
                                   f"For example:\n"
                                   f"    - postpone 1 hour and 3 minutes: \"+ 1:03\"\n"
                                   f"    - in advance 11 hour and 5 minutes: \"- 11:05\""))
    args_parser.add_argument("--cuda_device", type=int, nargs='?', default=0,
                             help="Input the gpu id")
    args_parser.add_argument("--log_suffix", type=str, nargs='?', default="",
                             help="Input the suffix of log file")
    ARGS = args_parser.parse_args()
    pprint(f"\n{vars(ARGS)}", indent=1, width=40, compact=True)
    operating_time_status = "postpone" if ARGS.operating_time.split(" ")[0] == "+" else "advance"
    operating_hours = int(ARGS.operating_time.split(" ")[1].split(":")[0])
    operating_minutes = int(ARGS.operating_time.split(" ")[1].split(":")[1].lstrip("0"))
    if operating_time_status == "postpone":
        experiments_start_t = datetime.now()+timedelta(hours=operating_hours, minutes=operating_minutes)
    elif operating_time_status == "advance":
        experiments_start_t = datetime.now()-timedelta(hours=operating_hours, minutes=operating_minutes)
    for i, (prev_model_time_len, model_args) in enumerate(zip(model_timedelta_list, args_list)):
        # print({"operate time length of previous model": prev_model_time_len, "model argumets": model_args})
        model_start_t = experiments_start_t if i == 0 else model_start_t + prev_model_time_len
        home_directory = os.path.expanduser("~")
        cron_args = [model_start_t.strftime("%M %H %d %m")+" *", home_directory, ARGS.script, f"--log_suffix {ARGS.log_suffix}", f"--cuda_device {ARGS.cuda_device}"] + list(model_args.values())
        args_cloze = " ".join(repeat("{}", len(model_args.values())))
        print(("{} {}/Documents/codes/correlation-change-predict/mts_corr_ad_model/{} {} {} "+args_cloze+" --save_model true").format(*cron_args))
