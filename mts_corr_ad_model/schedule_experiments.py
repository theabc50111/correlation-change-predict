import argparse
import os
from datetime import datetime, timedelta
from itertools import chain, product, repeat
from pprint import pprint

filt_mode_list = [""]  # ["", "--filt_mode keep_strong", "--filt_mode keep_positive", "--filt_mode keep_abs"]
filt_quan_list = [""]  # ["", "--filt_quan 0.25", "--filt_quan 0.5", "--filt_quan 0.75"]
nodes_v_mode_list = [""]  # ["", "--graph_nodes_v_mode all_values", "--graph_nodes_v_mode mean", "--graph_nodes_v_mode mean_std"]
discr_loss_list = [""]  # ["" , "--discr_loss"]
discr_loss_r_list = [""]  # ["", "--discr_loss_r 0.1", "--discr_loss_r 0.01", "--discr_loss_r 0.001"]
discr_pred_disp_r_list = [""]  # ["", "--discr_pred_disp_r 1", "--discr_pred_disp_r 2", "--discr_pred_disp_r 5"]
weight_decay_list = [""]  # ["--weight_decay 0.0001", "--weight_decay 0.0005", "--weight_decay 0.001", "--weight_decay 0.005", "--weight_decay 0.01", "--weight_decay 0.05", "--weight_decay 0.1"]
graph_enc_weight_l2_reg_lambda_list = ["--graph_enc_weight_l2_reg_lambda 0.01", "--graph_enc_weight_l2_reg_lambda 0.001"]  # ["", "--graph_enc_weight_l2_reg_lambda 0.01", "--graph_enc_weight_l2_reg_lambda 0.001"]
drop_pos_list = ["", "--drop_pos graph_encoder"]  # ["", "--drop_pos gru", "--drop_pos decoder --drop_pos gru", "--drop_pos gru --drop_pos decoder --drop_pos graph_encoder"]
drop_p_list = ["", "--drop_p 0.1", "--drop_p 0.01"]  # ["--drop_p 0.33", "--drop_p 0.5", "--drop_p 0.66"]
gra_enc_list = [""]  # ["", "--gra_enc gin", "--gra_enc gine"]
gra_enc_aggr_list = [""]  # ["", "mean", "add", "max"]
gra_enc_l_list = ["--gra_enc_l 2"]
gra_enc_h_list = ["--gra_enc_h 4"]

args_list = list(product(filt_mode_list, filt_quan_list, nodes_v_mode_list, discr_loss_list, discr_loss_r_list, discr_pred_disp_r_list, weight_decay_list, graph_enc_weight_l2_reg_lambda_list, drop_pos_list, drop_p_list, gra_enc_list, gra_enc_aggr_list, gra_enc_l_list, gra_enc_h_list))
args_list = list(filter(lambda x: not (x[12] == "--gra_enc_l 1" and x[13] == "--gra_enc_h 32"), args_list))
args_list = list(filter(lambda x: not (x[12] == "--gra_enc_l 2" and x[13] == "--gra_enc_h 32"), args_list))
args_list = list(filter(lambda x: not (x[12] == "--gra_enc_l 5" and x[13] == "--gra_enc_h 16"), args_list))
args_list = list(filter(lambda x: not (x[0] == "" and x[1] == "--filt_quan 0.25"), args_list))
args_list = list(filter(lambda x: not (x[0] == "" and x[1] == "--filt_quan 0.75"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[4] == "--discr_loss_r 0.1"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[4] == "--discr_loss_r 0.001"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[5] == "--discr_pred_disp_r 1"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[5] == "--discr_pred_disp_r 5"), args_list))
args_list = list(filter(lambda x: not ((not x[8] and x[9]) or (x[8] and not x[9])), args_list))  # Eliminate cases where either one of {drop_p, drop_pos} is null.
args_list = sorted(args_list, key=lambda x: int(x[13].replace("--gra_enc_h ", "")))
args_list = sorted(args_list, key=lambda x: int(x[12].replace("--gra_enc_l ", "")))
args_list = sorted(args_list, key=lambda x: x[3])

#if "--discr_loss" in discr_loss_list:
#    num_models = sum([1 for x in args_list if x[3] == "--discr_loss" and x[10] == "--gra_enc_l 5" and x[11] == "--gra_enc_h 16"])  # the main reasons for model operation time: discr_loss, gra_enc_l, gra_enc_h
#    model_timedelta_list = [timedelta(minutes=20), timedelta(minutes=55), timedelta(hours=1, minutes=20), timedelta(hours=1),  # The order of elements of model_timedelta_list should comply with the order of elements of args_list
#                            timedelta(hours=1, minutes=10), timedelta(hours=1, minutes=40), timedelta(hours=3, minutes=20), timedelta(hours=3, minutes=5)]
#else:
#    num_models = sum([1 for x in args_list if x[3] == "" and x[10] == "--gra_enc_l 1" and x[11] == "--gra_enc_h 4"])  # the main reasons for model operation time: discr_loss, gra_enc_l, gra_enc_h
#    model_timedelta_list = [timedelta(minutes=20), timedelta(minutes=55), timedelta(hours=1, minutes=20), timedelta(hours=1)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_list

num_models = sum([1 for x in args_list if x[3] == "" and x[12] == "--gra_enc_l 2"])  # the main reasons for model operation time: discr_loss, gra_enc_l
#model_timedelta_list = [timedelta(hours=3, minutes=40), timedelta(hours=5, minutes=20), timedelta(hours=10, minutes=0)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_list
model_timedelta_list = [timedelta(hours=5, minutes=50)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_list

model_timedelta_list = list(chain.from_iterable(repeat(x, num_models) for x in model_timedelta_list))
model_timedelta_list = [0] + model_timedelta_list
model_timedelta_list.pop()
assert len(args_list) == len(model_timedelta_list), f"The order of elements of model_timedelta_list〔length: {len(model_timedelta_list)}〕 should comply with the order of args_list: 〔length: {len(args_list)}〕"
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
        cron_args = [model_start_t.strftime("%M %H %d %m")+" *", home_directory, ARGS.script, f"--log_suffix {ARGS.log_suffix}", f"--cuda_device {ARGS.cuda_device}"] + list(model_args)
        print("{} {}/Documents/codes/correlation-change-predict/mts_corr_ad_model/{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} --save_model true".format(*cron_args))
        # if x[9]==1 and x[10]==4:
        #     print(prev_model_time_len, model_args)
