from datetime import datetime, timedelta
from itertools import chain, product, repeat

filt_mode_list = [""]  # ["", "--filt_mode keep_strong", "--filt_mode keep_positive", "--filt_mode keep_abs"]
filt_quan_list = [""]  # ["", "--filt_quan 0.25", "--filt_quan 0.5", "--filt_quan 0.75"]
nodes_v_mode_list = ["", "--graph_nodes_v_mode all_values", "--graph_nodes_v_mode mean", "--graph_nodes_v_mode mean_std"]
discr_loss_list = [""]  # ["" , "--discr_loss"]
discr_loss_r_list = [""]  # ["", "--discr_loss_r 0.1", "--discr_loss_r 0.01", "--discr_loss_r 0.001"]
discr_pred_disp_r_list = [""]  # ["", "--discr_pred_disp_r 1", "--discr_pred_disp_r 2", "--discr_pred_disp_r 5"]
drop_pos_list = [""]  # ["", "--drop_pos gru", "--drop_pos decoder --drop_pos gru"]
drop_p_list = [""]  # ["--drop_p 0.33", "--drop_p 0.5", "--drop_p 0.66"]
gra_enc_list = ["--gra_enc gine"]  # ["--gra_enc gin", "--gra_enc gine"]
gra_enc_aggr_list = ["--gra_enc_aggr mean", "--gra_enc_aggr add", "--gra_enc_aggr max"]  # ["mean", "add", "max"]
gra_enc_l_list = ["--gra_enc_l 1", "--gra_enc_l 2", "--gra_enc_l 5"]
gra_enc_h_list = ["--gra_enc_h 4", "--gra_enc_h 16", "--gra_enc_h 32"]

args_list = list(product(filt_mode_list, filt_quan_list, nodes_v_mode_list, discr_loss_list, discr_loss_r_list, discr_pred_disp_r_list, drop_pos_list, drop_p_list, gra_enc_list, gra_enc_aggr_list, gra_enc_l_list, gra_enc_h_list))
args_list = list(filter(lambda x: not (x[10] == "--gra_enc_l 1" and x[11] == "--gra_enc_h 32"), args_list))
args_list = list(filter(lambda x: not (x[10] == "--gra_enc_l 2" and x[11] == "--gra_enc_h 32"), args_list))
args_list = list(filter(lambda x: not (x[10] == "--gra_enc_l 5" and x[11] == "--gra_enc_h 16"), args_list))
args_list = list(filter(lambda x: not (x[0] == "" and x[1] == "--filt_quan 0.25"), args_list))
args_list = list(filter(lambda x: not (x[0] == "" and x[1] == "--filt_quan 0.75"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[4] == "--discr_loss_r 0.1"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[4] == "--discr_loss_r 0.001"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[5] == "--discr_pred_disp_r 1"), args_list))
args_list = list(filter(lambda x: not (x[3] == "" and x[5] == "--discr_pred_disp_r 5"), args_list))
args_list = list(filter(lambda x: not (x[6] == "" and x[7] == "--drop_p 0.33"), args_list))
args_list = list(filter(lambda x: not (x[6] == "" and x[7] == "--drop_p 0.66"), args_list))
args_list = sorted(args_list, key=lambda x: int(x[11].replace("--gra_enc_h ", "")))
args_list = sorted(args_list, key=lambda x: int(x[10].replace("--gra_enc_l ", "")))
args_list = sorted(args_list, key=lambda x: x[3])

#if "--discr_loss" in discr_loss_list:
#    num_models = sum([1 for x in args_list if x[3] == "--discr_loss" and x[10] == "--gra_enc_l 5" and x[11] == "--gra_enc_h 16"])  # the main reasons for model operation time: discr_loss, gra_enc_l, gra_enc_h
#    model_timedelta_list = [timedelta(minutes=20), timedelta(minutes=55), timedelta(hours=1, minutes=20), timedelta(hours=1),  # The order of elements of model_timedelta_list should comply with the order of elements of args_list
#                            timedelta(hours=1, minutes=10), timedelta(hours=1, minutes=40), timedelta(hours=3, minutes=20), timedelta(hours=3, minutes=5)]
#else:
#    num_models = sum([1 for x in args_list if x[3] == "" and x[10] == "--gra_enc_l 1" and x[11] == "--gra_enc_h 4"])  # the main reasons for model operation time: discr_loss, gra_enc_l, gra_enc_h
#    model_timedelta_list = [timedelta(minutes=20), timedelta(minutes=55), timedelta(hours=1, minutes=20), timedelta(hours=1)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_list

num_models = sum([1 for x in args_list if x[3] == "" and x[10] == "--gra_enc_l 1"])  # the main reasons for model operation time: discr_loss, gra_enc_l
model_timedelta_list = [timedelta(hours=3, minutes=50), timedelta(hours=5, minutes=30), timedelta(hours=10, minutes=30)]  # The order of elements of model_timedelta_list should comply with the order of elements of args_list

model_timedelta_list = list(chain.from_iterable(repeat(x, num_models) for x in model_timedelta_list))
model_timedelta_list = [0] + model_timedelta_list
model_timedelta_list.pop()
assert len(args_list) == len(model_timedelta_list), f"The order of elements of model_timedelta_list〔length: {len(model_timedelta_list)}〕 should comply with the order of args_list: 〔length: {len(args_list)}〕"
print(f"# len of experiments: {len(args_list)}")

experiments_start_t = datetime.now() + timedelta(minutes=3)
experiments_start_t = datetime.now() - timedelta(minutes=10)
for i, (prev_model_time_len, model_args) in enumerate(zip(model_timedelta_list, args_list)):
    # print({"operate time length of previous model": prev_model_time_len, "model argumets": model_args})
    model_start_t = experiments_start_t if i == 0 else model_start_t + prev_model_time_len
    cron_args = [model_start_t.strftime("%M %H %d %m") + " *"] + list(model_args)
    print("{} /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_main.sh {} {} {} {} {} {} {} {} {} {} {} {} --save_model true".format(*cron_args))
    # if x[9]==1 and x[10]==4:
    #     print(prev_model_time_len, model_args)
