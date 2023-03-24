from itertools import product, chain, repeat
from datetime import datetime, timedelta


filt_mode_list = ["", "--filt_mode keep_strong"]  # [None, "keep_strong", "keep_positive", "keep_abs"]
filt_quan_list = [0.5]  # [0.25, 0.5, 0.75]
discr_loss_list = ["", "--discr_loss true"]
discr_loss_r_list = [0.01]  # [0.1, 0.01, 0.001]
discr_pred_disp_r_list = [2] # [1, 2, 5]
drop_pos_list = ["", "--drop_pos fc"]  # ["--drop_pos gru", "--drop_pos fc --drop_pos gru"]
drop_p_list = [0.5]  # [0.33, 0.66]
gra_enc_list = ["gin", "gine"]  # ["gin", "gine"]
gra_enc_l_list = [1, 5]
gra_enc_h_list = [4, 16, 64]

args_list = list(product(filt_mode_list, filt_quan_list, discr_loss_list, discr_loss_r_list, discr_pred_disp_r_list, drop_pos_list, drop_p_list, gra_enc_list,  gra_enc_l_list, gra_enc_h_list))
args_list = list(filter(lambda x:not (x[8]==5 and x[9]==64), args_list))
args_list = list(filter(lambda x:not (x[8]==1 and x[9]==16), args_list))
args_list = list(filter(lambda x:not (x[0]==None and x[1]==0.25), args_list))
args_list = list(filter(lambda x:not (x[0]==None and x[1]==0.75), args_list))
args_list = list(filter(lambda x:not (x[2]==None and x[3]==.1), args_list))
args_list = list(filter(lambda x:not (x[2]==None and x[3]==0.001), args_list))
args_list = list(filter(lambda x:not (x[2]==None and x[4]==1), args_list))
args_list = list(filter(lambda x:not (x[2]==None and x[4]==5), args_list))
args_list = list(filter(lambda x:not (x[5]==None and x[6]==0.33), args_list))
args_list = list(filter(lambda x:not (x[5]==None and x[6]==0.66), args_list))
args_list = sorted(args_list, key=lambda x:x[9])
args_list = sorted(args_list, key=lambda x:x[8])
args_list = sorted(args_list, key=lambda x:x[2])

num_models = sum([1 for x in args_list if x[2]=="--discr_loss true" and  x[8]==5 and x[9]==16])  # the main reasons for model operation time: discr_loss, gra_enc_l, gra_enc_h
model_timedelta_list = [timedelta(minutes=20), timedelta(hours=2), timedelta(hours=2), timedelta(hours=2, minutes=40),  # The order of elements of model_timedelta_list should comply with the order of comply with args_list
                        timedelta(hours=2, minutes=30), timedelta(hours=3, minutes=35), timedelta(hours=7, minutes=30), timedelta(hours=8)]
model_timedelta_list = list(chain.from_iterable(repeat(x, num_models) for x in model_timedelta_list))
assert len(args_list) == len(model_timedelta_list), "The order of elements of model_timedelta_list should comply with the order of comply with args_list"
print(f"# len of experiments: {len(args_list)}")

#start_t = datetime.now() + timedelta(minutes = 10)
start_t = datetime.now() - timedelta(minutes = 55)
for i, (t_delta, model_args) in enumerate(zip(model_timedelta_list, args_list)):
    # print(t_delta, model_args)
    t = start_t if i==0 else t+t_delta
    args = [t.strftime("%M %H %d %m") + " *"] + list(model_args)
    print("{} /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.sh {} --filt_quan {} {} --discr_loss_r {} --discr_pred_disp_r {} {} --drop_p {} --gra_enc {} --gra_enc_l {} --gra_enc_h {} --save_model true".format(*args))
    # if x[8]==1 and x[9]==4:
    #     print(t_delta, model_args)
