#!/bin/bash

#EXPORT TZ=Asia/Taipei 

ARGUMENT_LIST=(
  "batch_size"
  "tr_epochs"
  "save_model"
  "seq_len"
  "corr_window"
  "corr_stride"
  "filt_mode"
  "filt_quan"
  "graph_nodes_v_mode"
  "drop_pos"
  "drop_p"
  "gra_enc"
  "gra_enc_aggr"
  "gra_enc_l"
  "gra_enc_h"
  "gru_l"
  "gru_h"
)

# read arguments
opts=$(getopt \
  --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
  --name "$(basename "$0")" \
  --options "" \
  -- "$@" 2>> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log)
#  -- "$@")

# if sending invalid option, stop script
if [ $? -ne 0 ]; then
  echo "========================== Error:Invalid option provided to crontab_mts_corr_ad_model.sh at $(/usr/bin/date) ================================" >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log
  exit 1
fi

# The eval in eval set --$opts is required as arguments returned by getopt are quoted.
eval set --$opts

# Default values of arguments
batch_size="--batch_size 32"
tr_epochs="--tr_epochs 1000"
seq_len="--seq_len 10"
corr_window="--corr_window 10"
corr_stride="--corr_stride 1"
filt_mode=""
filt_quan=""
graph_nodes_v_mode=""
drop_pos=()
drop_p=""
gra_enc=""
gra_enc_aggr=""
gra_enc_l=""
gra_enc_h=""
gru_l=""
gru_h=""
save_model=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch_size)
      batch_size="--batch_size $2" # Note: In order to handle the argument containing space, the quotes around '$2': they are essential!
      shift 2 # The 'shift' eats a commandline argument, i.e. converts $1=a, $2=b, $3=c, $4=d into $1=b, $2=c, $3=d. shift 2 moves it all the way to $1=c, $2=d. It's done since that particular branch uses an argument, so it has to remove two things from the list (the -r and the argument following it) not just one.
      ;;

    --tr_epochs)
      tr_epochs="--tr_epochs $2"
      shift 2
      ;;

    --save_model)
      save_model="--save_model"
      shift 2
      ;;

    --corr_window)
      corr_window="--corr_window $2"
      shift 2
      ;;

    --corr_stride)
      corr_stride="--corr_stride $2"
      shift 2
      ;;

    --filt_mode)
      filt_mode="--filt_mode $2"
      shift 2
      ;;

    --filt_quan)
      filt_quan="--filt_quan $2"
      shift 2
      ;;

    --graph_nodes_v_mode)
      filt_quan="--graph_nodes_v_mode $2"
      shift 2
      ;;

    --drop_pos)
      drop_pos+=("$2")
      shift 2
      ;;

    --drop_p)
      drop_p="--drop_p $2"
      shift 2
      ;;

    --gra_enc)
      gra_enc="--gra_enc $2"
      shift 2
      ;;

    --gra_enc_aggr)
      gra_enc="--gra_enc_aggr $2"
      shift 2
      ;;

    --gra_enc_l)
        gra_enc_l="--gra_enc_l $2"
      shift 2
      ;;

    --gra_enc_h)
      gra_enc_h="--gra_enc_h $2"
      shift 2
      ;;

    --gru_l)
      gru_l="--gru_l $2"
      shift 2
      ;;

    --gru_h)
      gru_h="--gru_h $2"
      shift 2
      ;;

    --)
      # if getopt reached the end of options, exit loop
      shift
      break
      ;;

    *)
      # if sending invalid option, stop script
      echo "========================== Error:Unrecognized option: $1 provided to crontab_mts_corr_ad_model.sh at $(/usr/bin/date) ================================" >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log
      exit 1
      ;;

  esac
done

echo "========================== Start training at $(/usr/bin/date) ==========================" >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log

/usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/mts_corr_ad_model.py $batch_size $tr_epochs $seq_len $corr_window $corr_stride $filt_mode $filt_quan $graph_nodes_v_mode ${drop_pos[@]} $drop_p $gra_enc $gra_enc_aggr $gra_enc_l $gra_enc_h $gru_l $gru_h $save_model >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log 2>&1

#if [ -n "$save_model" ];
#then
#    /usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/mts_corr_ad_model.py --tr_batch $tr_batch --val_batch $val_batch --test_batch $test_batch --tr_epochs $tr_epochs --corr_window $corr_window --corr_stride $corr_stride --filt_mode $filt_mode --filt_quan $filt_quan $discr_loss --discr_loss_r $discr_loss_r --discr_pred_disp_r $discr_pred_disp_r --drop_pos ${drop_pos[@]} --drop_p $drop_p --gra_enc $gra_enc --gra_enc_l $gra_enc_l --gra_enc_h $gra_enc_h --gru_l $gru_l --gru_h $gru_h --save_model >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log 2>&1
#else
#    /usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/mts_corr_ad_model.py --tr_batch $tr_batch --val_batch $val_batch --test_batch $test_batch --tr_epochs $tr_epochs --corr_window $corr_window --corr_stride $corr_stride --filt_mode $filt_mode --filt_quan $filt_quan $discr_loss --gra_enc $gra_enc --gra_enc_l $gra_enc_l --gra_enc_h $gra_enc_h --gru_l $gru_l --gru_h $gru_h >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log 2>&1
#fi

echo "========================== End training at $(/usr/bin/date) ================================" >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log
