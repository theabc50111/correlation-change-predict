#!/bin/bash

#EXPORT TZ=Asia/Taipei

ARGUMENT_LIST=(
  "log_suffix"
  "data_implement"
  "batch_size"
  "tr_epochs"
  "train_models"
  "save_model"
  "seq_len"
  "corr_type"
  "corr_window"
  "corr_stride"
  "filt_mode"
  "filt_quan"
  "quan_discrete_bins"
  "custom_discrete_bins"
  "graph_nodes_v_mode"
  "target_mats_path"
  "cuda_device"
  "weight_decay"
  "graph_enc_weight_l2_reg_lambda"
  "drop_pos"
  "drop_p"
  "gra_enc"
  "gra_enc_aggr"
  "gra_enc_l"
  "gra_enc_h"
  "gru_l"
  "gru_h"
  "edge_acc_loss_atol"
  "output_type"
  "output_bins"
)

# Default empty values of arguments
sh_script_err_log_file="$HOME/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_main_sh_err.log"
data_implement=""
log_suffix=""
batch_size=""
tr_epochs=""
seq_len=""
corr_type=""
corr_window=""
corr_stride=""
filt_mode=""
filt_quan=""
quan_discrete_bins=""
custom_discrete_bins=""
graph_nodes_v_mode=""
target_mats_path=""
cuda_device=""
weight_decay=""
graph_enc_weight_l2_reg_lambda=""
drop_pos=()
drop_p=""
gra_enc=""
gra_enc_aggr=""
gra_enc_l=""
gra_enc_h=""
gru_l=""
gru_h=""
edge_acc_loss_atol=""
output_type=""
output_bins=""
save_model=""

# read arguments
opts=$(getopt \
  --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
  --name "$(basename "$0")" \
  --options "" \
  -- "$@" 2>> $sh_script_err_log_file)
#  -- "$@")

# if sending invalid option, stop script
if [ $? -ne 0 ]; then
  echo "========================== Error:Invalid option provided to crontab_main.sh at $(/usr/bin/date) ================================" >> $sh_script_err_log_file
  exit 1
fi

# The eval in eval set --$opts is required as arguments returned by getopt are quoted.
eval set --$opts


while [[ $# -gt 0 ]]; do
  case "$1" in
    --log_suffix)
      log_suffix="$2" # Note: In order to handle the argument containing space, the quotes around '$2': they are essential!
      log_file="$HOME/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_main_${log_suffix}.log"
      shift 2 # The 'shift' eats a commandline argument, i.e. converts $1=a, $2=b, $3=c, $4=d into $1=b, $2=c, $3=d. shift 2 moves it all the way to $1=c, $2=d. It's done since that particular branch uses an argument, so it has to remove two things from the list (the -r and the argument following it) not just one.
      ;;

    --data_implement)
      data_implement="--data_implement $2"
      shift 2
      ;;

    --batch_size)
      batch_size="--batch_size $2"
      shift 2
      ;;

    --tr_epochs)
      tr_epochs="--tr_epochs $2"
      shift 2
      ;;

    --train_models)
      train_models_args+=("$2")
      train_models="--train_models ${train_models_args[@]}"
      shift 2
      ;;

    --save_model)
      save_model="--save_model"
      shift 2
      ;;

    --seq_len)
      seq_len="--seq_len $2"
      shift 2
      ;;

    --corr_type)
      corr_type="--corr_type $2"
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

    --quan_discrete_bins)
      quan_discrete_bins="--quan_discrete_bins $2"
      shift 2
      ;;

    --custom_discrete_bins)
      custom_discrete_bins="--custom_discrete_bins $2"
      shift 2
      ;;

    --graph_nodes_v_mode)
      filt_quan="--graph_nodes_v_mode $2"
      shift 2
      ;;

    --target_mats_path)
      target_mats_path="--target_mats_path $2"
      shift 2
      ;;

    --cuda_device)
      cuda_device="--cuda_device $2"
      shift 2
      ;;

    --weight_decay)
      weight_decay="--weight_decay $2"
      shift 2
      ;;

    --graph_enc_weight_l2_reg_lambda)
      graph_enc_weight_l2_reg_lambda="--graph_enc_weight_l2_reg_lambda $2"
      shift 2
      ;;

    --drop_pos)
      drop_pos_args+=("$2")
      drop_pos="--drop_pos ${drop_pos_args[@]}"
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

    --edge_acc_loss_atol)
      edge_acc_loss_atol="--edge_acc_loss_atol $2"
      shift 2
      ;;

    --output_type)
      output_type="--output_type $2"
      shift 2
      ;;

    --output_bins)
      output_bins_args+=("$2")
      output_bins="--output_bins ${output_bins_args[@]}"
      shift 2
      ;;

    --)
      # if getopt reached the end of options, exit loop
      shift
      break
      ;;

    *)
      # if sending invalid option, stop script
      echo "========================== Error:Unrecognized option: $1 provided to crontab_main.sh at $(/usr/bin/date) ================================" >> $log_file
      exit 1
      ;;

  esac
done

echo "========================== Start training at $(/usr/bin/date) ==========================" >> $log_file

/usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/main.py $data_implement $batch_size $tr_epochs $train_models $seq_len $corr_type $corr_window $corr_stride $filt_mode $filt_quan $quan_discrete_bins $custom_discrete_bins $graph_nodes_v_mode $target_mats_path $cuda_device $weight_decay $graph_enc_weight_l2_reg_lambda ${drop_pos[@]} $drop_p $gra_enc $gra_enc_aggr $gra_enc_l $gra_enc_h $gru_l $gru_h $edge_acc_loss_atol $output_type $output_bins $save_model >> "$log_file" 2>&1

echo "========================== End training at $(/usr/bin/date) ================================" >> $log_file
