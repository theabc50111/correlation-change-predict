#!/bin/bash


ARGUMENT_LIST=(
  "discr_loss_r"
  "discr_pred_disp_r"
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
discr_loss_r=3
discr_pred_disp_r=7


while [[ $# -gt 0 ]]; do
  case "$1" in

    --discr_loss_r)
      discr_loss_r="$2"
      shift 2
      ;;

    --discr_pred_disp_r)
      discr_pred_disp_r="$2"
      shift 2
      ;;

    --)
      # if getopt reached the end of options, exit loop
      shift
      break
      ;;

    *)
      # if sending invalid option, stop script
      echo "========================== Error:Invalid option: $1 provided to crontab_mts_corr_ad_model.sh at $(/usr/bin/date) ================================" >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log
      exit 1
      ;;

  esac
done

echo "--discr_loss_r $discr_loss_r --discr_pred_disp_r $discr_pred_disp_r"
