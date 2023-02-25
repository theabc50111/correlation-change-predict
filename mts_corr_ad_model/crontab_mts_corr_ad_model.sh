#!/bin/bash

#EXPORT TZ=Asia/Taipei 

ARGUMENT_LIST=(
  "tr_batch"
  "tr_epochs"
  "save_model"
  "corr_window"
  "corr_stride"
)


# read arguments
opts=$(getopt \
  --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
  --name "$(basename "$0")" \
  --options "" \
  -- "$@"
)

# if sending invalid option, stop script
if [ $? -ne 0 ]; then
  echo "Invalid option provided"
  exit 1
fi

eval set --$opts
# The eval in eval set --$opts is required as arguments returned by getopt are quoted.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tr_batch)
      tr_batch="$2" # Note: In order to handle the argument containing space, the quotes around '$2': they are essential!
      shift 2 # The 'shift' eats a commandline argument, i.e. converts $1=a, $2=b, $3=c, $4=d into $1=b, $2=c, $3=d. shift 2 moves it all the way to $1=c, $2=d. It's done since that particular branch uses an argument, so it has to remove two things from the list (the -r and the argument following it) not just one.
      ;;

    --tr_epochs)
      tr_epochs="$2"
      shift 2
      ;;

    --save_model)
      save_model="$2"
      shift 2
      ;;

    --corr_window)
      corr_window="$2"
      shift 2
      ;;

    --corr_stride)
      corr_stride="$2"
      shift 2
      ;;

    *)
      echo 
      break
      ;;
  esac
done

if [ -n "$save_model" ];
then
    /usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/mts_corr_ad_model.py --tr_batch $tr_batch --tr_epochs $tr_epochs --corr_window $corr_window --corr_stride $corr_stride --save_model >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log 2>&1
else
    /usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/mts_corr_ad_model.py --tr_batch $tr_batch  --tr_epochs $tr_epochs --corr_window $corr_window --corr_stride $corr_stride >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log 2>&1
fi

echo "========================== $(/usr/bin/date) ================================" >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log
