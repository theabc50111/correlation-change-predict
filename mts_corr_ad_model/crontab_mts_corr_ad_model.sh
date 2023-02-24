#!/bin/bash

#EXPORT TZ=Asia/Taipei 

ARGUMENT_LIST=(
  "tr_batch"
)


# read arguments
opts=$(getopt \
  --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
  --name "$(basename "$0")" \
  --options "" \
  -- "$@"
)

eval set --$opts
# The eval in eval set --$opts is required as arguments returned by getopt are quoted.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tr_batch)
      tr_batch="$2" # Note: In order to handle the argument containing space, the quotes around '$2': they are essential!
      shift 2 # The 'shift' eats a commandline argument, i.e. converts $1=a, $2=b, $3=c, $4=d into $1=b, $2=c, $3=d. shift 2 moves it all the way to $1=c, $2=d. It's done since that particular branch uses an argument, so it has to remove two things from the list (the -r and the argument following it) not just one.
      ;;

    *)
      echo 
      break
      ;;
  esac
done


/usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/mts_corr_ad_model.py --tr_batch $tr_batch >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log 2>&1
echo "========================== $(/usr/bin/date) ================================" >> /home/ywt01_dmlab/Documents/codes/correlation-change-predict/mts_corr_ad_model/crontab_mts_corr_ad_model.log
#usr/bin/docker container exec ywt-pytorch python /workspace/correlation-change-predict/mts_corr_ad_model/mts_corr_ad_model.py --tr_batch 48 >> /home/ywt01_dmlab/Documents/tmp/mts_corr_ad_model_crontab.log 2>&1
