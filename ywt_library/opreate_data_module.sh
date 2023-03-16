#!/bin/bash


# Define the list of items
filt_gra_mode_list=("keep_abs" "keep_positive" "keep_strong")
filt_gra_quan_list=(0.25 0.5 0.75)
corr_str_list=(1 5 10 50)
corr_win_list=(5 10 50)

# Loop through the list
for corr_win in "${corr_win_list[@]}"
do
    for corr_str in "${corr_str_list[@]}"
    do
        for filt_gra_mode in "${filt_gra_mode_list[@]}"
        do
            for filt_gra_quan in "${filt_gra_quan_list[@]}"
            do
                echo "start generate data with --corr_window $corr_win --corr_stride $corr_str --filt_gra_mode $filt_gra_mode --filt_gra_quan $filt_gra_quan"
                python ./data_module.py --corr_window $corr_win --corr_stride $corr_str --filt_gra_mode $filt_gra_mode --filt_gra_quan $filt_gra_quan --save_corr_graph_arr
            done
        done
    done
done
