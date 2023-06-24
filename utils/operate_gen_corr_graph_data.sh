#!/bin/bash


# Define the list of items
dataset_list=("PW_WAVE_CONST_DIM_60_BKPS_0_NOISE_STD_2" "PW_LINEAR_DIM_60_BKPS_0_NOISE_STD_2" "PW_WAVE_CONST_DIM_60_BKPS_0_NOISE_STD_50" "PW_LINEAR_DIM_60_BKPS_0_NOISE_STD_50" "PW_LINEAR_DIM_10_BKPS_0_NOISE_STD_2")
train_items_setting="train_all"  # "train_train" "train_all"
filt_gra_mode_list=("keep_abs" "keep_positive" "keep_strong")
filt_gra_quan_list=(0.25 0.5 0.75)
graph_nodes_v_mode_list=("all_values" "mean" "std" "mean_std" "min" "max")
corr_str_list=(1)
corr_win_list=(10 20 50)

# Loop through the list
for corr_win in "${corr_win_list[@]}"
do
    for corr_str in "${corr_str_list[@]}"
    do
        for filt_gra_mode in "${filt_gra_mode_list[@]}"
        do
            for filt_gra_quan in "${filt_gra_quan_list[@]}"
            do
                for graph_nodes_v_mode in "${graph_nodes_v_mode_list[@]}"
                do
                    for dataset in "${dataset_list[@]}"
                    do
                        echo "start generate data with --data_implement $dataset --corr_window $corr_win --corr_stride $corr_str --filt_gra_mode $filt_gra_mode --filt_gra_quan $filt_gra_quan --graph_nodes_v_mode $graph_nodes_v_mode"
                        python ./gen_corr_graph_data.py --data_implement $dataset --corr_window $corr_win --corr_stride $corr_str --filt_gra_mode $filt_gra_mode --filt_gra_quan $filt_gra_quan --graph_nodes_v_mode $graph_nodes_v_mode --save_corr_data --save_corr_graph_arr
                    done
                done
            done
        done
    done
done
