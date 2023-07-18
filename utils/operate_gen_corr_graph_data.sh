#!/bin/bash


# Define the list of items


dataset_list=("--data_implement MERGE_CLUSTER_2_NIKE_TS_DIM_30")


train_items_setting="train_all"  # "train_train" "train_all"
filt_gra_mode_list=("")  # ("--filt_gra_mode keep_abs" "--filt_gra_mode keep_positive" "--filt_gra_mode keep_strong")
filt_gra_quan_list=("") # ("--data_implement 0.25" "--data_implement 0.5" "--data_implement 0.75")
graph_nodes_v_mode_list=("--graph_nodes_v_mode all_values" "--graph_nodes_v_mode mean" "--graph_nodes_v_mode std" "--graph_nodes_v_mode mean_std")
corr_str_list=("--corr_stride 1")
corr_win_list=("--corr_window 10" "--corr_window 30" "--corr_window 50")  # ("--corr_window 10" "--corr_window 30" "--corr_window 50")
corr_type=("--corr_type pearson")  # ("--corr_type pearson" "--corr_type cross_corr")

# Loop through the list
for corr_type in "${corr_type[@]}"
do
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
                            echo "start generate data with $dataset $corr_type $corr_win $corr_str $filt_gra_mode $filt_gra_quan $graph_nodes_v_mode"
                            python ./gen_corr_graph_data.py $dataset $corr_type $corr_win $corr_str $filt_gra_mode $filt_gra_quan $graph_nodes_v_mode --save_corr_data --save_corr_graph_arr
                        done
                    done
                done
            done
        done
    done
done
