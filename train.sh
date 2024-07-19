#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

python train.py \
 --use_wandb \
 --seed 42 \
 --output_dir 'experiment' \
 --teacher_motion_model_dir "wangfuyun/AnimateLCM" \
 --pretrained_model_path "emilianJR/epiCRealism" \
 --sub_folder_name 'experiment_up_0_distill_from_teacher_distill_dataframe_8' \
 --max_train_steps 300000 \
 --config configs/training/v1/training.yaml \
 --sample_n_frames 8 \
 --datavideo_size 512 \
 --inference_step 6 \
 --num_frames 16 \
 --motion_control \
 --guidance_scale 2.0 \
 --skip_layers " ['up_0_0','up_1_0','up_2_0','up_3_0',]" \
 --csv_path "../MyData/video/TikTok/tiktok_dataset.csv" \
 --video_folder "../MyData/video/TikTok/TikTok_Video" \
 --distill_weight 1.0 --vlb_weight 0.0 --loss_feature_weight 1.0 \
 --adam_weight_decay 0.01 --learning_rate 0.0001



# layer_dict_short = {0: 'down_0_0', 1: 'down_0_1',
#                      2: 'down_1_0', 3: 'down_1_1',
#                    4: 'down_2_0', 5: 'down_2_1',
#                    6: 'down_3_0', 7: 'down_3_1',
#                    8: 'mid',
#                    9: 'up_0_0', 10: 'up_0_1', 11: 'up_0_2',
#                    12: 'up_1_0', 13: 'up_1_1', 14: 'up_1_2',
#                    15: 'up_2_0', 16: 'up_2_1', 17: 'up_2_2',
#                    18: 'up_3_0', 19: 'up_3_1', 20: 'up_3_2',}
