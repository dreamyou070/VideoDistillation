#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
# 'up_0_0','up_1_0','up_2_0','up_3_0',
#

CUDA_VISIBLE_DEVICES=0 python ../VideoDistillation/train.py \
 --use_wandb \
 --seed 42 \
 --output_dir 'experiment' \
 --teacher_motion_model_dir "wangfuyun/AnimateLCM" \
 --pretrained_model_path "emilianJR/epiCRealism" \
 --sub_folder_name 'mid_up_0_up_2_distill_from_teacher_distill_dataframe_14_tiktok_distill_loss_1_feature_loss_1_aesthetic_loss_1.0_hps_loss_1.0do_attention_map_loss_1.0' \
 --max_train_steps 300000 \
 --config configs/training/v1/training.yaml \
 --sample_n_frames 14 \
 --datavideo_size 512 \
 --inference_step 6 \
 --num_frames 16 \
 --motion_control \
 --guidance_scale 2.0 \
 --skip_layers " ['up_0_0','up_1_0','up_2_0','up_3_0','up_0_2','up_1_2','up_2_2','up_3_2','mid',]" \
 --csv_path "../MyData/video/TikTok/tiktok_dataset.csv" \
 --video_folder "../MyData/video/TikTok/TikTok_Video" \
 --do_aesthetic_loss --aesthetic_score_weight 1.0 \
 --do_hps_loss --hps_score_weight 1.0 \
 --do_attention_map_check --attn_map_weight 1.0 \
 --distill_weight 1.0 --vlb_weight 0.0 --loss_feature_weight 1.0 \
 --adam_weight_decay 0.01 --learning_rate 0.0001