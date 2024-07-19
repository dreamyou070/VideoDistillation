#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
# "guoyww/animatediff-motion-adapter-v1-5-2"
# ['up_0_0','up_1_0','up_2_0','up_3_0',] --csv_path     "../MyData/video/webvid-10M/webvid-10M-csv/0_300.csv" \
# feature matching loss .. ?

CUDA_VISIBLE_DEVICES=1 python train_final_real.py \
 --use_wandb \
 --seed 42 \
 --output_dir 'experiment' \
 --pretrained_model_path "runwayml/stable-diffusion-v1-5" \
 --teacher_motion_adapter_path "guoyww/animatediff-motion-adapter-v1-5-2" \
 --student_motion_adapter_path "wangfuyun/AnimateLCM" \
 --sub_folder_name 'up_0_distill_from_origin_do_aesthetic_loss_dataframe_12' \
 --max_train_steps 300000 \
 --config configs/training/v1/training.yaml \
 --sample_n_frames 10 \
 --datavideo_size 512 \
 --inference_step 6 \
 --num_frames 16 \
 --motion_control \
 --guidance_scale 2.0 \
 --skip_layers "['up_0_0','up_1_0','up_2_0','up_3_0']" \
 --csv_path "../MyData/video/TikTok/tiktok_dataset.csv" \
 --video_folder "../MyData/video/TikTok/TikTok_Video" \
 --distill_weight 1.0 --vlb_weight 0.0 --loss_feature_weight 0.0 \
 --do_aesthetic_loss \
 --adam_weight_decay 0.01 --learning_rate 0.0001