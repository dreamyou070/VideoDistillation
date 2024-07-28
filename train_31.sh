#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
# --do_aesthetic_loss --do_attention_map_check
#--pretrained_student_adapter_path "/share0/dreamyou070/dreamyou070/VideoDistillation/experiment/1_1_mid_up_02_webvid_distill_loss_1_feature_1_lr_scale_3/checkpoints/checkpoint_epoch_004.pt" \
#--first_epoch 5 \
CUDA_VISIBLE_DEVICES=0 python train.py \
 --use_wandb \
 --seed 42 \
 --output_dir 'experiment' \
 --teacher_motion_model_dir "wangfuyun/AnimateLCM" \
 --pretrained_model_path "emilianJR/epiCRealism" \
 --sub_folder_name '3_1_same_structure_webvid_distill_loss_1_feature_1_lr_scale_3' \
 --max_train_steps 300000 \
 --config configs/training/v1/training.yaml \
 --sample_n_frames 16 \
 --datavideo_size 256 \
 --inference_step 6 \
 --num_frames 16 \
 --motion_control \
 --guidance_scale 1.5 \
 --skip_layers "[]" \
 --csv_path '../MyData/video/webvid_genvideo/start_csv_file.csv' \
 --video_folder '../MyData/video/webvid_genvideo/sample' \
 --distill_weight 1.0 --vlb_weight 0.0 --loss_feature_weight 1.0 \
 --adam_weight_decay 0.01 --learning_rate 0.0001 \
 --lr_scale 3 \
 --aesthetic_score_weight 0.0 --t2i_score_weight 0.0 --attn_map_weight 0.0