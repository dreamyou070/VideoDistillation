#python train_distill.py --config configs/training/v1/training.yaml \
# weight decay
# layer_dict_short = {0: 'down_0_0', 1: 'down_0_1',
 #                    2: 'down_1_0', 3: 'down_1_1',
 #                    4: 'down_2_0', 5: 'down_2_1',
 #                    6: 'down_3_0', 7: 'down_3_1',
 #                    8: 'mid',
 #                    9: 'up_0_0', 10: 'up_0_1', 11: 'up_0_2',
 #                    12: 'up_1_0', 13: 'up_1_1', 14: 'up_1_2',
 #                    15: 'up_2_0', 16: 'up_2_1', 17: 'up_2_2',
 #                    18: 'up_3_0', 19: 'up_3_1', 20: 'up_3_2',}
# If I use down_0_0, can it be ... ?
# #--motion_control \
#--csv_path "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-csv/0_300.csv" \
 #--video_folder "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-partial-video" \
# experiment_5_16_mid_up_02_origin_data_1000_distill_loss_1_feature_1_with_image_feature_loss

port_number=50331
accelerate launch --config_file ../gpu_config/gpu_0_config \
 --main_process_port $port_number \
 test_distill.py \
 --inference_step 6 \
 --guidance_scale 1.5 --motion_control \
 --skip_layers "['up_0_0','up_1_0','up_2_0','up_3_0','up_0_2','up_1_2','up_2_2','up_3_2','mid']" \
 --output_dir '../VideoDistillation_Script/experiment/5_16_mid_up_02_origin_data_1000_distill_loss_1_feature_1_with_image_feature_loss' \
 --saved_epoch 4