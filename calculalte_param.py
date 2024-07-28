import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, load_image
from PIL import Image
import argparse, os
import shutil
import yaml
from utils.layer_dictionary import find_layer_name
from attn.masactrl_utils import (regiter_attention_editor_diffusers,
                                 regiter_motion_attention_editor_diffusers)
from attn.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
import logging
from datetime import datetime
import wandb

def main(args) :

    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
    teacher_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=teacher_adapter, )
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                   adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])
    unet = teacher_pipe.unet
    # [1] getting adapter parameter
    total_param = 0
    for name, param in unet.named_parameters():
        if 'motion' in name.lower():  # and 'motion_module_1' in name.lower():
            total_param += param.numel()
    print(f' original param num= {total_param}')

    # [2] after param
    down_param = 0
    mid_param = 0
    up_param = 0
    up_0_param = 0
    up_2_param = 0
    erasing_num = 0
    for name, param in unet.named_parameters():
        if 'motion' in name :
            if 'down' in name.lower() :
                down_param += param.numel()
            if 'mid' in name.lower() :
                mid_param += param.numel()
                erasing_num += param.numel()
            if 'up' in name.lower() :
                up_param += param.numel()
                if 'up' in name.lower() and 'motion_modules.0.' in name.lower(): # and 'motion_module_1' in name.lower():
                    up_0_param += param.numel()
                    erasing_num += param.numel()

                if 'up' in name.lower() and 'motion_modules.2.' in name.lower():  # and 'motion_module_1' in name.lower():
                    up_2_param += param.numel()
                    erasing_num += param.numel()

    print(f' down related param num = {down_param}')
    print(f' mid related param num = {mid_param}')
    print(f' up related param num = {up_param}')
    print(f' up_0 related param num = {up_0_param}')
    print(f' up_2 related param num = {up_2_param}')
    left_param = total_param - erasing_num
    print(f' left_param num = {left_param} = {left_param/total_param*100:.2f}')
    # how many video data is necessary .. ?
    # [3] calculate lora parameter
    lora_param = 0
    for name, param in unet.named_parameters():
        if 'lora' in name.lower():
            lora_param += param.numel()
    print(f' lora parameter num = {lora_param}')

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--project', type=str, default="video_test")
    parser.add_argument("--save_base_dir", type=str, )
    parser.add_argument('--prompt', type=str,
                        default="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution")
    parser.add_argument('--n_prompt', type=str,
                        default="bad quality, worse quality, low resolution")
    parser.add_argument('--image_dir', type=str, default="__assets__/imgs/space_rocket.jpg")
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--full_attention', action='store_true')
    parser.add_argument('--window_attention', action='store_true')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--motion_control', action='store_true')
    parser.add_argument('--inference_steps', type=int, default=6)
    parser.add_argument('--self_control', action='store_true')
    from utils import arg_as_list
    parser.add_argument('--skip_layers', type=arg_as_list)
    args = parser.parse_args()
    main(args)