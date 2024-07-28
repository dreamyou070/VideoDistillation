import os
import math
import wandb
import random
import logging
import inspect
import argparse
from diffusers.utils import export_to_gif, export_to_video
from datetime import datetime
import subprocess
from torch.utils.data import RandomSampler
from utils.layer_dictionary import find_layer_name
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from accelerate import Accelerator
import torch
#from dpo import aesthetic_loss_fn, hps_loss_fn
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from models.motion import MotionAdapter
from models.pipelines import AnimateDiffPipeline
from models.scheduler import LCMScheduler
from data.dataset_gen import DistillWebVid10M
from utils.layer_dictionary import find_layer_name
from attn.masactrl_utils import (regiter_attention_editor_diffusers, regiter_motion_attention_editor_diffusers)
from attn.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl

from diffusers.utils import export_to_gif, load_image
import GPUtil
import json
from diffusers.training_utils import EMAModel
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
from diffusers import DDPMScheduler
from utils.diffusion_misc import *
# lcm_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config)
import t2v_metrics
from diffusers.video_processor import VideoProcessor
import numpy as np


def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents

    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

    image = vae.decode(latents).sample
    video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    video = video.float()
    return video


def load_img(img):
    rgb_img = np.array(img, np.float32).squeeze()
    img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()

    img_tensor = (img_tensor / 255. - 0.5) * 2
    return img_tensor


def main(args):

    GPUtil.showUtilization()
    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )

    logger.info(f'\n step 1. wandb start')
    skip_layers, skip_layers_dot = find_layer_name(args.skip_layers)
    if args.use_wandb:
        folder_name = ""
        if len(args.skip_layers) > 0:
            for i in range(len(args.skip_layers)):
                layer_name = args.skip_layers[i]
                if i == len(args.skip_layers) - 1:
                    folder_name += f"{layer_name}"
                else:
                    folder_name += f"{layer_name}_"

    logger.info(f'\n step 2. preparing folder')
    logger.info(f' (2.1) seed')
    torch.manual_seed(args.seed)
    logger.info(f' (2.2) saving dir')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    folder_name = args.sub_folder_name
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_folder = os.path.join(output_dir, "samples")
    os.makedirs(save_folder, exist_ok=True)

    print(f' step 3. wandb logging')
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", beta_schedule=args.beta_schedule, )

    print(f' step 6. pretrained_teacher_model')
    weight_dtype = torch.float32
    untrained_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    untrained_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=untrained_adapter,torch_dtpe=weight_dtype)
    untrained_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    untrained_pipe.set_adapters(["lcm-lora"], [0.8])
    untrained_unet = untrained_pipe.unet

    print(f' step 7. trained model')
    trained_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype).to(device,dtype=weight_dtype)
    trained_adapter_dir = r'/share0/dreamyou070/dreamyou070/VideoDistillation/experiment/2_1_down1_mid_up_02_webvid_distill_loss_1_feature_1_lr_scale_3/checkpoints/'
    trained_adapter_dir = os.path.join(trained_adapter_dir, 'checkpoint_epoch_022.pt')
    trained_adapter_state_dict = torch.load(trained_adapter_dir)
    trained_adapter.load_state_dict(trained_adapter_state_dict)
    trained_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=trained_adapter,torch_dtpe=weight_dtype)
    trained_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",adapter_name="lcm-lora")
    trained_pipe.set_adapters(["lcm-lora"], [0.8])
    trained_unet = trained_pipe.unet

    vae = trained_pipe.vae
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    text_encoder = trained_pipe.text_encoder
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    print(f' step 9. motion control')
    window_size = 16
    guidance_scale = args.guidance_scale
    inference_step = args.inference_step
    student_motion_controller = None
    teacher_motion_controller = None
    if args.motion_control:
        untrained_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=False,
                                                                 do_attention_map_check=True)
        regiter_motion_attention_editor_diffusers(untrained_unet, untrained_motion_controller)
        trained_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=False,
                                                                 do_attention_map_check=True)
        regiter_motion_attention_editor_diffusers(trained_unet, trained_motion_controller)


    print(f' step 10. untrained attnmap check')
    num_frames = 16
    num_inference_steps = 6
    n_prompt = "bad quality, worse quality, low resolution"
    prompt = "Woman is shopping online"
    with torch.no_grad():
        output = untrained_pipe(prompt=prompt,
                                negative_prompt=n_prompt,
                                num_frames=num_frames,
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps,
                                generator=torch.Generator("cpu").manual_seed(args.seed), )
        attention_maps = untrained_motion_controller.attnmap_dict
        untrained_motion_controller.reset()
        for layer_name in attention_maps.keys():
            values = attention_maps[layer_name][-1]
            print(f'[untrained] layer_name = {layer_name}, values = {values.shape}')

        ########### ---------------- ########### ---------------- ########### ---------------- ########### ----------------
        output = trained_pipe(prompt=prompt,
                                negative_prompt=n_prompt,
                                num_frames=num_frames,
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps,
                                generator=torch.Generator("cpu").manual_seed(args.seed), )
        attention_maps = trained_motion_controller.attnmap_dict
        trained_motion_controller.reset()
        for layer_name in attention_maps.keys():
            values = attention_maps[layer_name][-1]
            print(f'[trained] layer_name = {layer_name}, values = {values.shape}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--project', type=str, default='video_distill')
    parser.add_argument('--sub_folder_name', type=str, default='result_sy')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", default='fp16')
    parser.add_argument('--full_attention', action='store_true')
    parser.add_argument('--window_attention', action='store_true')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--motion_control', action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    from utils import arg_as_list

    parser.add_argument('--skip_layers', type=arg_as_list)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument('--vlb_weight', type=float, default=1.0)
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--loss_feature_weight', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    parser.add_argument('--inference_step', type=int, default=6)
    parser.add_argument('--csv_path', type=str, default='data/webvid-10M.csv')
    parser.add_argument('--video_folder', type=str, default='data/webvid-10M')
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--name', type=str, default='video_distill')
    parser.add_argument('--output_dir', type=str, default='experiment')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--teacher_motion_model_dir', type=str, default="wangfuyun/AnimateLCM")
    parser.add_argument('--cfg_random_null_text', action='store_true')
    parser.add_argument('--cfg_random_null_text_ratio', type=float, default=0.1)
    parser.add_argument('--unet_checkpoint_path', type=str, default='')
    parser.add_argument('--unet_additional_kwargs', type=Dict)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--noise_scheduler_kwargs', type=Dict)
    parser.add_argument('--max_train_epoch', type=int, default=-1)
    parser.add_argument('--max_train_steps', type=int, default=-1)
    parser.add_argument('--validation_steps', type=int, default=100)
    parser.add_argument('--validation_steps_tuple', type=Tuple, default=(-1,))
    parser.add_argument('--scale_lr', action='store_true')
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--trainable_modules', type=arg_as_list, default="['motion_modules.']")
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--checkpointing_epochs', type=int, default=5)
    parser.add_argument('--checkpointing_steps', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--mixed_precision_training', action='store_true')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true')
    parser.add_argument('--is_debug', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.", )
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--do_window_attention", action="store_true")
    parser.add_argument("--datavideo_size", type=int, default=512)
    parser.add_argument(
        "--beta_schedule",
        default="scaled_linear",
        type=str,
        help="The schedule to use for the beta values.",
    )
    parser.add_argument("--do_aesthetic_loss", action='store_true', )
    parser.add_argument("--aesthetic_score_weight", type=float, default=0.5)
    parser.add_argument("--do_t2i_loss", action='store_true')
    parser.add_argument("--do_hps_loss", action='store_true')
    parser.add_argument("--clip_flant5_score", action='store_true')
    parser.add_argument("--t2i_score_weight", type=float, default=0.5)
    parser.add_argument("--hps_version", type=str, default="v2.1", help="hps version: 'v2.0', 'v2.1'")
    parser.add_argument("--do_attention_map_check", action='store_true')
    parser.add_argument("--attn_map_weight", type=float, default=0.5)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--up_module_attention", action='store_true')
    parser.add_argument("--down_module_attention", action='store_true')
    args = parser.parse_args()
    name = Path(args.config).stem
    main(args)
