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
from dpo import aesthetic_loss_fn, hps_loss_fn
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
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
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
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    log_folder = os.path.join(output_dir, "logs")
    os.makedirs(log_folder, exist_ok=True)

    print(f' step 3. wandb logging')
    wandb.init(project=args.project,
               entity='dreamyou070',
               mode='online',
               name=f'experiment_{args.sub_folder_name}',
               dir=log_folder)
    weight_dtype = torch.float32

    logger.info(f' step 4. noise scheduler')
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",
                                                    beta_schedule=args.beta_schedule, )

    print(f' step 5. ODE Solver (erasing noise)')
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(device, dtype=weight_dtype)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(device, dtype=weight_dtype)

    print(f' step 6. pretrained_teacher_model')
    teacher_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=teacher_adapter,
                                                       torch_dtpe=weight_dtype)
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                   adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])

    print(f' step 7. teacher model')
    vae = teacher_pipe.vae
    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder
    teacher_unet = teacher_pipe.unet

    print(f' step 8. student model')
    student_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype).to(device,
                                                                                                               dtype=weight_dtype)
    student_adapter_config = student_adapter.config
    pretrained_state_dict = student_adapter.state_dict()
    if args.random_init:  # make another module
        print(f' student adapter random initialization')
        student_adapter = MotionAdapter(**student_adapter_config)
        random_state_dict = student_adapter.state_dict()
        for key in random_state_dict.keys():
            raw_value = random_state_dict[key]
            pretrained_value = pretrained_state_dict[key]
            equal_check = torch.equal(raw_value, pretrained_value)
            print(f' [randomize] {key} equal_check with pretrained : {equal_check}')
    student_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=student_adapter,
                                                       torch_dtpe=weight_dtype)
    student_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                   adapter_name="lcm-lora")
    student_pipe.set_adapters(["lcm-lora"], [0.8])
    student_unet = student_pipe.unet

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    student_unet.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    teacher_unet.to(device, dtype=weight_dtype)
    student_unet.to(device, dtype=weight_dtype)  # this cannot be ?
    # make scheduler

    print(f' step 9. motion control')
    window_size = 16
    guidance_scale = args.guidance_scale
    inference_step = args.inference_step
    student_motion_controller = None
    teacher_motion_controller = None
    if args.motion_control:
        teacher_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=False,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=True,
                                                                 do_attention_map_check=args.do_attention_map_check)  # 32
        regiter_motion_attention_editor_diffusers(teacher_unet, teacher_motion_controller)
        student_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=False,
                                                                 do_attention_map_check=args.do_attention_map_check)
        regiter_motion_attention_editor_diffusers(student_unet, student_motion_controller)

    if args.gradient_checkpointing:
        # gradient checkpointing reduce momoery consume
        student_unet.enable_gradient_checkpointing()

    print(f' step 9. Scale lr')
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.per_gpu_batch_size)

    #################################################################################################################
    print(f' step 10. Initialize the optimizer')
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    parameters_list = []
    student_unet.requires_grad_(True)
    for name, para in student_unet.named_parameters():
        # update all unet parameter
        if 'motion' in name:
            para.requires_grad = True
            if skip_layers_dot is not None:
                for skip_layer in skip_layers_dot:
                    if skip_layer in name:
                        para.requires_grad = False
                        break
        else:
            para.requires_grad = False
    for name, para in student_unet.named_parameters():
        if para.requires_grad:
            parameters_list.append(para)
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon, )

    print(f' step 11. recording parameters')
    rec_txt1 = open('recording_param_untraining.txt', 'w')
    rec_txt2 = open('recording_param_training.txt', 'w')
    for name, para in student_unet.named_parameters():
        if para.requires_grad is False:
            rec_txt1.write(f'{name}\n')
        else:
            rec_txt2.write(f'{name}\n')
    rec_txt1.close()
    rec_txt2.close()

    if args.gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()

    print(f' step 12. data loader')
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=args.datavideo_size,
                                     sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=True)

    print(f' step 13. training steps')
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    print(f' step 14. learning rate scheduler')
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps)

    unet_config = student_unet.config
    # save config
    config_dir = 'unet_models'
    os.makedirs(config_dir, exist_ok=True)
    # save config
    student_unet.save_config(config_dir)
    # unet_config.save_pretrained(config_dir)

    print(f' step 15. prepare model with our `accelerator')
    student_unet.to(device, dtype=weight_dtype)
    aesthetic_loss_fnc = aesthetic_loss_fn(grad_scale=0.1,
                                           aesthetic_target=10,
                                           torch_dtype=weight_dtype,
                                           device=device)
    if args.do_t2i_loss:
        print(f' dp t2i loss ! ')
        if args.do_hps_loss:
            t2i_loss_fnc = hps_loss_fn(weight_dtype,
                                       device,
                                       hps_version=args.hps_version)
        elif args.clip_flant5_score:
            t2i_loss_fnc = t2v_metrics.VQAScore(model='clip-flant5-xxl')  # our recommended scoring model

    print(f' step 16. training num')
    # train_dataloader = 300, gradient_accumulation_steps = 1
    # num_update_steps_per_epoch = 300
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print(f'args.max_train_steps : {args.max_train_steps}')
    print(f'num_update_steps_per_epoch : {num_update_steps_per_epoch}')
    print(f'args.num_train_epochs : {args.num_train_epochs}')

    print(f' step 17. Train!')
    total_batch_size = args.per_gpu_batch_size * args.gradient_accumulation_steps  # 300
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    ##########################################################################################
    print(f' [inference condition] ')
    guidance_scale = 1.5

    ##########################################################################################
    # Only show the progress bar once on each machine.
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision_training else None
    progress_bar = tqdm(range(global_step, args.max_train_steps), desc="Steps")

    unet_dict = {}
    for epoch in range(first_epoch, args.num_train_epochs):
        teacher_unet.train()
        student_unet.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):

            if step == 0:
                print(f' [epoch {epoch}] evaluation')
                validation_prompts = [
                    # "A person dances outdoors, shifting from one leg extended and arms outstretched to an upright stance with arms at shoulder height. They then alternate between poses: one leg bent, the other extended, and arms in various positions."
                    "A video of a woman, having a selfie",
                    # "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
                    # "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    # "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                    # "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
                    # "Cute small corgi sitting in a movie theater eating popcorn, unreal engine.",
                    # "A Pikachu with an angry expression and red eyes, with lightning around it, hyper realistic style.",
                    # "A dog is reading a thick book.",
                    # "Three cats having dinner at a table at new years eve, cinematic shot, 8k.",
                    # "An astronaut riding a pig, highly realistic dslr photo, cinematic shot.",
                ]
                with torch.no_grad():

                    ###########################################################################################################
                    # --------------- # --------------- # --------------- # --------------- # --------------- # ---------------
                    # [1] motion adapter
                    if epoch != 0:
                        eval_adapter = MotionAdapter.from_config(student_adapter_config)
                        eval_adapter_state_dict = {}
                        for key in student_unet.state_dict().keys():
                            if 'motion' in key:
                                eval_adapter_state_dict[key] = student_unet.state_dict()[key]
                        eval_adapter.load_state_dict(eval_adapter_state_dict)
                    else:
                        eval_adapter = teacher_adapter
                    # --------------- # --------------- # --------------- # --------------- # --------------- # ---------------
                    ############################################################################################################

                    # [2] basic unet
                    evaluation_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                                          motion_adapter=eval_adapter,
                                                                          torch_dtype=torch.float16)
                    # [3] scheduler
                    evaluation_pipe.scheduler = LCMScheduler.from_config(evaluation_pipe.scheduler.config,
                                                                         beta_schedule="linear")

                    # [4]
                    evaluation_pipe.load_lora_weights("wangfuyun/AnimateLCM",
                                                      weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                                      adapter_name="lcm-lora")
                    evaluation_pipe.set_adapters(["lcm-lora"], [0.8])

                    # [5]
                    eval_unet = evaluation_pipe.unet
                    if step != 0:
                        eval_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                              frame_num=16,
                                                                              full_attention=args.full_attention,
                                                                              window_attention=args.window_attention,
                                                                              window_size=window_size,
                                                                              total_frame_num=args.num_frames,
                                                                              skip_layers=skip_layers,
                                                                              is_teacher=False, )
                    else:
                        eval_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                              frame_num=16,
                                                                              full_attention=args.full_attention,
                                                                              window_attention=False,
                                                                              window_size=window_size,
                                                                              total_frame_num=args.num_frames,
                                                                              skip_layers=skip_layers,
                                                                              is_teacher=True,
                                                                              do_attention_map_check=args.do_attention_map_check)
                    regiter_motion_attention_editor_diffusers(eval_unet, eval_motion_controller)
                    evaluation_pipe.unet = eval_unet

                    # [4] lcm lora
                    evaluation_pipe.enable_vae_slicing()
                    evaluation_pipe.to('cuda')

                    num_frames = args.num_frames
                    num_inference_steps = args.inference_step
                    n_prompt = "bad quality, worse quality, low resolution"
                    for p, prompt in enumerate(validation_prompts):
                        save_p = str(p).zfill(3)
                        output = evaluation_pipe(prompt=prompt,
                                                 negative_prompt=n_prompt,
                                                 num_frames=num_frames,
                                                 guidance_scale=guidance_scale,
                                                 num_inference_steps=num_inference_steps,
                                                 generator=torch.Generator("cpu").manual_seed(args.seed), )
                        student_motion_controller.reset()
                        frames = output.frames[0]
                        export_to_gif(frames,
                                      os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}_{save_p}.gif'))
                        export_to_video(frames,
                                        os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}_{save_p}.mp4'))
                        text_dir = os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}_{save_p}.txt')
                        with open(text_dir, 'w') as f:
                            f.write(f'prompt : {prompt}\n')
                            f.write(f'n_prompt : {n_prompt}\n')
                            f.write(f'guidance_scale : {guidance_scale}\n')
                            f.write(f'num_inference_steps : {num_inference_steps}\n')
                            f.write(f'seed : {args.seed}\n')
                        fps = 10
                        wandb.log({"video": wandb.Video(
                            data_or_path=os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}_{save_p}.gif'),
                            caption=f'epoch_{str(epoch).zfill(3)}_{save_p}', fps=fps)})

                    print(f' [epoch {epoch}] saving model')
                    # [1] State Saving
                    trained_value = student_unet.state_dict()
                    save_state_dict = {}
                    for trained_key, trained_value in trained_value.items():
                        if 'motion' in trained_key:
                            save_state_dict[trained_key] = trained_value.to('cpu')
                    save_epoch = str(epoch).zfill(3)
                    torch.save(save_state_dict,
                               os.path.join(output_dir, f"checkpoints/checkpoint_epoch_{save_epoch}.pt"))
                    del evaluation_pipe, eval_unet
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            # ------------------------------------------------------------------------------------------------------------
            # [1]
            pixel_values = batch["pixel_values"]
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                with torch.no_grad():
                    latents = vae.encode(pixel_values.to(device, dtype=weight_dtype)).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215
            noise = torch.randn_like(latents).to(device, dtype=weight_dtype)
            bsz = latents.shape[0]

            # [2]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()  # torch
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            with torch.no_grad():
                prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding="max_length",
                                       truncation=True, return_tensors="pt").input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0].to(device, dtype=weight_dtype)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.no_grad():
                teacher_model_pred = teacher_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if args.motion_control:
                    t_hdict = teacher_motion_controller.layerwise_hidden_dict  #
                    t_attn_dict = teacher_motion_controller.attnmap_dict
                    teacher_motion_controller.reset()
                    teacher_motion_controller.layerwise_hidden_dict = {}
                    teacher_motion_controller.attnmap_dict = {}

            ########################################################################################################
            # Here Problem (student, problem)
            student_model_pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            if args.motion_control:
                s_hdict = student_motion_controller.layerwise_hidden_dict
                s_attn_dict = student_motion_controller.attnmap_dict
                student_motion_controller.reset()
                student_motion_controller.layerwise_hidden_dict = {}
                student_motion_controller.attnmap_dict = {}
                loss_feature = 0
                for layer_name in s_hdict.keys():
                    s_h = s_hdict[layer_name]
                    t_h = t_hdict[layer_name]
                    for s_h_, t_h_ in zip(s_h, t_h):
                        loss_feature += F.mse_loss(s_h_.float(), t_h_.float(), reduction="mean")

                if args.do_attention_map_check:
                    loss_attn_map = 0
                    for layer_name in s_attn_dict.keys():
                        s_attn = s_attn_dict[layer_name]
                        t_attn = t_attn_dict[layer_name]
                        for s_attn_, t_attn_ in zip(s_attn, t_attn):
                            loss_attn_map += F.mse_loss(s_attn_.float(), t_attn_.float(), reduction="mean")

            #########################################################################################################
            # [1] Teacher Distillation
            loss_vlb = F.mse_loss(student_model_pred.float(), target.float(), reduction="mean")
            loss_distill = F.mse_loss(student_model_pred.float(), teacher_model_pred.float(), reduction="mean")
            total_loss = args.vlb_weight * loss_vlb + args.distill_weight * loss_distill
            if args.motion_control:
                total_loss = args.vlb_weight * loss_vlb + args.distill_weight * loss_distill + args.loss_feature_weight * loss_feature
                if args.do_attention_map_check:
                    total_loss += args.attn_map_weight * loss_attn_map
            ########################################################################################################

            # [3] aesthetic
            if args.do_aesthetic_loss or args.do_t2i_loss:
                pred_x_0_stu = get_predicted_original_sample(student_model_pred,
                                                             timesteps, noisy_latents,
                                                             noise_scheduler.config.prediction_type,
                                                             alpha_schedule,
                                                             sigma_schedule, )
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                total_frame = pred_x_0_stu.shape[2]
                random_frame_idx = random.randint(0, total_frame - 1)
                with torch.no_grad():
                    frame = pred_x_0_stu[:, :, random_frame_idx, :, :]
                    if frame.ndim == 4:
                        frame = frame.unsqueeze(2)
                    video_processor = VideoProcessor(do_resize=False, vae_scale_factor=vae_scale_factor)
                    video_tensor = decode_latents(vae, frame)
                    output_type = 'pil'
                    video = video_processor.postprocess_video(video=video_tensor, output_type=output_type)[0]
                    # check video
                    # saving
                    save_dir = os.path.join(output_dir, f"sanity_check/aesthetic_{str(global_step).zfill(3)}.png")
                    video[0].save(save_dir)
                    video_tensor = load_img(video).unsqueeze(0)
                    frames = video_tensor
                if args.do_aesthetic_loss:
                    aesthetic_loss, aesthetic_rewards = aesthetic_loss_fnc(
                        frames.to(device=device, dtype=weight_dtype))  # video_frames_ in range [-1, 1]
                    wandb.log({"aesthetic_loss": aesthetic_loss.item()}, step=global_step)
                    total_loss += args.aesthetic_score_weight * aesthetic_loss
                if args.do_t2i_loss:
                    if args.do_hps_loss:
                        hps_loss, hps_rewards = t2i_loss_fnc(frames.to(device=device, dtype=weight_dtype),
                                                             batch['text'])
                        wandb.log({"hps_loss": hps_loss.item()}, step=global_step)
                        total_loss += args.t2i_score_weight * hps_loss
                    elif args.clip_flant5_score:
                        t2i_score = t2i_loss_fnc(frames.to(device=device, dtype=weight_dtype), batch['text'])
                        t2i_loss = 1 - t2i_score
                        wandb.log({"t2i_loss": t2i_loss.item()}, step=global_step)
                        total_loss += args.t2i_score_weight * t2i_loss

            optimizer.zero_grad()
            total_loss.backward()
            if args.mixed_precision_training:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters_list, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
                optimizer.step()
            """
            for name, param in student_unet.named_parameters():
                if param.requires_grad:
                    if name not in unet_dict:
                        unet_dict[name] = []
                        unet_dict[name].append(param.detach().cpu())
                    else:
                        before = unet_dict[name][-1]
                        present = param.detach().cpu()
                        unet_dict[name] = []
                        unet_dict[name].append(param.detach().cpu())
                        equal_check = torch.equal(before, present)
                        print(f'{name} equal check = {equal_check}')
            """

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            wandb.log({"train_loss": total_loss.item()}, step=global_step)
            wandb.log({"loss_vlb": loss_vlb.item()}, step=global_step)
            wandb.log({"loss_distill": loss_distill.item()}, step=global_step)
            if args.motion_control:
                if type(loss_feature) == torch.Tensor:
                    wandb.log({"loss_feature": loss_feature.item()}, step=global_step)

            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()

        if args.motion_control:
            student_motion_controller.reset()
            teacher_motion_controller.reset()


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
    args = parser.parse_args()
    name = Path(args.config).stem
    main(args)
