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
# from utils.validation import log_validation
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
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.models import MotionAdapter
from diffusers.pipelines import AnimateDiffPipeline
from data.dataset_gen import DistillWebVid10M
from utils.layer_dictionary import find_layer_name
from attn.masactrl_utils import (regiter_attention_editor_diffusers, regiter_motion_attention_editor_diffusers)
from attn.masactrl import MutualSelfAttentionControl, MutualMotionAttentionControl
from scheduler import LCMScheduler
from diffusers.utils import export_to_gif, load_image
import GPUtil
import json
from diffusers.training_utils import EMAModel
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
from diffusers import DDPMScheduler
from scheduler.solver import DDIMSolver
from transformers import CLIPTokenizer, CLIPTextModel
from torch import nn
from utils.diffusion_misc import *

#
def main(args):

    GPUtil.showUtilization()
    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )

    logger.info(f'\n step 1. path')
    logging_dir = Path(args.output_dir, "logs")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    folder_name = args.sub_folder_name
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    save_folder = os.path.join(output_dir, "samples")
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger.info(f' step 2. seed / weight_dtype / device')
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_dtype = torch.float32

    logger.info(f' step 3. wandb logging')
    wandb.init(project=args.project, entity='dreamyou070', mode='online', name=f'experiment_{args.sub_folder_name}')

    logger.info(f' step 4. noise scheduler')
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path,
                                                    subfolder="scheduler",
                                                    revision=args.teacher_revision,
                                                    beta_schedule=args.beta_schedule, )

    print(f' step 5. ODE Solver (erasing noise)')
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),
                        timesteps=noise_scheduler.config.num_train_timesteps,
                        ddim_timesteps=args.num_ddim_timesteps, )

    print(f' step 6. pretrained_teacher_model')

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()


    print(f' step 7. teacher model')
    teacher_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path,
                                                        subfolder="unet",
                                                        revision=args.teacher_revision, )
    teacher_motion_adapter = MotionAdapter.from_pretrained(args.teacher_motion_adapter_path)
    teacher_unet = UNetMotionModel.from_unet2d(teacher_unet, teacher_motion_adapter)
    guidance_scale = 2.0
    window_size = 16
    skip_layers, skip_layers_dot = find_layer_name(args.skip_layers)
    teacher_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                             frame_num=16,
                                                             full_attention=args.full_attention,
                                                             window_attention=False,
                                                             window_size=window_size,
                                                             total_frame_num=args.num_frames,
                                                             skip_layers=skip_layers,
                                                             is_teacher=True)  # 32
    regiter_motion_attention_editor_diffusers(teacher_unet, teacher_motion_controller)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    logger.info(f' (4.3) student U-Net')
    student_adapter = MotionAdapter.from_pretrained(args.student_motion_adapter_path,
                                                    torch_dtpe=weight_dtype).to(device, dtype=weight_dtype)
    student_adapter_config = student_adapter.config

    student_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism" ,#args.pretrained_model_path,
                                                       motion_adapter=student_adapter,
                                                       torch_dtpe=weight_dtype)
    student_unet = student_pipe.unet
    student_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                             frame_num=16,
                                                             full_attention=args.full_attention,
                                                             window_attention=args.window_attention,
                                                             window_size=window_size,
                                                             total_frame_num=args.num_frames,
                                                             skip_layers=skip_layers,
                                                             is_teacher=False, )
    regiter_motion_attention_editor_diffusers(student_unet, student_motion_controller)

    logger.info(f' (4.4) freeze vae / text encoder / teache unet')
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    student_unet.requires_grad_(False)

    logger.info(f' step 6. count trainable parameters')
    parameters_list = []
    student_unet.requires_grad_(True)
    for name, param in student_unet.named_parameters():
        if 'motion' in name:
            param.requires_grad = True
            if skip_layers_dot is not None:
                for skip_layer in skip_layers_dot:
                    if skip_layer in name:
                        param.requires_grad = False
                        break
        else:
            param.requires_grad = False
    for name, param in student_unet.named_parameters():
        if param.requires_grad:
            parameters_list.append(param)
    print(f'len(parameters_list) : {len(parameters_list)}')

    logger.info(f' step 9. handle mixed precision and device placement')
    weight_dtype = torch.float32
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    teacher_unet.to(device, dtype=weight_dtype)
    student_unet.to(device, dtype=weight_dtype)  # this cannot be ?
    alpha_schedule = alpha_schedule.to(device, dtype=weight_dtype)
    sigma_schedule = sigma_schedule.to(device, dtype=weight_dtype)
    solver.to(device)
    if args.gradient_checkpointing:
        # gradient checkpointing reduce momoery consume
        student_unet.enable_gradient_checkpointing()


    logger.info(f' step 11. enable optimizer')
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.per_gpu_batch_size)

    logger.info(f' step 12. optimizer creationg')
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon, )

    rec_txt1 = open('recording_param_untraining.txt', 'w')
    rec_txt2 = open('recording_param_training.txt', 'w')
    for name, para in student_unet.named_parameters():
        if para.requires_grad is False:
            rec_txt1.write(f'{name}\n')
        else:
            rec_txt2.write(f'{name}\n')
    rec_txt1.close()
    rec_txt2.close()

    logger.info(f' step 13. dataset creationg and data processing')
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

    logger.info(f' step 14. LR Scheduler creation')
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps)

    logger.info(f' step 15. Reward Functions')

    from dpo import aesthetic_loss_fn
    aesthetic_loss_fn = aesthetic_loss_fn(grad_scale=0.1,
                                        aesthetic_target=10,
                                        torch_dtype=weight_dtype,
                                        device=device)

    logger.info(f' step 15. Prepare for training')
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    student_unet.to(device, dtype=weight_dtype)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print(f'args.max_train_steps : {args.max_train_steps}')
    print(f'num_update_steps_per_epoch : {num_update_steps_per_epoch}')
    print(f'args.num_train_epochs : {args.num_train_epochs}')

    print(f' step 16. Train !')
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

    # From LCMScheduler.get_scalings_for_boundary_condition_discrete
    def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
        scaled_timestep = timestep_scaling * timestep
        c_skip = sigma_data ** 2 / (scaled_timestep ** 2 + sigma_data ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + sigma_data ** 2) ** 0.5
        return c_skip, c_out

    uncond_input_ids = tokenizer([""] * args.train_batch_size, return_tensors="pt",
                                 padding="max_length",max_length=77,).input_ids.to(device=device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]
    unet_dict = {}
    progress_bar = tqdm(range(global_step, args.max_train_steps), desc="Steps")
    for epoch in range(first_epoch, args.num_train_epochs):
        teacher_unet.train()
        student_unet.train()
        for step, batch in enumerate(train_dataloader):
            loss_dict = {}
            loss_unet_total = 0.0
            # ------------------ # ------------------ # ------------------ # ------------------ # ------------------ #

            if step == 0 :
                 print(f' [epoch {epoch}] evaluation')
                 validation_prompts = ["A video of a woman, having a selfie",
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
                     eval_adapter = MotionAdapter.from_config(student_adapter_config)
                     eval_adapter_state_dict = {}
                     eval_dict = eval_adapter.state_dict()
                     for key, value in eval_dict.items():
                         if key in student_unet.state_dict().keys():
                             eval_adapter_state_dict[key] = value
                     eval_adapter.load_state_dict(eval_adapter_state_dict)
                     evaluation_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                                           motion_adapter = eval_adapter)
                     eval_unet = evaluation_pipe.unet
                     eval_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                           frame_num=16,
                                                                           full_attention=args.full_attention,
                                                                           window_attention=args.window_attention,
                                                                           window_size=window_size,
                                                                           total_frame_num=args.num_frames,
                                                                           skip_layers=skip_layers,
                                                                           is_teacher=False, )
                     regiter_motion_attention_editor_diffusers(eval_unet, eval_motion_controller)

                     # load state dict
                     trained_value = student_unet.state_dict()
                     eval_unet.load_state_dict(trained_value)
                     scheduler_basic_config = noise_scheduler.config
                     # [3] scheduler
                     evaluation_pipe.scheduler = LCMScheduler.from_config(scheduler_basic_config)
                     # [4] lcm lora
                     evaluation_pipe.load_lora_weights("wangfuyun/AnimateLCM",
                                                       weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                                       adapter_name="lcm-lora")
                     evaluation_pipe.set_adapters(["lcm-lora"], [0.8])
                     evaluation_pipe.enable_vae_slicing()
                     evaluation_pipe.to('cuda')

                     num_frames = args.num_frames
                     num_inference_steps = args.inference_step
                     n_prompt = "bad quality, worse quality, low resolution"
                     for p, prompt in enumerate(validation_prompts) :
                         save_p = str(p).zfill(3)
                         output = evaluation_pipe(prompt=prompt,
                                                  negative_prompt=n_prompt,
                                                  num_frames=num_frames,
                                                  guidance_scale=guidance_scale,
                                                  num_inference_steps=num_inference_steps,
                                                  generator=torch.Generator("cpu").manual_seed(args.seed), )
                         student_motion_controller.reset()
                         frames = output.frames[0]
                         export_to_gif(frames, os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}_{save_p}.gif'))
                         export_to_video(frames, os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}_{save_p}.mp4'))
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

            with torch.no_grad():
                prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding="max_length",
                                       truncation=True, return_tensors="pt").input_ids.to(device)
                prompt_embeds = text_encoder(prompt_ids)[0].to(device, dtype=weight_dtype)
            pixel_values = batch["pixel_values"]
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                with torch.no_grad():
                    latents = vae.encode(pixel_values.to(device, dtype=weight_dtype)).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215
            latents = latents.to(weight_dtype) # batch, channel, frame, h, w
            bsz = latents.shape[0]
            # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
            # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
            time_elapse = (noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps)
            index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
            start_timesteps = solver.ddim_timesteps[index]
            timesteps_prev = start_timesteps - time_elapse
            timesteps_prev = torch.where(timesteps_prev < 0, torch.zeros_like(timesteps_prev), timesteps_prev)

            # 3. Get boundary scalings for start_timesteps and (end) timesteps.
            c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps,timestep_scaling=args.timestep_scaling_factor)
            c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
            c_skip_prev, c_out_prev = scalings_for_boundary_conditions(timesteps_prev,timestep_scaling=args.timestep_scaling_factor)
            c_skip_prev, c_out_prev = [append_dims(x, latents.ndim) for x in [c_skip_prev, c_out_prev]]

            # 4. Get the noise schedule for the current timestep t_n.
            noise = torch.randn_like(latents)
            noisy_model_input_list = []
            for b_idx in range(bsz):
                if index[b_idx] != args.num_ddim_timesteps - 1:
                    noisy_model_input = noise_scheduler.add_noise(latents[b_idx, None],noise[b_idx, None],start_timesteps[b_idx, None],)
                else:
                    noisy_model_input = noise[b_idx, None]
                noisy_model_input_list.append(noisy_model_input)
            noisy_model_input = torch.cat(noisy_model_input_list, dim=0)

            # --------------------------------------------------------------------------------------------------------- #
            # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
            noise_pred = student_unet(noisy_model_input,start_timesteps, encoder_hidden_states=prompt_embeds,).sample
            pred_x_0_stu = get_predicted_original_sample(noise_pred, start_timesteps, noisy_model_input, noise_scheduler.config.prediction_type, alpha_schedule, sigma_schedule,)
            # tensor
            # shape of pred_x_0_stu : batch, channel=4, frame, h=64, w=64
            # decoding
            frames = pred_x_0_stu
            def decode_latents(latents):
                latents = 1 / vae.config.scaling_factor * latents

                batch_size, channels, num_frames, height, width = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

                image = vae.decode(latents).sample
                video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                video = video.float()
                return video

            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            from diffusers.video_processor import VideoProcessor
            with torch.no_grad() :
                first_frame = pred_x_0_stu[:,:, 0, :, :]
                if first_frame.ndim == 4:
                    first_frame = first_frame.unsqueeze(2)
                video_processor = VideoProcessor(do_resize=False, vae_scale_factor=vae_scale_factor)
                video_tensor = decode_latents(first_frame)
                output_type = 'pil'
                video = video_processor.postprocess_video(video=video_tensor, output_type=output_type)[0]

                def load_img(img):
                    rgb_img = np.array(img, np.float32).squeeze()
                    img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()

                    img_tensor = (img_tensor / 255. - 0.5) * 2
                    return img_tensor
                video_tensor = load_img(video).unsqueeze(0)
                print(F'video_tensor : {video_tensor.shape}')
                frames = video_tensor
                #bs, c, h_, w_ = frames.shape #
                #assert nf == 1
                #frames = frames.squeeze(2)
                if args.do_aesthetic_loss :
                    aesthetic_loss, aesthetic_rewards = aesthetic_loss_fn(frames.to(device = device, dtype = weight_dtype))  # video_frames_ in range [-1, 1]
                    #export_to_gif(frames, os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}.gif'))
                    #export_to_video(frames, os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}.mp4'))
                    loss_dict["aesthetic_loss"] = aesthetic_loss
                    loss_unet_total += aesthetic_loss
                    wandb.log({"aesthetic_loss": aesthetic_loss.item()}, step=global_step)

            model_pred = (c_skip_start * noisy_model_input + c_out_start * pred_x_0_stu)
            # present timestep self-consistency value
            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()

            # --------------------------------------------------------------------------------------------------------- #
            # 8.1 Compute next timestep from teacher

            with torch.no_grad():
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w = w.reshape(bsz, 1, 1, 1, 1)
                w = w.to(device=latents.device, dtype=latents.dtype)
                with torch.autocast("cuda"):
                    cond_teacher_output = teacher_unet(noisy_model_input.to(weight_dtype),start_timesteps,
                                                       encoder_hidden_states=prompt_embeds.to(weight_dtype),).sample
                    if args.motion_control:
                        student_motion_controller.reset()
                        teacher_motion_controller.reset()
                    cond_pred_x0 = get_predicted_original_sample(cond_teacher_output,start_timesteps,noisy_model_input,
                                                                 noise_scheduler.config.prediction_type,alpha_schedule,sigma_schedule,)
                    cond_pred_noise = get_predicted_noise(cond_teacher_output,start_timesteps,noisy_model_input,noise_scheduler.config.prediction_type,
                                                          alpha_schedule,sigma_schedule,)
                    uncond_teacher_output = teacher_unet(noisy_model_input.to(weight_dtype),start_timesteps,
                                                         encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),).sample
                    if args.motion_control:
                        student_motion_controller.reset()
                        teacher_motion_controller.reset()
                    uncond_pred_x0 = get_predicted_original_sample(uncond_teacher_output,start_timesteps,noisy_model_input,
                                                                   noise_scheduler.config.prediction_type,alpha_schedule,sigma_schedule,)
                    uncond_pred_noise = get_predicted_noise(uncond_teacher_output,start_timesteps,noisy_model_input,noise_scheduler.config.prediction_type,
                                                            alpha_schedule,sigma_schedule,)
                    pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                    pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                    x_prev = solver.ddim_step(pred_x0, pred_noise, index)
            # --------------------------------------------------------------------------------------------------------- #
            # (8.2) Get target LCM prediction on x_prev, w, c, t_n (timesteps)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):

                    target_noise_pred = student_unet(x_prev.float(),timesteps_prev,timestep_cond=None,encoder_hidden_states=prompt_embeds.float(),).sample
                if args.motion_control:
                    student_motion_controller.reset()
                    teacher_motion_controller.reset()
                # This Is Good Result #
                pred_x_0 = get_predicted_original_sample(target_noise_pred,timesteps_prev,x_prev,noise_scheduler.config.prediction_type,alpha_schedule,sigma_schedule,)
                # previous timestep self-consistency value
                target = c_skip_prev * x_prev + c_out_prev * pred_x_0
            # --------------------------------------------------------------------------------------------------------- #

            with torch.no_grad():
                target_cd = prepare_cd_target(target.float(), args.cd_target)
                model_pred_cd = prepare_cd_target(model_pred, args.cd_target)
            if args.loss_type == "l2":
                loss_unet_cd = F.mse_loss(model_pred_cd.float(), target_cd.float(), reduction="mean")
            elif args.loss_type == "huber":
                loss_unet_cd = torch.mean(torch.sqrt((model_pred_cd.float() - target_cd.float()) ** 2+ args.huber_c ** 2)- args.huber_c)
            loss_dict["loss_unet_cd"] = loss_unet_cd
            wandb.log({"distill_loss": loss_unet_cd.item()}, step=global_step)

            loss_unet_total += loss_unet_cd
            optimizer.zero_grad()
            loss_unet_total.backward()

            torch.nn.utils.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
            optimizer.step()


            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            wandb.log({"train_loss": loss_unet_total.item()}, step=global_step)
            #wandb.log({"loss_vlb": loss_vlb.item()}, step=global_step)
            #wandb.log({"loss_distill": loss_distill.item()}, step=global_step)
            #if args.motion_control:
            #    if type(loss_feature) == torch.Tensor:
            #        wandb.log({"loss_feature": loss_feature.item()}, step=global_step)

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
    parser.add_argument('--teacher_motion_adapter_path', type=str, default="guoyww/animatediff-motion-adapter-v1-5-2")
    parser.add_argument('--student_motion_adapter_path', type=str, default="wangfuyun/AnimateLCM")
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
    parser.add_argument("--timestep_scaling_factor",type=float,default=10.0,
                        help=("The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
                              " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
                              " suffice."),)
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",)
    parser.add_argument(
        "--beta_schedule",
        default="scaled_linear",
        type=str,
        help="The schedule to use for the beta values.",
    )
    parser.add_argument("--num_ddim_timesteps",type=int,default=50,
                        help="The number of timesteps to use for DDIM sampling.",)
    parser.add_argument("--w_min",type=float,default=5.0,required=False,
                        help=("The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
                              " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
                              " compared to the original paper."),)
    parser.add_argument("--w_max",type=float,default=15.0,required=False,
                        help=("The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
                              " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
                              " compared to the original paper."),)
    parser.add_argument(
        "--cd_target",
        type=str,
        default="raw",
        choices=[
            "raw",
            "diff",
            "freql",
            "freqh",
            "learn",
            "hlearn",
            "lcor",
            "gcor",
            "sgcor",
            "sgcord",        ],
        help=(
            "The loss target for consistency distillation."
            " raw: use the raw latent;"
            " diff: use latent difference;"
            " freql: use latent low-frequency component;"
            " freqh: use latent high-frequency component;"
            " learn: use light-weight learnable spatial head;"
            " hlearn: use heavy-weight learnable spatial head;"
            " lcor: use latent local correlation;"
            " gcor: use latent global correlation;"
            " sgcor: use latent scaled global correlation;"
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument("--do_aesthetic_loss",
                        action ='store_true',)
    args = parser.parse_args()
    name = Path(args.config).stem
    main(args)
