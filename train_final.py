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
#from utils.validation import log_validation
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

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def get_predicted_original_sample(
    model_output, timesteps, sample, prediction_type, alphas, sigmas
):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)


    # following predictin type,
    # we can get pred_x_0 as ...
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
        # sample = alphas * x_0 + sigmas * model_output
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0
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
    teacher_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=teacher_adapter,
                                                       torch_dtpe=weight_dtype)

    #noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path,
    #                                                subfolder="scheduler",
    #                                                revision=False,
    #                                                rescale_betas_zero_snr=False,
    #                                                beta_schedule='linear',)
    noise_scheduler = DDPMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    args.num_ddim_timesteps = 50
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),
                        timesteps=noise_scheduler.config.num_train_timesteps,
                        ddim_timesteps=args.num_ddim_timesteps,)
    logger.info(f' (4.1) tokenizer, text encoder, vae')
    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder
    vae = teacher_pipe.vae
    logger.info(f' (4.2) teacher U-Net')
    guidance_scale = 2.0
    window_size = 16
    skip_layers, skip_layers_dot = find_layer_name(args.skip_layers)
    teacher_unet = teacher_pipe.unet
    teacher_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                             frame_num=16,
                                                             full_attention=args.full_attention,
                                                             window_attention=False,
                                                             window_size=window_size,
                                                             total_frame_num=args.num_frames,
                                                             skip_layers=skip_layers,
                                                             is_teacher=True)  # 32
    regiter_motion_attention_editor_diffusers(teacher_unet, teacher_motion_controller)

    logger.info(f' (4.3) student U-Net')
    student_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir,
                                                    torch_dtpe=weight_dtype).to(device, dtype=weight_dtype)
    student_config = student_adapter.config
    if args.random_init:  # make another module
        student_adapter = MotionAdapter(**student_config)
    student_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=student_adapter,
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
    # student_unet.requires_grad_(False)

    # make scheduler

    logger.info(f' step 5. count trainable parameters')
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

    logger.info(f' step 10. handle saving and loading')

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

    progress_bar = tqdm(range(global_step, args.max_train_steps), desc = "Steps")
    for epoch in range(first_epoch, args.num_train_epochs):
        teacher_unet.train()
        student_unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            #if step == 0:
            #    log_validation()
            #    save_unet()

            # [1] load and process the image and text conditioning
            pixel_values = batch["pixel_values"]
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                with torch.no_grad():
                    latents = vae.encode(pixel_values.to(device, dtype=weight_dtype)).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215
            bsz = latents.shape[0]

            # [2] sample a random timestep from ODE solver timesteps without bias
            # For the DDIM solver, the timestep schedule is [T-1, T-k-1, ...]
            topk = (noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps)
            index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device) # [0, 50]
            start_timesteps = solver.ddim_timesteps[index]
            timesteps = start_timesteps - topk
            timesteps = torch.where(timesteps < 0, timesteps + noise_scheduler.config.num_train_timesteps, timesteps)
            print(f'topk : {topk}')
            print(f'index : {index}')
            print(f'solver.ddim_timesteps : {solver.ddim_timesteps}')
            print(f'start_timesteps : {start_timesteps}')
            print(f'timesteps : {timesteps}')

            c_skip, c_out = scalings_for_boundary_conditions(timesteps)
            c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]

            c_skip, c_out = scalings_for_boundary_conditions(timesteps, timestep_scaling=args.timestep_scaling_factor)
            c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

            print(f'c_skip_start : {c_skip_start} | c_out_start : {c_out_start}')
            print(f'c_skip : {c_skip} | c_out : {c_out}')

            use_pred_x0 = True
            if use_pred_x0:
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        noise = torch.randn_like(latents)
                        last_timestep = solver.ddim_timesteps[-1].unsqueeze(0) # last timestep = 999
                        last_timestep = last_timestep.repeat(bsz)
                        if args.use_lora:
                            # last time noise pred
                            x_0_noise_pred = unet(noise.float(), last_timestep, timestep_cond=None,
                                                  encoder_hidden_states=prompt_embeds.float(), ).sample
                        else:
                            x_0_noise_pred = target_unet(noise.float(), last_timestep, timestep_cond=w_embedding,
                                                         encoder_hidden_states=prompt_embeds.float(), ).sample
                        latents = get_predicted_original_sample(x_0_noise_pred, last_timestep, noise,
                                                                noise_scheduler.config.prediction_type,
                                                                alpha_schedule, sigma_schedule, ) # pred_x0

            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            # [1] Student Unet
            noise = torch.randn_like(latents)
            noisy_model_input_list = []
            for b_idx in range(bsz):
                if index[b_idx] != args.num_ddim_timesteps - 1:
                    noisy_model_input = noise_scheduler.add_noise(latents[b_idx, None], noise[b_idx, None],
                                                                  start_timesteps[b_idx, None], )
                else:
                    # hard swap input to pure noise to ensure zero terminal SNR
                    noisy_model_input = noise[b_idx, None]
                noisy_model_input_list.append(noisy_model_input)
            noisy_model_input = torch.cat(noisy_model_input_list, dim=0)
            # unet noise pred on start_timesteps
            noise_pred = unet(noisy_model_input, start_timesteps, timestep_cond=None if args.use_lora else w_embedding,
                              encoder_hidden_states=prompt_embeds, ).sample
            # unet x_0 stu??
            # x_0 predict model output --> makes final model prediction
            pred_x_0_stu = get_predicted_original_sample(noise_pred, start_timesteps, noisy_model_input,
                                                         noise_scheduler.config.prediction_type, alpha_schedule,
                                                         sigma_schedule, )
            model_pred = (c_skip_start * noisy_model_input + c_out_start * pred_x_0_stu)

            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
            # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
            # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
            # solver timestep.
            with torch.no_grad():
                with torch.autocast("cuda"):
                    # ------------------------------------------------------------------------------------------------------------------------------------------------------
                    # [2] Teacher Model
                    # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                    cond_teacher_output = teacher_unet(noisy_model_input.to(weight_dtype), start_timesteps,
                                                       encoder_hidden_states=prompt_embeds.to(weight_dtype), ).sample
                    cond_pred_x0 = get_predicted_original_sample(cond_teacher_output, start_timesteps,
                                                                 noisy_model_input,
                                                                 noise_scheduler.config.prediction_type, alpha_schedule,
                                                                 sigma_schedule, )
                    cond_pred_noise = get_predicted_noise(cond_teacher_output, start_timesteps, noisy_model_input,
                                                          noise_scheduler.config.prediction_type, alpha_schedule,
                                                          sigma_schedule, )
                    uncond_teacher_output = teacher_unet(noisy_model_input.to(weight_dtype), start_timesteps,
                                                         encoder_hidden_states=uncond_prompt_embeds.to(
                                                             weight_dtype), ).sample
                    uncond_pred_x0 = get_predicted_original_sample(uncond_teacher_output, start_timesteps,
                                                                   noisy_model_input,
                                                                   noise_scheduler.config.prediction_type,
                                                                   alpha_schedule, sigma_schedule, )
                    uncond_pred_noise = get_predicted_noise(uncond_teacher_output, start_timesteps, noisy_model_input,
                                                            noise_scheduler.config.prediction_type, alpha_schedule,
                                                            sigma_schedule, )

                    # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                    # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                    # print(f"cond_pred_x0: {cond_pred_x0.shape}; uncond_pred_x0: {uncond_pred_x0.shape}; cond_pred_noise: {cond_pred_noise.shape}; uncond_pred_noise: {uncond_pred_noise.shape}; w: {w.shape}")
                    pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                    pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)

                    # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                    # augmented PF-ODE trajectory (solving backward in time)
                    # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                    x_prev = solver.ddim_step(pred_x0, pred_noise, index)

            #
            # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
            # Note that we do not use a separate target network for LCM-LoRA distillation.
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    if args.use_lora:
                        target_noise_pred = unet(
                            x_prev.float(),
                            timesteps,
                            timestep_cond=None,
                            encoder_hidden_states=prompt_embeds.float(),
                        ).sample
                    else:
                        target_noise_pred = target_unet(
                            x_prev.float(),
                            timesteps,
                            timestep_cond=w_embedding,
                            encoder_hidden_states=prompt_embeds.float(),
                        ).sample
                pred_x_0 = get_predicted_original_sample(
                    target_noise_pred,
                    timesteps,
                    x_prev,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                target = c_skip * x_prev + c_out * pred_x_0 # distillation the next (next should be same with before of offline)










            exit()
            noise = torch.randn_like(latents).to(device, dtype=weight_dtype)
            # ------------------------------------------------------------------------------------------------------- #
            # Change Here !
            # 0, ..., 19

            basic_timepool = noise_scheduler.config.num_train_timesteps
            print(f' [basic_timepool] : {basic_timepool}')
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            print(f' [timesteps] : {timesteps}')
            timesteps = timesteps.long()  # torch
            print(f'type of timesteps : {type(timesteps)}')
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
                    teacher_motion_controller.reset()
                    teacher_motion_controller.layerwise_hidden_dict = {}
            distill_target = teacher_model_pred
            ########################################################################################################
            # Here Problem (student, problem)
            student_model_pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # x0_predict = lcm_scheduler.step(student_model_pred, timesteps, latents).denoised

            sample = noisy_latents
            model_output = student_model_pred
            # 1. get previous step value

            # 2. compute alphas, betas
            alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            # 3. Get scalings for boundary conditions
            c_skip, c_out = scalings_for_boundary_condition_discrete(timesteps)
            # 4. Different Parameterization:

            parameterization = noise_scheduler.config.prediction_type
            if parameterization == "epsilon":  # noise-prediction
                pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            elif parameterization == "sample":  # x-prediction
                pred_x0 = model_output
            elif parameterization == "v_prediction":  # v-prediction
                pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output


            ########################################################################################################
            if args.motion_control:
                s_hdict = student_motion_controller.layerwise_hidden_dict
                student_motion_controller.layerwise_hidden_dict = {}
                loss_feature = 0
                for layer_name in s_hdict.keys():
                    s_h = s_hdict[layer_name]
                    t_h = t_hdict[layer_name]
                    for s_h_, t_h_ in zip(s_h, t_h):
                        loss_feature += F.mse_loss(s_h_.float(), t_h_.float(), reduction="mean")
                student_motion_controller.reset()
                teacher_motion_controller.reset()
            #########################################################################################################
            # [1] task loss
            loss_vlb = F.mse_loss(student_model_pred.float(), target.float(), reduction="mean")
            loss_distill = F.mse_loss(student_model_pred.float(), teacher_model_pred.float(), reduction="mean")
            loss = args.vlb_weight * loss_vlb + args.distill_weight * loss_distill
            if args.motion_control:
                loss = args.vlb_weight * loss_vlb + args.distill_weight * loss_distill + args.loss_feature_weight * loss_feature
            optimizer.zero_grad()
            loss.backward()

            if args.mixed_precision_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters_list, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
                optimizer.step()

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

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            wandb.log({"train_loss": loss.item()}, step=global_step)
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

    exit()


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
    args = parser.parse_args()
    name = Path(args.config).stem
    main(args)
