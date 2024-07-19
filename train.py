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
from  diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
from diffusers import DDPMScheduler
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
        if  len(args.skip_layers) > 0:
            for i  in range(len(args.skip_layers)) :
                layer_name = args.skip_layers[i]
                if i == len(args.skip_layers) - 1:
                    folder_name += f"{layer_name}"
                else:
                    folder_name += f"{layer_name}_"

    logger.info(f'\n step 2. preparing folder')
    torch.manual_seed(args.seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    folder_name = args.sub_folder_name
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    print(f' step 4. wandb logging')
    # project
    wandb.init(project=args.project,
               entity='dreamyou070',
               mode='online',
               name=f'experiment_{args.sub_folder_name}')
    weight_dtype = torch.float32

    print(f' step 5. make directory')
    save_folder = os.path.join(output_dir, "samples")
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f' step 6. make model')
    logger.info(f' (4.1) teacher')
    teacher_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=teacher_adapter, torch_dtpe=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config)
    teacher_pipe.scheduler = noise_scheduler

    logger.info(f' (4.2) student')
    student_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir,
                                                    torch_dtpe=weight_dtype).to(device, dtype=weight_dtype)
    student_config = student_adapter.config
    pretrained_state_dict = student_adapter.state_dict()
    if args.random_init: # make another module
        print(f' student adapter random initialization')
        student_adapter = MotionAdapter(**student_config)
        random_state_dict = student_adapter.state_dict()
        for key in random_state_dict.keys():
            raw_value = random_state_dict[key]
            pretrained_value = pretrained_state_dict[key]
            equal_check = torch.equal(raw_value, pretrained_value)
            print(f' [randomize] {key} equal_check with pretrained : {equal_check}')

    student_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path,
                                                       motion_adapter=student_adapter,
                                                       torch_dtpe=weight_dtype)
    vae = teacher_pipe.vae
    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder
    teacher_unet = teacher_pipe.unet
    student_unet = student_pipe.unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    student_unet.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    teacher_unet.to(device, dtype=weight_dtype)
    student_unet.to(device, dtype=weight_dtype) # this cannot be ?
    # make scheduler

    logger.info(f' (4.4) motion controller')
    window_size = 16
    guidance_scale = args.guidance_scale
    inference_step = args.inference_step
    student_motion_controller = None
    teacher_motion_controller = None
    if args.motion_control:
        student_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=args.window_attention,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=False,)
        regiter_motion_attention_editor_diffusers(student_unet, student_motion_controller)

        teacher_motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scale,
                                                                 frame_num=16,
                                                                 full_attention=args.full_attention,
                                                                 window_attention=False,
                                                                 window_size=window_size,
                                                                 total_frame_num=args.num_frames,
                                                                 skip_layers=skip_layers,
                                                                 is_teacher=True)  # 32
        regiter_motion_attention_editor_diffusers(teacher_unet, teacher_motion_controller)

    print(f' step 8. Create EMA for the unet')
    if args.gradient_checkpointing:
         #gradient checkpointing reduce momoery consume 
        student_unet.enable_gradient_checkpointing()

    print(f' step 9. Scale lr')
    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.per_gpu_batch_size)

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
                    if skip_layer in name :
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
    student_unet.to(device, dtype= weight_dtype)

    print(f' step 16. training num')
    # train_dataloader = 300, gradient_accumulation_steps = 1
    # num_update_steps_per_epoch = 300
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs

    # max_train_steps = 100000
    # num_update_steps_per_epoch = 300
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print(f'args.max_train_steps : {args.max_train_steps}')
    print(f'num_update_steps_per_epoch : {num_update_steps_per_epoch}')
    print(f'args.num_train_epochs : {args.num_train_epochs}')

    print(f' step 17. Train!')
    total_batch_size = args.per_gpu_batch_size * args.gradient_accumulation_steps # 300
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
    prompt = "A video of a woman, having a selfie"
    n_prompt = "bad quality, worse quality, low resolution"
    num_frames = 16
    guidance_scale = 1.5
    num_inference_steps = 6
    ##########################################################################################
    # Only show the progress bar once on each machine.
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision_training else None
    progress_bar.set_description("Steps")
    unet_dict = {}
    for epoch in range(first_epoch, args.num_train_epochs):
        teacher_unet.train()
        student_unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            """
            
                print(f' [epoch {epoch}] saving model')
                # [1] State Saving
                trained_value = student_unet.state_dict()
                save_state_dict= {}
                for trained_key, trained_value in trained_value.items():
                    if 'motion' in trained_key :
                        save_state_dict[trained_key] = trained_value.to('cpu')
                save_epoch = str(epoch).zfill(3)
                torch.save(save_state_dict, os.path.join(output_dir, f"checkpoints/checkpoint_epoch_{save_epoch}.pt"))

                # [2] Evaluation
                print(f' [epoch {epoch}] evaluation')
                with torch.no_grad():
                    # model copy (detach)
                    student_unet_config = student_unet.config
                    eval_unet = UNetMotionModel.from_config(student_unet_config)
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
                    evaluation_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                                                          unet=eval_unet)
                    evaluation_pipe.scheduler = noise_scheduler
                    evaluation_pipe.load_lora_weights("wangfuyun/AnimateLCM",
                                                      weight_name="AnimateLCM_sd15_t2v_lora.safetensors",adapter_name="lcm-lora")
                    evaluation_pipe.set_adapters(["lcm-lora"], [0.8])
                    evaluation_pipe.enable_vae_slicing()
                    evaluation_pipe.to('cuda')
                    output = evaluation_pipe(prompt=prompt,
                                             negative_prompt=n_prompt,
                                             num_frames=num_frames,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps,
                                             generator=torch.Generator("cpu").manual_seed(args.seed), )
                    student_motion_controller.reset()
                    frames = output.frames[0]
                    export_to_gif(frames, os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}.gif'))
                    export_to_video(frames, os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}.mp4'))
                    text_dir = os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}.txt')
                    with open(text_dir, 'w') as f:
                        f.write(f'prompt : {prompt}\n')
                        f.write(f'n_prompt : {n_prompt}\n')
                        f.write(f'guidance_scale : {guidance_scale}\n')
                        f.write(f'num_inference_steps : {num_inference_steps}\n')
                        f.write(f'seed : {args.seed}\n')
                    fps=10
                    wandb.log({"video": wandb.Video(data_or_path=os.path.join(save_folder, f'sample_epoch_{str(epoch).zfill(3)}.gif'),
                                                    caption=f'epoch_{epoch}', fps=fps)})

                    del evaluation_pipe, eval_unet

            
            if epoch != 0 :
                before_epoch = epoch - 1
                before_state_dict = torch.load(os.path.join(output_dir, f"checkpoints/checkpoint_epoch{before_epoch}.pt"), map_location="cpu")
                present_state_dict = torch.load(os.path.join(output_dir, f"checkpoints/checkpoint_epoch{epoch}.pt"), map_location="cpu")
                for key, value in present_state_dict.items():
                    present_state = present_state_dict[key]
                    before_state = before_state_dict[key]
                    if torch.equal(present_state, before_state):
                        print(f'epoch {epoch} {key} is equal')
            
            """
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
            # ------------------------------------------------------------------------------------------------------- #
            # Change Here !
            # 0, ..., 19

            basic_timepool = noise_scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long() # torch
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
            #x0_predict = lcm_scheduler.step(student_model_pred, timesteps, latents).denoised

            sample = noisy_latents
            model_output = student_model_pred
            # 1. get previous step value

            # 2. compute alphas, betas
            alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            # 3. Get scalings for boundary conditions
            c_skip, c_out = noise_scheduler.get_scalings_for_boundary_condition_discrete(timesteps)
            # 4. Different Parameterization:

            parameterization = noise_scheduler.config.prediction_type
            if parameterization == "epsilon":  # noise-prediction
                pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            elif parameterization == "sample":  # x-prediction
                pred_x0 = model_output
            elif parameterization == "v_prediction":  # v-prediction
                pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output

            #video_predict =
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
                torch.nn.utils.clip_grad_norm_(parameters_list,args.max_grad_norm)
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
    parser.add_argument("--use_8bit_adam",action="store_true",help="Whether or not to use 8-bit Adam from bitsandbytes.",)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--do_window_attention", action="store_true")
    parser.add_argument("--datavideo_size", type = int, default = 512)
    args = parser.parse_args()
    name = Path(args.config).stem
    main(args)
