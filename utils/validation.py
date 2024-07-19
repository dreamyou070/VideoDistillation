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
def log_validation(
    vae,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    name="target",
    scheduler: str = "lcm",
    num_inference_steps: int = 4,
    add_to_trackers: bool = True,
    use_lora: bool = False,
    disc_gt_images: Optional[List] = None,
    guidance_scale: float = 1.0,
    spatial_head: Optional = None,
    logger_prefix: str = "",
):
    logger.info("Running validation... ")
    scheduler_additional_kwargs = {}
    if args.base_model_name == "animatediff":
        scheduler_additional_kwargs["beta_schedule"] = "linear"
        scheduler_additional_kwargs["clip_sample"] = False
        scheduler_additional_kwargs["timestep_spacing"] = "linspace"

    if scheduler == "lcm":
        # set beta_schedule="linear" according to https://huggingface.co/wangfuyun/AnimateLCM
        scheduler = LCMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    elif scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    elif scheduler == "euler":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    else:
        raise ValueError(f"Scheduler {scheduler} is not supported.")

    unet = deepcopy(accelerator.unwrap_model(unet))
    if args.base_model_name == "animatediff":
        pipeline_cls = AnimateDiffPipeline
    elif args.base_model_name == "modelscope":
        pipeline_cls = TextToVideoSDPipeline

    if use_lora:
        pipeline = pipeline_cls.from_pretrained(
            args.pretrained_teacher_model,
            vae=vae,
            scheduler=scheduler,
            revision=args.revision,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )
        lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
        pipeline.load_lora_weights(lora_state_dict)
        pipeline.fuse_lora()
    else:
        pipeline = pipeline_cls.from_pretrained(
            args.pretrained_teacher_model,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            revision=args.revision,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )

    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)

    if (
        args.enable_xformers_memory_efficient_attention
        and args.base_model_name != "animatediff"
    ):
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            logger.warning(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        "Cute small corgi sitting in a movie theater eating popcorn, unreal engine.",
        "A Pikachu with an angry expression and red eyes, with lightning around it, hyper realistic style.",
        "A dog is reading a thick book.",
        "Three cats having dinner at a table at new years eve, cinematic shot, 8k.",
        "An astronaut riding a pig, highly realistic dslr photo, cinematic shot.",
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        output = []
        with torch.autocast("cuda", dtype=weight_dtype):
            output = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=args.resolution,
                width=args.resolution,
                generator=generator,
                guidance_scale=guidance_scale,
                output_type="latent",
            ).frames
            if spatial_head is not None:
                output = spatial_head(output)

            output = pipeline.decode_latents(output)
            video = tensor2vid(output, pipeline.image_processor, output_type="np")
            # video should be a tensor of shape (t, h, w, 3), min 0, max 1
            video = video[0]

        save_dir = os.path.join(args.output_dir, "output", f"{name}-step-{step}")
        if accelerator.is_main_process:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        image_logs.append({"validation_prompt": prompt, "video": video})
        save_to_local(save_dir, prompt, video)

    if add_to_trackers:
        try:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for log in image_logs:
                        images = log["video"]
                        validation_prompt = (
                            f"{logger_prefix}{num_inference_steps} steps/"
                            + log["validation_prompt"]
                        )
                        formatted_images = []
                        for image in images:
                            formatted_images.append(np.asarray(image))

                        formatted_images = np.stack(formatted_images)

                        tracker.writer.add_images(
                            validation_prompt,
                            formatted_images,
                            step,
                            dataformats="NHWC",
                        )
                    if disc_gt_images is not None:
                        for i, image in enumerate(disc_gt_images):
                            tracker.writer.add_image(
                                f"discriminator gt image/{i}",
                                image,
                                step,
                                dataformats="HWC",
                            )
                elif tracker.name == "wandb":
                    # log image for comparison
                    formatted_images = []

                    for log in image_logs:
                        images = log["video"]
                        validation_prompt = log["validation_prompt"]
                        image = wandb.Image(images[0], caption=validation_prompt)
                        formatted_images.append(image)

                    if args.use_lora:
                        tracker.log(
                            {
                                f"{logger_prefix}validation image {num_inference_steps} steps": formatted_images
                            },
                            step=step,
                        )
                    else:
                        tracker.log(
                            {
                                f"{logger_prefix}validation image {num_inference_steps} steps/{name}": formatted_images
                            },
                            step=step,
                        )

                    # log video
                    formatted_video = []
                    for log in image_logs:
                        video = (log["video"] * 255).astype(np.uint8)
                        validation_prompt = log[
                            "validation_prompt"
                        ]  # wandb does not support video logging with caption
                        video = wandb.Video(
                            np.transpose(video, (0, 3, 1, 2)), fps=4, format="mp4"
                        )
                        formatted_video.append(video)

                    if args.use_lora:
                        tracker.log(
                            {
                                f"{logger_prefix}validation video {num_inference_steps} steps": formatted_video
                            },
                            step=step,
                        )
                    else:
                        tracker.log(
                            {
                                f"{logger_prefix}validation video {num_inference_steps} steps/{name}": formatted_video
                            },
                            step=step,
                        )
                    # log discriminator ground truth images
                    if disc_gt_images is not None:
                        formatted_disc_gt_images = []
                        for i, image in enumerate(disc_gt_images):
                            image = wandb.Image(
                                image, caption=f"discriminator gt image {i}"
                            )
                            formatted_disc_gt_images.append(image)
                        tracker.log(
                            {"discriminator gt images": formatted_disc_gt_images},
                            step=step,
                        )
                else:
                    logger.warning(f"image logging not implemented for {tracker.name}")
        except Exception as e:
            logger.error(f"Failed to log images: {e}")

    del pipeline
    del unet
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs
"""
##########################################################################################
    print(f' [inference condition] ')
    prompt = "A video of a woman, having a selfie"
    n_prompt = "bad quality, worse quality, low resolution"
    num_frames = 16
    guidance_scale = 1.5
    num_inference_steps = 6