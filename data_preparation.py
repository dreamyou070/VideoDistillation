import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import os
from attn.masactrl import MutualMotionAttentionControl
from attn.masactrl_utils import regiter_motion_attention_editor_diffusers
import time

def main(args):

    print(f' \n step 1. make Motion Base Pipeline with LCM Scheduler')
    from diffusers import LCMScheduler, AutoPipelineForText2Image
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                               motion_adapter=adapter,
                                               torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    print(f' (1.1) lcm lora')
    pipe.load_lora_weights("wangfuyun/AnimateLCM",
                           weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                           adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    print(f' (1.2) LCM Scheduler')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.to('cuda')

    print(f' \n step 2. save_base_dir')
    save_base_dir = f'/share0/dreamyou070/dreamyou070/MyData/video/webvid_genvideo/sample'
    os.makedirs(save_base_dir, exist_ok=True)

    print(f' \n step 3. inference test')
    prompt_folder_base = r'/share0/dreamyou070/dreamyou070/MyData/video/webvid_genvideo/start_txt_files'
    prompt_file_name = f'filtered_captions_train_{str(args.m).zfill(3)}.txt'
    prompt_file_dir = os.path.join(prompt_folder_base, prompt_file_name)
    with open(prompt_file_dir, 'r') as f:
        prompts = f.readlines()

    guidance_scale = 1.5
    num_inference_step = 6
    seed = 0
    n_prompt = "bad quality, worse quality, low resolution"
    contents = []
    for p, prompt in enumerate(prompts):
        save_idx = str(args.m * 1000 + p).zfill(6)
        output = pipe(prompt=prompt,
                      negative_prompt=n_prompt,
                      num_frames=16,
                      guidance_scale=guidance_scale,
                      num_inference_steps=num_inference_step,
                      generator=torch.Generator("cpu").manual_seed(seed),).frames[0]
        video_dir = os.path.join(save_base_dir, f'sample_{save_idx}.mp4')
        export_to_video(output, video_dir)
        elem = [f'sample_{save_idx}',f'sample_{save_idx}.mp4',prompt]
        contents.append(elem)

    # make csv file
    import csv
    header = ['videoid','page_dir','name']
    prompt_folder_base = f'/share0/dreamyou070/dreamyou070/MyData/video/webvid_genvideo/start_csv_file_{str(args.m).zfill(3)}.csv'
    with open(prompt_folder_base, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 헤더 작성
        writer.writerows(contents)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--is_teacher', action='store_true')
    parser.add_argument('--start_num', type=int, default=100)
    parser.add_argument('--end_num', type=int, default=140)
    from utils import arg_as_list

    parser.add_argument('--skip_layers', type=arg_as_list)
    args = parser.parse_args()
    main(args)