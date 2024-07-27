import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import os
from attn.masactrl import MutualMotionAttentionControl
from attn.masactrl_utils import regiter_motion_attention_editor_diffusers
import time
import csv

layer_dict = {0: 'down_blocks_0_motion_modules_0',
              1: 'down_blocks_0_motion_modules_1',
              2: 'down_blocks_1_motion_modules_0',
              3: 'down_blocks_1_motion_modules_1',
              4: 'down_blocks_2_motion_modules_0',
              5: 'down_blocks_2_motion_modules_1',
              6: 'down_blocks_3_motion_modules_0',
              7: 'down_blocks_3_motion_modules_1',
              8: 'mid_block_motion_modules_0',
              9: 'up_blocks_0_motion_modules_0',
              10: 'up_blocks_0_motion_modules_1',
              11: 'up_blocks_0_motion_modules_2',
              12: 'up_blocks_1_motion_modules_0',
              13: 'up_blocks_1_motion_modules_1',
              14: 'up_blocks_1_motion_modules_2',
              15: 'up_blocks_2_motion_modules_0',
              16: 'up_blocks_2_motion_modules_1',
              17: 'up_blocks_2_motion_modules_2',
              18: 'up_blocks_3_motion_modules_0',
              19: 'up_blocks_3_motion_modules_1',
              20: 'up_blocks_3_motion_modules_2'}


def main(args):

    print(f' \n step 1. make Motion Base Pipeline with LCM Scheduler')
    from diffusers import LCMScheduler, AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained("emilianJR/epiCRealism",
                                                     torch_dtype=torch.float16)
    print(f' (1.1) LCM Scheduler')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    print(f' (1.2) LCM Lora')
    pipe.load_lora_weights("wangfuyun/AnimateLCM",
                           weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                           adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    pipe.enable_vae_slicing()
    pipe.to('cuda')

    print(f' \n step 2. save_base_dir')
    save_base_dir = f'../MyData/video/webvid_genvimage'
    os.makedirs(save_base_dir, exist_ok=True)
    sample_folder = os.path.join(save_base_dir, 'sample')
    os.makedirs(sample_folder, exist_ok=True)
    print(f' ----------------- sample_folder : {sample_folder} ----------------- ')

    print(f' \n step 3. inference test')
    prompt_dir = f'./configs/prompts/filtered_captions_val_{args.m}.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()
    guidance_scale = 1.5
    inference_scale = 6
    n_prompt = "bad quality, worse quality, low resolution"
    seed = 0
    elems = []
    header = ['videoid', 'page_dir', 'name']
    # --m = 0
    for p, prompt in enumerate(prompts):
        save_p = str((args.m) * 200 + p).zfill(3)

        image = pipe(prompt=prompt,
                     num_inference_steps=int(inference_scale),
                     guidance_scale=0).images[0]
        image_dir = os.path.join(sample_folder, f'prompt_{save_p}.png')
        image.save(image_dir)
        elem = [f'prompt_{save_p}', os.path.join(sample_folder, f'prompt_{save_p}.png'), prompt]
        elems.append(elem)

    csv_file = os.path.join(save_base_dir, f'webvid_genvideo_{args.m}.csv')
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(header)
        # write content
        writer.writerows(elems)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--is_teacher', action='store_true')
    parser.add_argument('--start_num', type=int, default=100)
    parser.add_argument('--end_num', type=int, default=140)
    args = parser.parse_args()
    main(args)