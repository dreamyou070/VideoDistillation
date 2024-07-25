import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import os
from attn.masactrl import MutualMotionAttentionControl
from attn.masactrl_utils import regiter_motion_attention_editor_diffusers
import time

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


def main(args) :

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
    save_base_dir = f'experiment'
    os.makedirs(save_base_dir, exist_ok=True)
    save_folder = os.path.join(save_base_dir, f't2i_t2v_comparison_man')
    os.makedirs(save_folder, exist_ok=True)

    print(f' \n step 3. inference test')
    guidance_scales = [1.5]
    num_inference_steps = [1,2,3,4,5,6,7,8,9,10, 20,30,50]
    prompts = ["a man is walking on the street",]
    n_prompt = "bad quality, worse quality, low resolution"
    seeds = [0]
    for p, prompt in enumerate(prompts):
        for guidance_scale in guidance_scales :
            for inference_scale in num_inference_steps :
                for seed in seeds :
                    image = pipe(prompt=prompt,
                                 num_inference_steps=int(inference_scale),
                                 guidance_scale=0).images[0]
                    image_dir = os.path.join(save_folder, f'inference_step_{inference_scale}.png')
                    image.save(image_dir)
                    test_dir = os.path.join(save_folder, f'inference_step_{inference_scale}.txt')
                    with open(test_dir, 'w') as f :
                        f.write(f'prompt : {prompt}\n')
                        f.write(f'guidance_scale : {guidance_scale}\n')
                        f.write(f'inference_scale : {inference_scale}\n')
                        f.write(f'seed : {seed}\n')
    # Video Pipeline
    print(f' \n step 4. make Video Pipeline')
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism",
                                               motion_adapter=adapter,
                                               torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    print(f' (4.1) lcm lora')
    pipe.load_lora_weights("wangfuyun/AnimateLCM",
                           weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                           adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    print(f' (4.2) LCM Scheduler')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.to('cuda')
    print(f' \n step 3. inference test')
    for p, prompt in enumerate(prompts):
        for guidance_scale in guidance_scales:
            for inference_scale in num_inference_steps:
                for seed in seeds:
                    output = pipe(prompt=prompt,
                                  negative_prompt=n_prompt,
                                  num_frames=16,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=inference_scale,
                                  generator=torch.Generator("cpu").manual_seed(seed),).frames[0]

                    video_dir = os.path.join(save_folder, f'inference_step_{inference_scale}_video.mp4')
                    export_to_video(output, video_dir)

                    text_dir = os.path.join(save_folder, f'inference_step_{inference_scale}_video.txt')
                    with open(text_dir, 'w') as f :
                        f.write(f'prompt : {prompt}\n')
                        f.write(f'guidance_scale : {guidance_scale}\n')
                        f.write(f'inference_scale : {inference_scale}\n')
                        f.write(f'frame_num : 16\n')
                        f.write(f'seed : {seed}\n')



if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--m', type=int,default=1)
    parser.add_argument('--is_teacher', action='store_true')
    parser.add_argument('--start_num', type=int, default=100)
    parser.add_argument('--end_num', type=int, default=140)
    args = parser.parse_args()
    main(args)