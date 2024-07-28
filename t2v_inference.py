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


def main(args) :

    print(f' \n step 1. make Motion Base Pipeline with LCM Scheduler')
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    print(f'\n step 2. LCM Lora')
    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    """
    # ---------------------------------------------------------------------------------------------------- #
    # using motion lora
    motion_lora_dir = r'/share0/dreamyou070/dreamyou070/SD/pretrained/Animatediff/motion_lora/motion_lora_zom_in.safetensors'
    # safe open
    from safetensors import safe_open
    motion_lora_tensors = {}
    with safe_open(motion_lora_dir, framework="pt", device=0) as f:
        for k in f.keys():
            print(f'motion lora, key : {k}')
            motion_lora_tensors[k] = f.get_tensor(k)
    
    # applying motion lora
    unet = pipe.unet
    unet.load_state_dict(motion_lora_tensors)
    #pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name=motion_lora_dir, adapter_name="motion-lora")
    unet = pipe.unet
    """
    pipe.enable_vae_slicing()
    pipe.to('cuda')

    print(f' \n step 2. save_base_dir')
    num_frames = 16
    save_base_dir = f'../MyData/video/webvid_genvideo'
    os.makedirs(save_base_dir, exist_ok=True)
    sample_folder = os.path.join(save_base_dir, 'sample')
    os.makedirs(sample_folder, exist_ok=True)

    print(f' \n step 3. inference test')
    prompt_dir = f'./configs/prompts/filtered_captions_val_7.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()
    guidance_scale = 1.5
    inference_scale = 6
    n_prompt = "bad quality, worse quality, low resolution"
    seed = 0
    elems = []
    header = ['videoid','page_dir','name']
    # --m = 0
    for p, prompt in enumerate(prompts):
        save_p = str(p).zfill(7)
        #base_folder = os.path.join(prompt_folder, f'guidance_{guidance_scale}_inference_{inference_scale}')
        #os.makedirs(base_folder, exist_ok=True)
        #motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scales[0],
        #                                                 frame_num=16,
        #                                                 full_attention=True,
        #                                                 window_attention=False,
        #                                                 window_size=16,
        #                                                 total_frame_num=16,
        #                                                 is_teacher = args.is_teacher,
        #                                                 skip_layers=[])  # 32
        #regiter_motion_attention_editor_diffusers(unet, motion_controller)

        #pipe.unet = unet
        #start_time = time.time()
        # seed setting
        output = pipe(prompt=prompt,
                      negative_prompt=n_prompt,
                      num_frames=num_frames,
                      guidance_scale=guidance_scale,
                      num_inference_steps=inference_scale,
                      generator=torch.Generator("cpu").manual_seed(seed), )
        #end_time = time.time()
        #elapse_time = end_time - start_time
        frames = output.frames[0]
        
        #save_folder = os.path.join(base_folder, f'origin')
        #save_folder = os.path.join(base_folder, f'origin_elapse_time_{elapse_time}')
        #os.makedirs(save_folder, exist_ok=True)
        # [1] frame image save
        #for frame_idx, img in enumerate(frames) :
        #    save_frame_idx = str(frame_idx).zfill(2)
        #    img.save(os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}_frame_idx_{save_frame_idx}.jpg'))
        

        #export_to_gif(frames, os.path.join(sample_folder, f'prompt_{save_p}.gif'))
        export_to_video(frames, os.path.join(sample_folder, f'prompt_sample_{save_p}.mp4'))
        elem = [f'prompt_sample_{save_p}',
                os.path.join(sample_folder, f'prompt_sample_{save_p}.mp4'),
                prompt ]
        elems.append(elem)
    csv_file = os.path.join(save_base_dir, f'webvid_genvideo_2_7.csv')
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(header)
        # write content
        writer.writerows(elems)

        """
        
        # text recording
        with open(os.path.join(save_folder, 'elapse_time.txt'), 'w') as f :
            f.write(f'elapse_time = {elapse_time}')

        for i, skip_layer in layer_dict.items() :
            print(f' ** {skip_layer} Test ** ')
            motion_controller = MutualMotionAttentionControl(guidance_scale=guidance_scales[0],
                                                             frame_num=16,
                                                             full_attention=True,
                                                             window_attention=False,
                                                             window_size=16,
                                                             total_frame_num=16,
                                                             is_teacher = args.is_teacher,
                                                             is_eval = False,
                                                             skip_layers=[skip_layer])  # 32
            regiter_motion_attention_editor_diffusers(unet, motion_controller)
            pipe.unet = unet
            output = pipe(prompt=prompt,
                          negative_prompt="bad quality, worse quality, low resolution",
                          num_frames=num_frames,
                          guidance_scale=guidance_scale,
                          num_inference_steps=inference_scale,
                          generator=torch.Generator("cpu").manual_seed(seed), )
            
            frames = output.frames[0]
            save_folder = os.path.join(base_folder, f'{skip_layer}')
            os.makedirs(save_folder, exist_ok=True)
            #save_name = f'prompt_{save_p}_seed_{seed}.mp4'
            ##export_to_gif(frames, os.path.join(save_folder, save_name))
            #export_to_video(frames, os.path.join(save_folder, save_name))
            # [1] frame image save
            for frame_idx, img in enumerate(frames):
                save_frame_idx = str(frame_idx).zfill(2)
                img.save(
                    os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}_frame_idx_{save_frame_idx}.jpg'))
            export_to_gif(frames, os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}.gif'))
            export_to_video(frames, os.path.join(save_folder, f'prompt_{save_p}_seed_{seed}.mp4'))
        """
if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--m', type=int,default=1)
    parser.add_argument('--is_teacher', action='store_true')
    parser.add_argument('--start_num', type=int, default=100)
    parser.add_argument('--end_num', type=int, default=140)
    args = parser.parse_args()
    main(args)