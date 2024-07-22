import os
import argparse
import cv2
import PIL
from PIL import Image

def main(args) :

    #base_dir = r'/share0/dreamyou070/dreamyou070/VideoDistillation_Script/experiment'
    base_dir = r'./experiment'
    target_folder = args.target_folder
    target_folder_dir = os.path.join(base_dir, target_folder)
    sample_folders = os.path.join(target_folder_dir, 'samples')
    files = os.listdir(sample_folders)
    for file in files :
        name, ext = os.path.splitext(file)
        if ext == '.txt' :
            text_dir = os.path.join(sample_folders, file)
            with open(text_dir, 'r') as f:
                lines = f.readlines()
            prompt = lines[0].strip().split(': ')[-1]
            gif_dir = os.path.join(sample_folders, name+'.gif')
            mp4_dir = os.path.join(sample_folders, name+'.mp4')
            # mp4 open

            cap = cv2.VideoCapture(mp4_dir)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            first_frame = cap.read()[1] # numpy, [512,512,3]
            # BGR to RGB
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            first_frame = Image.fromarray(first_frame) # pil

            # [1] text and image similarity
            from utils.eval import t2i_ClipSim
            t2i_ClipSim(first_frame,prompt)
            break




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, default=r'5_4_mid_up_02_distill_loss_1_feature_1_hps')
    args = parser.parse_args()
    main(args)