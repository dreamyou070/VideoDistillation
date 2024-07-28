import os
import cv2
import os
from PIL import Image
import numpy as np

base_folder = r'TikTok_dataset/TikTok_dataset'
folders = os.listdir(base_folder)
save_folder = r'TikTok_dataset/TikTok_dataset_mp4_2'
os.makedirs(save_folder, exist_ok=True)
for folder in folders :

    folder_dir = os.path.join(base_folder, folder)
    image_folder = os.path.join(folder_dir, 'images')
    images = os.listdir(image_folder)
    num_image = len(images)

    for start_idx in range(num_image // 16) :

        video_name = f'{str(start_idx).zfill(4)}.mp4'
        image_list = []
        start_num = start_idx * 16 + 1
        end_num = start_num + 32 + 1
        end_image_name = str(end_num).zfill(4) + '.png'
        if end_image_name in images :
            for m in range(start_num, end_num) :
                image_name = str(m).zfill(4) + '.png'
                image_list.append(os.path.join(image_folder, image_name))
            # ----------------------------------------------------
            # [1] first frame
            first_frame_dir = image_list[0]
            first_frame = cv2.imread(first_frame_dir)
            height, width, layers = first_frame.shape
            # ----------------------------------------------------
            # [2] make video
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            save_video_name = os.path.join(save_folder, f'folder_{folder}_{str(start_idx).zfill(6)}.mp4')
            fps = 20
            video_writer = cv2.VideoWriter(save_video_name, fourcc, fps=fps, frameSize=(width, height))
            video_frames = []
            for img_dir in image_list :
                print(f'img_dir = {img_dir}')
                pil_img = Image.open(img_dir)
                video_frames.append(np.array(pil_img))
            for i in range(len(video_frames)):
                img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
                video_writer.write(img)
