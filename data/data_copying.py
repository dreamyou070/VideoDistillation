import os
# 0 - 199

import os
import glob
def main() :

    # [1] make final csv file
    final_csv_file = "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-csv/0_200_sj.csv"
    # file name, base_dir, prompt
    header = ['videoid', 'page_dir', 'name']

    base_folder = r'/share0/MIR_LAB/jerry0110/samsung_video_gen/general_prompt_test_gpt_num_frames_16/'
    folders = os.listdir(base_folder)
    for folder in folders:
        folder_dir = os.path.join(base_folder, folder)
        original_dir = os.path.join(folder_dir, 'origin')
        prompt_folders = os.listdir(original_dir)
        for prompt_folder in prompt_folders:
            prompt_folder_dir = os.path.join(original_dir, prompt_folder)
            print(f'prompt_folder_dir = {prompt_folder_dir}')
            mp4_dir = glob.glob(os.path.join(prompt_folder_dir, '*.gif'))

            page_dir = os.path.split(mp4_dir)[:-1]
            videoid = os.path.split(mp4_dir)[-1].split('.')[0]
            print(f'page_dir = {page_dir}')
            print(f'videoid = {videoid}')
            """
            # [2] read previous csv files
            csv_base_folder = r'/share0/dreamyou070/dreamyou070/OneStepVideo/experiment_20240710_jpg_gif_mp4'
            csv_files = glob.glob(os.path.join(csv_base_folder, '*.csv'))
            csv_files.sort()
            total_csv_elem = []
            for i, csv_file in enumerate(csv_files) :
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                if i == 0 :
                    header = lines[0].strip()
                    header = header.split(',')
        
                contents = lines[-1].strip()
                contents = contents.split(',')
                #print(f'type of contents: {contents}')
                total_csv_elem.append(contents)
            # [e1,e2]
            import csv
            with open(final_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)  # 헤더 작성
                writer.writerows(total_csv_elem)  # 데이터 작성
            """


if __name__ == "__main__" :
    main()




