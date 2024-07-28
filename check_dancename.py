import os
from collections import Counter
counter = Counter()
base_folder = r'D:\1.연구\[연구4] VideoDistillation\data\TikTok\archive\TikTok_Raw_Videos\TikTok_Raw_Videos'
folders = os.listdir(base_folder)
print(f'folders: {len(folders)}')
for folder in folders :
    folder_dir = os.path.join(base_folder, folder)
    dance_dir = os.path.join(folder_dir, 'dance_name.txt')
    try :
        with open(dance_dir, 'r') as f:
            lines = f.readlines()
        counter[lines[0].strip()] += 1
        print(f'folder: {folder}, dance: {lines[0].strip()}')
    except Exception as e :
        print(f'folder: {folder}, dance:')
        print(f'Error: {e}')

print(f'counter: {counter.most_common()}')
