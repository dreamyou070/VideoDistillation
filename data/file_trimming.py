import os

save_base_folder = r'C:\Users\hpuser\2_1_samples_trim'
os.makedirs(save_base_folder, exist_ok=True)
base_folder = r'C:\Users\hpuser\2_1_samples'
files = os.listdir(base_folder)
for file in files :
    name, ext = os.path.splitext(file)
    epoch = name.split('_')[-2]
    epoch_folder = os.path.join(save_base_folder, f'epoch_{epoch}')
    os.makedirs(epoch_folder, exist_ok=True)
    org_file = os.path.join(base_folder, file)
    save_file = os.path.join(epoch_folder, file)
    os.rename(org_file, save_file)