import os

data_dir = 'filtered_captions_val.txt'
with open(data_dir, 'r', encoding='utf-8') as f:
    data = f.readlines()
data_dir = 'filtered_captions_val.txt'
with open(data_dir, 'w', encoding='utf-8') as ff :
    for line in data :
        line_ = line.strip()
        if line_ != '':
            ff.write(line_ + '\n')
