import os
import csv
base_folder = '/share0/dreamyou070/dreamyou070/MyData/video/webvid_genvideo/csv_folder'
files = os.listdir(base_folder)
header = ['videoid','page_dir','name']
contents = []
for file in files:
    if file.endswith('.csv'):
        with open(os.path.join(base_folder, file), 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines) :
            if i != 0 :
                elem = line.strip()
                if elem.startswith('prompt_'):
                    video_id = elem.split(',')[0]
                    dir = elem.split(',')[1]
                    prompt = elem.split(',')[2:]
                    prompt = ','.join(prompt)
                    prompt = prompt.replace('"', '', 1).strip()
                    elem = [video_id, dir, prompt]
                    contents.append(elem)
                #elem = elem.split(',')
                #contents.append(elem)
print(f'contents = {contents}')
# make final csv file
final_csv = '/share0/dreamyou070/dreamyou070/MyData/video/webvid_genvideo.csv'
with open(final_csv, 'w', encoding = 'utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for content in contents:
        writer.writerow(content)