import csv, os

csv_folder = '/share0/dreamyou070/dreamyou070/MyData/video/webvid_genvideo/csv_folder'
files = os.listdir(csv_folder)
header = ['videoid','page_dir','name']
datas = []
for file in files :
    csv_path = os.path.join(csv_folder, file)
    with open(csv_path, 'r') as f :
        contents = f.readlines()
    for i, line in enumerate(contents):
        if i != 0 :
            elem = line.strip()
            if elem != '"' :
                video_id = elem.split(',')[0]
                dir = elem.split(',')[1]
                prompt = elem.split(',')[2:]
                prompt = ','.join(prompt)
                prompt = prompt.replace('"', '', 1).strip()
                elem = [video_id, dir, prompt]
                datas.append(elem)

make_csv = '/share0/dreamyou070/dreamyou070/MyData/video/webvid_genvideo/webvid_genvideo.csv'
with open(make_csv, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for content in datas :
        writer.writerow(content)
# check saved file
with open(make_csv, 'r') as csvfile:
    dataset = list(csv.DictReader(csvfile))