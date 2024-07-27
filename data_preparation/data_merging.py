import os, csv

header = ['videoid','page_dir','name']
trg_folder = 'csv_folder'
files = os.listdir(trg_folder)
total_elems = []
for file in files:
    if file.endswith('.csv'):
        with open(os.path.join(trg_folder, file), 'r', encoding = 'utf-8') as f:
            contents = f.readlines()
        contents = contents[1:]
        for line in contents:
            line = line.strip()
            if len(line) > 5 :
                videoid, page_dir = line.split(',')[0], line.split(',')[1]
                name = line.split(',')[2:]
                name = ','.join(name)
                name = name.replace('"', '', 1)
                elem = [videoid, page_dir, name]
                total_elems.append(elem)

# make new csv file
new_csv_file = 'start_csv_file.csv'
with open(new_csv_file, 'w', encoding = 'utf-8',newline='', ) as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(total_elems)