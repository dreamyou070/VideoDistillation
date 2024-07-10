import csv
csv_path = '0_100.csv'
with open(csv_path, 'r') as csvfile:
    dataset = list(csv.DictReader(csvfile))
#  'videoid': 'prompt_64_seed_0',
#  'page_dir': 'prompt_64_seed_0.gif',
#  'name': 'Thai boxing training young european man in the gym. 4k', None: [' slow motion', ' face', ' close-up.']}
#print(dataset)
#  {'videoid': 'prompt_64_seed_0', 'page_d
# ir': 'prompt_64_seed_0.gif', 'name': 'Thai boxing training young european man in the gym. 4k', None: [' slow motion', ' face', ' close-up.']},

csv_dir = '0_100_1.csv'
elems = []
elem = ['videoid','page_dir','name']
elems.append(elem)

for k, elem in enumerate(dataset) :
    name = elem['name']
    key_list = list(elem.keys())
    if len(key_list) > 3 :
        name = elem['name']
        none_list = elem[None]
        print(f'none_list = {none_list}')
        non_str = ','.join(none_list)
        print(f'non_str = {non_str}')
        name = name + non_str
        elem['name'] = name
    elem = [elem['videoid'], elem['page_dir'], elem['name']]
    #print(f'elem = {elem}')
    elems.append(elem)

"""
# write csv

print(f'len of elems : {len(elems)}')
with open(csv_dir, 'w') as f:
    for elem in elems:
        f.write(','.join(elem) + '\n')
"""