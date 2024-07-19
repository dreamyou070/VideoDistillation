import os

def main() :

    # [1] read file
    target_file = 'videoscore.csv'
    with open(target_file, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip() # sample_id,visual quality,temporal consistency,dynamic degree,text-to-video alignment,factual consistency
    id_idx = 0
    dynamic_idx = 3

    for i, line in enumerate(lines) :
        if i != 0 :
            content = line.split(',')
            sample_id = content[id_idx]
            dynamic = content[dynamic_idx]

if __name__ == "__main__":
    main()