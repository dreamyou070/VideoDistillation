import os
import clip
import torch

base_file = 'filtered_captions_train_2.txt'
with open(base_file, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()

nsfw_nonnsfw_concepts = ["violent, death, blood, wounds, mutilation, injury, gore, graphic",
                         "nudity, naked, explicit, private parts, unclothed, bare, nude",
                         "pornography, explicit, sexual, adult, mature, x-rated, obscene",
                         "explicit, sexual, intercourse, graphic, adult, mature, obscene",
                         "child, minor, exploitation, inappropriate, sexual, abuse",
                         "solicitation, sexual, explicit, adult, services, proposition",
                         "violence, gore, violent, blood, wounds, injury, death",
                         "suicide, self-harm, self-injury, self-destructive, death, kill",
                         "harassment, bullying, cyberbullying, threat, intimidation, abuse",
                         "hate, discrimination, racism, bigotry, prejudice, intolerance",
                         "intolerance, discrimination, bigotry, prejudice, bias, hate",
                         "drugs, narcotics, controlled substances, illegal, abuse, misuse",
                         "alcohol, drinking, drunk, intoxication, abuse, underage",
                         "tobacco, smoking, cigarettes, nicotine, underage, addiction",
                         "weapons, guns, firearms, violence, illegal, dangerous",
                         "gambling, bet, wager, casino, risk, addiction",
                         "controversial, sensitive, divisive, polarizing, debate, conflict",]

# Features
device = 'cuda'
model, preprocess = clip.load("ViT-B/32", device=device)
# Tokenize
nsfw_concepts_text_tokens = clip.tokenize(nsfw_nonnsfw_concepts).to(device)
with torch.no_grad():
    nsfw_text_features = model.encode_text(nsfw_concepts_text_tokens)

global_num = 0
for line in lines :
    line = line.strip()
    if line != "" :
        # [1] prompt filtering
        prompt = line
        with torch.no_grad():
            prompt_tokens = clip.tokenize(prompt).to(device)
            prompt_features = model.encode_text(prompt_tokens)
            # [2] Normalize
            nsfw_text_features /= nsfw_text_features.norm(dim=-1, keepdim=True)
            prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
            # [3] similarity and results
            similarity = (100.0 * prompt_features @ nsfw_text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            if values[0].item() < 0.5 : # pass

                global_num += 1
                target_number = int(global_num // 100000) # 몫
                file_num = str(target_number).zfill(3)
                if target_number < 10 :
                    target_file = f'filtered_captions_train_2_{file_num}.txt'
                    with open(target_file, 'a', encoding = 'utf-8') as ff:
                        ff.write(line+'\n')
                    print(target_file)


