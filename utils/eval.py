import t2v_metrics
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
"""
def t2i_score(img_dir, text) :
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
    image = img_dir
    score = clip_flant5_score(images=[image], texts=[text])
    return score

def t2v_score_llava(images ) :
    llava_score = t2v_metrics.VQAScore(model='llava-v1.5-13b')
    score = llava_score(images,texts)
    return score


instructblip_score = t2v_metrics.VQAScore(model='instructblip-flant5-xxl')
clip_score = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336')
blip_itm_score = t2v_metrics.ITMScore(model='blip2-itm')
pick_score = t2v_metrics.CLIPScore(model='pickscore-v1')
hpsv2_score = t2v_metrics.CLIPScore(model='hpsv2')
image_reward_score = t2v_metrics.ITMScore(model='image-reward-v1')
"""
# clip-flant5-xxl

def t2i_ClipSim(image, text : str) :
    # ViT-H/14 LAION-2B
    #

    #model, preprocess = create_model_from_pretrained('ViT-H-14-CLIPA')
    #tokenizer = get_tokenizer('hf-hub:ViT-H-14-CLIPA')
    model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
    tokenizer = get_tokenizer('ViT-H-14')

    # [1] processing img
    image = preprocess(image).unsqueeze(0)
    text = tokenizer([text], context_length=model.context_length)
    # [2]
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        print(f'image_features = {image_features.shape}')
        print(f'text_features = {text_features.shape}')

        #image_features = F.normalize(image_features, dim=-1)
        #text_features = F.normalize(text_features, dim=-1)
    return sim


