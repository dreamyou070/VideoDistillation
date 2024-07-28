import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_video
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(img):
    #img = np.array(Image.open(imfile)).astype(np.uint8) # np
    #img = torch.from_numpy(img).permute(2, 0, 1).float() # torch
    img = img.float() # torch, channel, h, w
    return img[None].to(DEVICE)



def raft_score(video_path):

    # [1] argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",
                        default='/share0/dreamyou070/dreamyou070/EvalCrafter/RAFT/models/raft-things.pth')
    parser.add_argument('--path',
                        help="dataset for evaluation")  # images
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images, _, _ = read_video(str(video_path), output_format="TCHW")  # 333 video length
        optical_flows = []
        for image1, image2 in zip(images[:-1], images[1:]):
            # frame wise
            image1 = load_image(image1)
            image2 = load_image(image2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
            mean_optical_flow = flow_magnitude.mean().item()
            optical_flows.append(mean_optical_flow)
        mean_optical_flow_video = np.mean(optical_flows)
    return mean_optical_flow_video

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation") # images
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
"""