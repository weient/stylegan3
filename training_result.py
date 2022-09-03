import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import os, sys
import cv2

def img_to_np(img_path):
    img = []
    im = PIL.Image.open(img_path).convert("RGB")
    im = np.array(im)
    im = im.transpose(2, 0, 1)
    img.append(im)
    img = np.array(img)
    return img
    
def gen_img(pkl_path, bounding_box, img_style, img_text, c = None):
    bounding_box = torch.Tensor([bounding_box])
    device = torch.device('cuda')
    with dnnlib.util.open_url(pkl_path) as f:
        G_ema = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        G = legacy.load_network_pkl(f)['G'].to(device)
    img = G(bounding_box, img_style, img_text, c)
    img_ema = G_ema(bounding_box, img_style, img_text, c)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_ema = (img_ema.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'G_out.png')
    PIL.Image.fromarray(img_ema[0].cpu().numpy(), 'RGB').save(f'G_ema_out.png')
    

