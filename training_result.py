import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy


    
def gen_img(pkl_path, bounding_box, img_style, img_text, c):
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
    

