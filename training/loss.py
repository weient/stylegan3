# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""
import string
import argparse
import numpy as np
import torch
from ..torch_utils import training_stats
from ..torch_utils.ops import conv2d_gradfix
from ..torch_utils.ops import upfirdn2d
from ..OCR.demo import *

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from IPython.display import Image
import sys
from torch.nn.functional import cross_entropy
from .typeloss_test import *
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, bounding_box, phase, real_img, real_img_rec, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

def to_onehot(input, max_length):
    characters = string.printable  # All printable ASCII characters.
    token_index = dict(zip(characters, range(1, len(characters) + 1)))
    results = np.zeros((len(input), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(input):
        for j, character in enumerate(sample[:max_length]):
            index = token_index.get(character)
            results[i, j, index] = 1.
    return results
def call_OCR(img_tensor, batch_size, word_label):
    dic = {"image_folder":img_tensor, "workers":1, "batch_size": batch_size, "saved_model":'/content/drive/Shareddrives/styleGAN3/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth', "batch_max_length":25,
    "imgH":32, "imgW":100, "rgb":False, "character":string.printable[:-6],
    "sensitive":True, "PAD":False, "Transformation":'TPS', "FeatureExtraction":'ResNet', 
    "SequenceModeling":'BiLSTM', "Prediction":'Attn', "num_fiducial":20, "input_channel":1, 
    "output_channel": 512, "hidden_size":256, "num_gpu":torch.cuda.device_count()}
    opt = argparse.Namespace(**dic)
    str_list = demo(opt)
    #print("OCR str_list: ", str_list)
    input_l = []
    target_l = []
    max_length = 15
    for i, j in enumerate(str_list):
        input = torch.from_numpy(to_onehot([str_list[i]], max_length))
        target = torch.from_numpy(to_onehot([word_label[i]], max_length))
        input_l.append(input)
        target_l.append(target)
    input = torch.cat(input_l, 0)
    target = torch.cat(target_l, 0)
    #print("input:", input.size(), "target: ", target.size())
    return cross_entropy(input, target)


def paste_img(tensor_square, tensor_gen, batch_size):
    transform = T.ToPILImage()
    transform_back = T.ToTensor()
    cyclic_list = []
    for i in range(batch_size):
        img_square = transform(tensor_square[i])
        img_gen = transform(tensor_gen[i])
        new_square = img_square.copy()
        new_square.paste(img_gen, (0, 96))
        new_square_tensor = transform_back(new_square)
        cyclic_list.append(torch.unsqueeze(new_square_tensor, 0))
    cyclic_tensor = torch.cat(cyclic_list, 0)
    return cyclic_tensor

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

    def run_G(self, bounding_box, img_style, img_text, c, update_emas=False):
        print("Running Generator\n")
        style_out = self.G.style_encoder(img_style, bounding_box)
        content_out = self.G.content_encoder(img_text)
        ws = self.G.mapping(style_out, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            print("in loss.py line 45, style_mixing_prob > 0 !!!")
        #    with torch.autograd.profiler.record_function('style_mixing'):
        #        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #        cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #        ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img, Mask = self.G.synthesis(content_out, ws, update_emas=update_emas)
        return img, ws, Mask

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        print("Running Discriminator\n")
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, bounding_box, phase, real_img, real_img_rec, real_text, word_label, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        #print("word_label: ", word_label)
        # Gmain: Maximize logits for generated images.
        
        dif_text = torch.flip(real_text, [0])
        dif_word_label = list(reversed(word_label))
        print(dif_word_label)
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_Mask = self.run_G(bounding_box, real_img, real_text, gen_c)
                gen_Mask = torch.cat((gen_Mask, gen_Mask, gen_Mask), 1)
                cyc_img = paste_img(real_img, gen_img, real_img.shape[0])
                cyc_img = cyc_img.to(self.device)
                gen_img_2, _gen_ws_2, gen_Mask_2 = self.run_G(bounding_box, cyc_img, real_text, gen_c)
                gen_img_dif, _, gen_Mask_dif = self.run_G(bounding_box, real_img, dif_text, gen_c)
                gen_Mask_dif = torch.cat((gen_Mask_dif, gen_Mask_dif, gen_Mask_dif), 1)
                loss_R_dif = call_OCR(gen_Mask_dif, real_img.shape[0], dif_word_label)
                loss_cyc = torch.nn.functional.l1_loss(gen_img_2, real_img_rec)
                loss_rec = torch.nn.functional.l1_loss(gen_img, real_img_rec)
                loss_R = call_OCR(gen_Mask, real_img.shape[0], word_label)
                loss_type = call_type(gen_img, real_img_rec)
                final_R = (loss_R + loss_R_dif) / 2
                print("loss_type: ", loss_type)
                print("loss_cyc: ", loss_cyc)
                print("loss_rec: ", loss_rec)
                print("loss_R: ", final_R)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_D = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain = 10*loss_rec + loss_cyc + final_R + loss_type + loss_D
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = real_img.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, gen_Mask = self.run_G(bounding_box[:batch_size], real_img[:batch_size], real_text[:batch_size], gen_c)
                '''
                gen_Mask = torch.cat((gen_Mask, gen_Mask, gen_Mask), 1)
                cyc_img = paste_img(real_img[:batch_size], gen_img, batch_size)
                cyc_img = cyc_img.to(self.device)
                gen_img_2, _gen_ws_2, gen_Mask_2 = self.run_G(bounding_box[:batch_size], cyc_img, real_text[:batch_size], gen_c)
                loss_cyc = torch.nn.functional.l1_loss(gen_img_2, real_img_rec[:batch_size])
                loss_R = call_OCR(gen_Mask, batch_size, word_label)
                loss_rec = torch.nn.functional.l1_loss(gen_img, real_img_rec[:batch_size])
                loss_type = call_type(gen_img, real_img_rec[:batch_size])
                print("loss_cyc: ", loss_cyc)
                print("loss_rec: ", loss_rec)
                print("loss_R: ", loss_R)
                print("loss_type: ", loss_type)
                '''
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_Mask = self.run_G(bounding_box, real_img, real_text, gen_c, update_emas=True)
                #call_OCR(gen_img, real_img.shape[0])
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img_rec.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                #real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
