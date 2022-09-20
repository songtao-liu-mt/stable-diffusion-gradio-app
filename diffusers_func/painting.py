from pickle import FALSE
from tkinter.tix import IMAGE
from torch import autocast
from torchvision.utils import make_grid
from einops import rearrange
import torch
import random
import numpy as np
from PIL import Image, ImageFilter
import PIL
import cv2
import math
import gc
from diffusers_utils import paint_pipeline, image2image_pipeline, text2image_pipeline, device
import gradio as gr

class GaussianBlur(ImageFilter.Filter):
    
    def __init__(self, radius=100, bounds=None):
        self.radius = radius
        self.bounds = bounds
    
    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)
        
def image_grid(imgs, num_images):
    w, h = imgs[0].size
    rows = math.floor(num_images ** 0.5)
    cols = rows
    grids = Image.new('RGB', size=(cols * w, rows*h))
    grid_w, grid_h = grids.size
    for i, img in enumerate(imgs):
        grids.paste(img, box=(i % cols * w, i // cols * h))
    return grids

def inpaint_predict(dict, prompt, width, height, steps=50, scale=7.5, strength=0.8, seed=42, num_images=4):
    gc.collect()
    torch.cuda.empty_cache()
    button_update = gr.Button.update(visible=True)
    init_img = dict['image'].convert("RGB").resize((width, height))
    mask_img = dict['mask'].convert("RGB").resize((width, height))
    result_images = []
    with autocast("cuda"):
        for i in range(num_images):
            generator = torch.Generator("cuda").manual_seed(seed + i)
            images = paint_pipeline(prompt=prompt, init_image=init_img, mask_image=mask_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["sample"]
            result_images.append(images[0])
    
    grids = image_grid(result_images, num_images)

    return grids, result_images, button_update


def outpaint_predict(image, prompt, direction, expand_lenth, steps=50, scale=7.5, strength=0.8, seed=2022):
    gc.collect()
    torch.cuda.empty_cache()
    # image = image.resize((512, 512))
    button_update_1 = gr.Button.update(value='Re-Run Generation')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)
    w, h = image.size
    max_side = 1024
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    assert expand_lenth % 64 == 0
    if direction == 'left' or direction == 'right':
        w_expanded = w +  expand_lenth
        h_expanded = h
    else:
        w_expanded = w
        h_expanded = h + expand_lenth
    if w_expanded > max_side:
        expand_lenth = max_side - w
        w_expanded = max_side
    if h_expanded > max_side:
        expand_lenth = max_side - h
        h_expanded = max_side
    if h_expanded == h and w_expanded == w:
        result_image = image
        return result_image, button_update_1, button_update_2, button_update_3
    
    init_img = image.convert("RGB")
    init_img = np.array(init_img)
    if direction == 'top':
        init_img = cv2.copyMakeBorder(init_img, 
                                    expand_lenth, 
                                    0,
                                    0,
                                    0,
                                    cv2.BORDER_DEFAULT)
        left_upper = (0, expand_lenth)
    elif direction == 'bottom':
        init_img = cv2.copyMakeBorder(init_img, 
                                    0, 
                                    expand_lenth,
                                    0,
                                    0,
                                    cv2.BORDER_DEFAULT)
        left_upper = (0, 0)
    elif direction == 'left':
        init_img = cv2.copyMakeBorder(init_img, 
                                    0, 
                                    0,
                                    expand_lenth,
                                    0,
                                    cv2.BORDER_DEFAULT)
        left_upper = (expand_lenth, 0)
    elif direction == 'right':
        init_img = cv2.copyMakeBorder(init_img, 
                                    0, 
                                    0,
                                    0,
                                    expand_lenth,
                                    cv2.BORDER_DEFAULT)
        left_upper = (0, 0)
        
    init_img = Image.fromarray(init_img)
    # create mask
    mask_img = Image.new(mode="RGB", size=(w_expanded, h_expanded), color="white")
    invert_mask = Image.new(mode="RGB", size=(w, h), color="black")
    mask_img.paste(im=invert_mask, box=left_upper)
    # mask_img = mask_img.filter(GaussianBlur(radius=1))
    
    with autocast("cuda"):
        generator = torch.Generator("cuda").manual_seed(seed)
        images = paint_pipeline(prompt=prompt, init_image=init_img, mask_image=mask_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["sample"]
        result_image = images[0]
    

    return result_image, button_update_1, button_update_2, button_update_3

def multi2image_predict(init_img, prompt, width, height, steps=50, scale=7.5, strength=0.8, seed=2022, num_images=9):
    gc.collect()
    torch.cuda.empty_cache()
    button_update_1 = gr.Button.update(value='Re-Run Generation')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)
    result_images = []
    if init_img is not None:
        init_img = init_img.convert("RGB").resize((width, height))
        
        with autocast("cuda"):
            for i in range(num_images):
                generator = torch.Generator("cuda").manual_seed(seed + i)
                images = image2image_pipeline(prompt=prompt, init_image=init_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["sample"]
                result_images.append(images[0])
    else:
        with autocast("cuda"):
            for i in range(num_images):
                generator = torch.Generator("cuda").manual_seed(seed + i)
                images = text2image_pipeline(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["sample"]
                result_images.append(images[0])
    
    grids = image_grid(result_images, num_images)

    return grids, result_images, button_update_1, button_update_2, button_update_3

def text2image_predict(prompt, fixed_size=512, num_images=9, seed=2022):
    gc.collect()
    torch.cuda.empty_cache()
    result_images = []
    with autocast("cuda"):
        for i in range(num_images):
            generator = torch.Generator("cuda").manual_seed(seed + i)
            images = text2image_pipeline(prompt=prompt, height=fixed_size, width=fixed_size, generator=generator)["sample"]
            result_images.append(images[0])
            
    grids = image_grid(result_images, num_images)

    return grids, result_images

def variance_predict(prompt, width, height, steps=50, scale=7.5, strength=0.8, seed=2022, num_images=9):
    gc.collect()
    torch.cuda.empty_cache()
    button_update = gr.Button.update(visible=True)
    result_images = []
    with autocast("cuda"):
        init_imgs = text2image_pipeline(prompt=prompt, height=height, width=width)["sample"]
        init_img = init_imgs[0]
        result_images.append(init_img)
        for i in range(num_images - 1):
            generator = torch.Generator("cuda").manual_seed(seed + i)
            images = image2image_pipeline(prompt=prompt, init_image=init_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["sample"]

            result_images.append(images[0])
            
    grids = image_grid(result_images, num_images)

    return grids, result_images, button_update
