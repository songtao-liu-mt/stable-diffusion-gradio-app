from email import generator
from pickle import FALSE
from tkinter.tix import IMAGE
from torch import autocast
from einops import rearrange
import torch
import random
import numpy as np
from PIL import Image
import PIL
import cv2
import math
import gc
from diffusers_utils import paint_pipeline, image2image_pipeline, text2image_pipeline, translator_zh2en, deal_width_exceed_maxside, image_grid, GaussianBlur, device, MAX_SIDE, ChineseFilter, EnglishFilter, logo_image_pil, forbidden_pil, is_contain_chinese
import gradio as gr
import logging
import re
from io import BytesIO
import base64


def generate_empty_content_image():
    return forbidden_pil
    
def check_prompt_to_empty_image(language_filter, prompt):
    filter_dict = language_filter.filter(prompt)
    if '1' in filter_dict:
        logging.warn(filter_dict)
        return generate_empty_content_image()
    else:
        logging.info(filter_dict)
        return None
    
def generate_images_with_logo(original_images):
    w, h = original_images[0].size
    
    logo_w, logo_h = logo_image_pil.size
    ratio = w / logo_w
    h_resized = int(logo_h * ratio)
    logo_resized = logo_image_pil.convert("RGB").resize((w, h_resized))
    
    logo_images = []
    for ori in original_images:
        canvas_image = Image.new(mode="RGB", size=(w, h + h_resized), color="black")
        canvas_image.paste(im=ori, box=(0, 0))
        canvas_image.paste(im=logo_resized, box=(0, h))
        logo_images.append(canvas_image)
        
    return logo_images
    
def check_empty_text_prompt(text_prompt):
    if len(text_prompt) == 0:
        logging.warn("There is no text content.")
        return 'asdfghjkl'
    return text_prompt

def inpaint_predict(dict, prompt, steps=50, scale=7.5, strength=0.8, seed=42):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    button_update_1 = gr.Button.update(value='重新生成')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)
    
    init_img = dict['image'].convert("RGB")
    mask_img = dict['mask'].convert("RGB")
    init_img = deal_width_exceed_maxside(init_img)
    mask_img = deal_width_exceed_maxside(mask_img)
    
    w, h = init_img.size
    empty_image = None
    prompt = check_empty_text_prompt(prompt)
    if is_contain_chinese(prompt):
        empty_image = check_prompt_to_empty_image(ChineseFilter, prompt)
        if empty_image:
            return empty_image, empty_image, button_update_1, button_update_2, button_update_3
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        return empty_image, empty_image, button_update_1, button_update_2, button_update_3
    with autocast("cuda"):
        generator = torch.Generator("cuda").manual_seed(seed)
        images = paint_pipeline(prompt=prompt, init_image=init_img, mask_image=mask_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["sample"]
        result_image = images[0]
    result_image_with_logo = generate_images_with_logo([result_image])[0]

    return result_image, result_image_with_logo, button_update_1, button_update_2, button_update_3


def outpaint_predict(image, prompt, direction, expand_lenth, steps=50, scale=7.5, strength=0.8, seed=2022):
    outpainting_max_side = MAX_SIDE - 192
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    button_update_1 = gr.Button.update(value='重新生成')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)

    image = deal_width_exceed_maxside(image)
    w, h = image.size
    empty_image = None
    prompt = check_empty_text_prompt(prompt)
    if is_contain_chinese(prompt):
        empty_image = check_prompt_to_empty_image(ChineseFilter, prompt)
        if empty_image:
            return empty_image, empty_image, button_update_1, button_update_2, button_update_3
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        return empty_image, empty_image, button_update_1, button_update_2, button_update_3
    assert expand_lenth % 64 == 0
    if direction == '左' or direction == '右':
        w_expanded = w +  expand_lenth
        h_expanded = h
    else:
        w_expanded = w
        h_expanded = h + expand_lenth
    if w_expanded > outpainting_max_side:
        expand_lenth = outpainting_max_side - w
        w_expanded = outpainting_max_side
    if h_expanded > outpainting_max_side:
        expand_lenth = outpainting_max_side - h
        h_expanded = outpainting_max_side
    if h_expanded == h and w_expanded == w:
        result_image = image
        return result_image, button_update_1, button_update_2, button_update_3
    
    init_img = image.convert("RGB")
    init_img = np.array(init_img)
    if direction == '上':
        init_img = cv2.copyMakeBorder(init_img, 
                                    expand_lenth, 
                                    0,
                                    0,
                                    0,
                                    cv2.BORDER_DEFAULT)
        left_upper = (0, expand_lenth)
    elif direction == '下':
        init_img = cv2.copyMakeBorder(init_img, 
                                    0, 
                                    expand_lenth,
                                    0,
                                    0,
                                    cv2.BORDER_DEFAULT)
        left_upper = (0, 0)
    elif direction == '左':
        init_img = cv2.copyMakeBorder(init_img, 
                                    0, 
                                    0,
                                    expand_lenth,
                                    0,
                                    cv2.BORDER_DEFAULT)
        left_upper = (expand_lenth, 0)
    elif direction == '右':
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
    result_image_with_logo = generate_images_with_logo([result_image])[0]

    return result_image, result_image_with_logo, button_update_1, button_update_2, button_update_3

def multi2image_predict(init_img, prompt, width, height, steps=50, scale=7.5, strength=0.8, seed=2022, num_images=9):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    button_update_1 = gr.Button.update(value='重新生成')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)
    empty_image = None
    prompt = check_empty_text_prompt(prompt)
    if is_contain_chinese(prompt):
        empty_image = check_prompt_to_empty_image(ChineseFilter, prompt)
        if empty_image:
            empty_images = [empty_image] * num_images
            return image_grid(empty_images), empty_images, empty_images, button_update_1, button_update_2, button_update_3
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        empty_images = [empty_image] * num_images
        return image_grid(empty_images), empty_images, empty_images, button_update_1, button_update_2, button_update_3
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
                
    
    grids = image_grid(result_images)
    grids = generate_images_with_logo([grids])[0]
    result_images_with_logo = generate_images_with_logo(result_images)

    return grids, result_images, result_images_with_logo, button_update_1, button_update_2, button_update_3

def variance_predict(prompt, width, height, steps=50, scale=7.5, strength=0.8, seed=2022, num_images=9):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    button_update = gr.Button.update(visible=True)
    empty_image = None
    prompt = check_empty_text_prompt(prompt)
    if is_contain_chinese(prompt):
        empty_image = check_prompt_to_empty_image(ChineseFilter, prompt)
        if empty_image:
            empty_images = [empty_image] * num_images
            return image_grid(empty_images), empty_images, empty_images, button_update
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        empty_images = [empty_image] * num_images
        grids = image_grid(empty_images)
        return grids, empty_images, empty_images, button_update
    result_images = []
    with autocast("cuda"):
        init_imgs = text2image_pipeline(prompt=prompt, height=height, width=width)["sample"]
        init_img = init_imgs[0]
        result_images.append(init_img)
        for i in range(num_images - 1):
            generator = torch.Generator("cuda").manual_seed(seed + i)
            images = image2image_pipeline(prompt=prompt, init_image=init_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["sample"]

            result_images.append(images[0])
            
    grids = image_grid(result_images)
    grids = generate_images_with_logo([grids])[0]
    result_images_with_logo = generate_images_with_logo(result_images)
    return grids, result_images, result_images_with_logo, button_update

class RealESGAN_warp:
    def __init__(self, model):
        self.model = model

    def run(self, image):
        
        assert image is not None, "Input image should not be None"
        if isinstance(image, str):
            image_data = re.sub('^data:image/.+;base64,', '', image)
            image = Image.open(BytesIO(base64.b64decode(image_data)))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        print('start hr convert...')
        output, img_mode = self.model.enhance(np.array(image, dtype=np.uint8))
        res = Image.fromarray(output)
        print('finished.')
        image = generate_images_with_logo([image])[0]
        res = generate_images_with_logo([res])[0]
        tab_update = gr.update(selected='hr_tab')
            
        return image, res, tab_update
