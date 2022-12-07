import logging
import re
from io import BytesIO
import base64
import torch
import numpy as np
from PIL import Image
import cv2
import gc

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
from load_models import (
    text2image_pipeline,
    image2image_pipeline,
    paint_pipeline,
    real_esr,
    translator_zh2en,
    adain_trasfer,
    image2text_pipeline,
    clipseg,
    logo_image_pil,
    forbidden_pil,
)

from common import (
    deal_width_exceed_maxside, 
    image_grid, 
    MAX_SIDE,
    is_contain_chinese,
)
sys.path.append(os.path.dirname(__dir__))
from app.language_filter.sensewords import (
    ChineseFilter, 
    EnglishFilter
)

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
        return ' '
    return text_prompt

def multi2image(init_img, prompt, width, height, steps=50, scale=7.5, strength=0.8, seed=2022, num_images=9):
    gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    
    empty_image = None
    prompt = check_empty_text_prompt(prompt)
    if is_contain_chinese(prompt):
        empty_image = check_prompt_to_empty_image(ChineseFilter, prompt)
        if empty_image:
            empty_images = [empty_image] * num_images
            return image_grid(empty_images), empty_images, empty_images
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        empty_images = [empty_image] * num_images
        return image_grid(empty_images), empty_images, empty_images
    result_images = []
    if init_img is not None:
        init_img = init_img.convert("RGB").resize((width, height))
        
        with torch.autograd.inference_mode(mode=True):
            for i in range(num_images):
                generator = torch.Generator("cpu").manual_seed(seed + i)
                images = image2image_pipeline(prompt=prompt, init_image=init_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["images"]
                result_images.append(images[0])
    else:
        with torch.autograd.inference_mode(mode=True):
            for i in range(num_images):
                generator = torch.Generator("cpu").manual_seed(seed + i)
                images = text2image_pipeline(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["images"]
                result_images.append(images[0])
                # images[0].save('logos/{:05d}.png'.format(i))
    
    grids = image_grid(result_images)
    grids = generate_images_with_logo([grids])[0]
    result_images_with_logo = generate_images_with_logo(result_images)

    return grids, result_images, result_images_with_logo


def variance(prompt, width, height, steps=50, scale=7.5, strength=0.8, seed=2022, num_images=9):
    gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    
    empty_image = None
    prompt = check_empty_text_prompt(prompt)
    if is_contain_chinese(prompt):
        empty_image = check_prompt_to_empty_image(ChineseFilter, prompt)
        if empty_image:
            empty_images = [empty_image] * num_images
            return image_grid(empty_images), empty_images, empty_images
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        empty_images = [empty_image] * num_images
        grids = image_grid(empty_images)
        return grids, empty_images, empty_images
    result_images = []
    with torch.autograd.inference_mode(mode=True):
        init_imgs = text2image_pipeline(prompt=prompt, height=height, width=width)["images"]
        init_img = init_imgs[0]
        result_images.append(init_img)
        for i in range(num_images - 1):
            generator = torch.Generator("cpu").manual_seed(seed + i)
            images = image2image_pipeline(prompt=prompt, init_image=init_img, num_inference_steps=steps, guidance_scale=scale, strength=strength, generator=generator)["images"]

            result_images.append(images[0])
            
    grids = image_grid(result_images)
    grids = generate_images_with_logo([grids])[0]
    result_images_with_logo = generate_images_with_logo(result_images)
    return grids, result_images, result_images_with_logo

def inpaint(dict, prompt, steps=50, scale=7.5, seed=42):
    gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    
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
            return empty_image, empty_image
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        return empty_image, empty_image
    with torch.autograd.inference_mode(mode=True):
        generator = torch.Generator("cpu").manual_seed(seed)
        images = paint_pipeline(prompt=prompt, init_image=init_img, mask_image=mask_img, height=h, width=w, num_inference_steps=steps, guidance_scale=scale, generator=generator)["images"]
        result_image = images[0]
    result_image_with_logo = generate_images_with_logo([result_image])[0]

    return result_image, result_image_with_logo


def outpaint(image, prompt, direction, expand_lenth, steps=50, scale=7.5, seed=2022):
    outpainting_max_side = MAX_SIDE - 192
    gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()

    image = deal_width_exceed_maxside(image)
    w, h = image.size
    empty_image = None
    prompt = check_empty_text_prompt(prompt)
    if is_contain_chinese(prompt):
        empty_image = check_prompt_to_empty_image(ChineseFilter, prompt)
        if empty_image:
            return empty_image, empty_image
        prompt = translator_zh2en(prompt)
    empty_image = check_prompt_to_empty_image(EnglishFilter, prompt)
    if empty_image:
        return empty_image, empty_image
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
        return result_image, generate_images_with_logo([result_image])[0]
    
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
    
    
    
    with torch.autograd.inference_mode(mode=True):
        generator = torch.Generator("cpu").manual_seed(seed)
        images = paint_pipeline(prompt=prompt, init_image=init_img, mask_image=mask_img, num_inference_steps=steps, guidance_scale=scale, generator=generator)["images"]
        result_image = images[0]
    result_image_with_logo = generate_images_with_logo([result_image])[0]

    return result_image, result_image_with_logo

def sr(image):
    image = generate_images_with_logo([image])[0]
    res = real_esr(image)
    res = generate_images_with_logo([res])[0]
    return image, res

def styletransfer(content_pil, style_pil):
    gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()

    content_pil = deal_width_exceed_maxside(content_pil)
    
    result_image = adain_trasfer(content_pil, style_pil)
    result_image_with_logo = generate_images_with_logo([result_image])[0]

    return result_image, result_image_with_logo
    
# def replaced(init_img, part_prompt, result_prompt, steps=50, scale=7.5, seed=42):
#     mask, edge = clipseg(part_prompt, init_img)
#     h = 512
#     w = 512
#     init_img = init_img.resize((512, 512)).convert('RGB')
#     with torch.autograd.inference_mode(mode=True):
#         generator = torch.Generator("cpu").manual_seed(seed)
#         images = paint_pipeline(prompt=result_prompt, init_image=init_img, mask_image=mask, height=h, width=w, num_inference_steps=steps, guidance_scale=scale, generator=generator)["images"]
#         # images = paint_pipeline(prompt=' ', init_image=images[0], mask_image=edge, height=h, width=w, num_inference_steps=steps, guidance_scale=scale, generator=generator)["images"]
#         result_image = images[0]
#     result_image_with_logo = generate_images_with_logo([result_image])[0]

#     return result_image, result_image_with_logo



# def image2text(image):
#     gc.collect()
#     # torch.cuda.empty_cache()
#     # torch.cuda.ipc_collect()
#     text = image2text_pipeline(image)
#     return text