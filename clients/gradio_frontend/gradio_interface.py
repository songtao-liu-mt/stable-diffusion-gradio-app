import gradio as gr

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
sys.path.append(__dir__)
from app import (
    multi2image,
    variance,
    inpaint,
    outpaint,
    sr,
    styletransfer,
    # replaced,
    # image2text
)

def multi2image_predict(init_img, prompt, width, height, steps=30, scale=7.5, strength=0.8, seed=2022, num_images=9):
    button_update_1 = gr.Button.update(value='重新生成')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)
    grids, result_images, result_images_with_logo = multi2image(init_img, prompt, width, height, steps, scale, strength, seed, num_images)
    return grids, result_images, result_images_with_logo, button_update_1, button_update_2, button_update_3

def variance_predict(prompt, width, height, steps=30, scale=7.5, strength=0.8, seed=2022, num_images=9):
    button_update = gr.Button.update(visible=True)
    grids, result_images, result_images_with_logo = variance(prompt, width, height, steps, scale, strength, seed, num_images)
    return grids, result_images, result_images_with_logo, button_update

def inpaint_predict(dict, prompt, steps=30, scale=7.5, seed=42):
    button_update_1 = gr.Button.update(value='重新生成')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)
    result_image, result_image_with_logo = inpaint(dict, prompt, steps, scale, seed)
    return result_image, result_image_with_logo, button_update_1, button_update_2, button_update_3

def outpaint_predict(image, prompt, direction, expand_lenth, steps=30, scale=7.5, seed=2022):
    button_update_1 = gr.Button.update(value='重新生成')
    button_update_2 = gr.Button.update(visible=True)
    button_update_3 = gr.Button.update(visible=False)
    result_image, result_image_with_logo = outpaint(image, prompt, direction, expand_lenth, steps, scale, seed)
    return result_image, result_image_with_logo, button_update_1, button_update_2, button_update_3

def sr_predict(image):
    tab_update = gr.update(selected='hr_tab')
    image, res = sr(image)
    return image, res, tab_update

def styletransfer_predict(content_pil, style_pil):
    button_update_1 = gr.Button.update(value='重新生成')
    button_update_2 = gr.Button.update(visible=True)
    result_image, result_image_with_logo = styletransfer(content_pil, style_pil)
    return result_image, result_image_with_logo, button_update_1, button_update_2
    
# def replaced_predict(init_img, part_prompt, result_prompt, steps=30, scale=7.5, seed=42):
#     result_image, result_image_with_logo = replaced(init_img, part_prompt, result_prompt, steps, scale, seed)
#     return result_image, result_image_with_logo

# def image2text_predict(image):
    
#     text = image2text(image)
#     return text



