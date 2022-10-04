#!/usr/bin/env python

from __future__ import annotations

import argparse, os, sys, glob, re
from select import select
from os import path
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import mimetypes
import base64
import json
import torch

import gradio as gr



__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
from diffusers_func import inpaint_predict, outpaint_predict, multi2image_predict, text2image_predict, variance_predict

DESCRIPTION = '''# 马良 AIGC 创作平台
注意: 此应用目前只支持英文输入
'''

RealESRGAN_dir = 'models/realesrgan'
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    return parser.parse_args()

def readTextFile(*args):
    dir = path.dirname(__file__)
    entry = path.join(dir, *args)
    with open(entry, "r", encoding="utf8") as f:
        data = f.read()
    return data

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def js():
    data = readTextFile("js", "index.js")
    data = "(z) => {" + data + "; return z ?? [] }"
    return data

def call_JS(sd_method, **kwargs):
    param_str = json.dumps(kwargs)
    return f"async (x) => {{ return await SD.{sd_method}({{ x, ...{param_str} }}) ?? []; }}"

def load_RealESRGAN(model_name="RealESRGAN_x4plus", checking=False):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    }
    model_path = os.path.join(RealESRGAN_dir, model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(model_name+".pth not found at path "+model_path)
    if checking == True:
        return True
    sys.path.append(os.path.abspath(RealESRGAN_dir))
    from realesrgan import RealESRGANer

    instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=True)
    instance.model.name = model_name
    return instance

def copy_img_to_sr(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.update(selected='sr_tab')
        img_update = gr.update(value=processed_image)
        return processed_image, tab_update,
    except IndexError:
        return [None, None]

def cp_one_image_from_gallery_to_original(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        button_update_1 = gr.Button.update(value='运行生成')
        button_update_2 = gr.Button.update(visible=False)
        button_update_3 = gr.Button.update(visible=True)
        return processed_image, button_update_1, button_update_2, button_update_3
    except IndexError:
        return [None] * 5


def cp_to_original(img):
    try:
        button_update_1 = gr.Button.update(value='运行生成')
        button_update_2 = gr.Button.update(visible=False)
        button_update_3 = gr.Button.update(visible=True)
        return img, button_update_1, button_update_2, button_update_3
    except IndexError:
        return [None] * 4
    
def cp_to_original_inpaint(img):
    try:
        button_update_1 = gr.Button.update(value='运行生成')
        button_update_2 = gr.Button.update(visible=False)
        button_update_3 = gr.Button.update(visible=True)
        result_img = gr.Image.update(value=img)
        return result_img, button_update_1, button_update_2, button_update_3
    except IndexError:
        return [None] * 4
    
def change_status():
    return gr.Text.update(value="重新生成")   
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
        tab_update = gr.update(selected='hr_tab')
            
        return image, res, tab_update


def main():
    gr.close_all()
    args = parse_args()

    SR_model = RealESGAN_warp(load_RealESRGAN())

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs() as tabs:
            with gr.TabItem('马良生成'):
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Tabs():
                                    with gr.TabItem('文本输入'):
                                        text_prompt = gr.Textbox(value="A fantasy landscape, trending on artstation.",
                                                            lines=12, label='输入文本')
                                    with gr.TabItem('图像输入'):
                                        init_img = gr.Image(type="pil", label="初始图像")
                                        gr.Examples(
                                            examples = [os.path.join(os.path.dirname(__file__), "assets/stable-samples/img2img/sketch-mountains-input.jpg"),
                                                        os.path.join(os.path.dirname(__file__), "assets/logo.jpeg")],
                                            inputs = init_img,
                                        )
                                with gr.Row():
                                    run_button = gr.Button('运行生成', variant="primary")
                        with gr.Column():
                            with gr.Group():
                                width = gr.Slider(512,
                                                    960,
                                                    step=64,
                                                    value=960,
                                                    label='图像宽度')
                                height = gr.Slider(512,
                                                    576,
                                                    step=64,
                                                    value=512,
                                                    label='图像高度')
                                steps = gr.Slider(20,
                                                    200,
                                                    step=1,
                                                    value=50,
                                                    label='步数')
                                scale = gr.Slider(1.0,
                                                    20.0,
                                                    step=0.5,
                                                    value=7.5,
                                                    label='尺度')
                                strength = gr.Slider(0.0,
                                                        0.99,
                                                        step=0.01,
                                                        value=0.9,
                                                        label='文本权重')
                                seed = gr.Slider(0,
                                                    1000,
                                                    step=1,
                                                    value=42,
                                                    label='种子数')
                                num_images = gr.Slider(1,
                                                        9,
                                                        step=1,
                                                        value=4,
                                                        label='图像数量')

                with gr.Column():
                    with gr.Group():
                        #translated_text = gr.Textbox(label='Translated Text')
                        with gr.Tabs() as result_tabs_sd:
                            with gr.TabItem('结果预览'):
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown('选择一张图片并点击 “超分” 生成高清图像')
                                    with gr.Row():
                                        confirm_button = gr.Button(visible=False)
                                        hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                result_gallery = gr.Gallery(labels="Images", elem_id="sd_outputs",).style(grid=[2,2])
                            with gr.TabItem('超分辨率', id='hr_tab'):
                                with gr.TabItem('高清图像'):
                                    hr_image = gr.Image(labels="高清图像", show_label=True,)
                                with gr.TabItem('v.s. 原图'):
                                    low_image = gr.Image(labels="原图", show_label=True)
                            with gr.TabItem('网格展示'):
                                result_grid = gr.Image(show_label=False)
                run_button.click(fn=multi2image_predict,
                                inputs=[
                                    None if not init_img else init_img,
                                    text_prompt,
                                    width,
                                    height,
                                    steps,
                                    scale,
                                    strength,
                                    seed,
                                    num_images
                                ],
                                outputs=[
                                    result_grid,
                                    result_gallery,
                                    run_button,
                                    hr_button,
                                    confirm_button
                                ],
                                )
                
                hr_button.click(fn=SR_model.run,
                                inputs=[result_gallery],
                                outputs=[low_image, hr_image, result_tabs_sd],
                                # scroll_to_output=True,
                                status_tracker=gr.components.StatusTracker(),
                                _js=call_JS("moveImageFromGallery",
                                            fromId="sd_outputs",
                                            toId="hr_outputs")
                                )

            
                # copy_button.click(fn=copy_img_to_sr,
                #                 inputs=[result_gallery],
                #                 outputs=[SR_input_img, tabs],
                #                 _js=call_JS("moveImageFromGallery",
                #                             fromId="sd_outputs",
                #                             toId="sr_input")
                #                 )
        
            with gr.TabItem('相似生成'):
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                
                                    
                                with gr.Tabs():
                                    with gr.TabItem('文本输入'):
                                        text_prompt = gr.Textbox(value="A fantasy landscape, trending on artstation.",
                                                            lines=12, label='输入文本')
                                with gr.Row():
                                    run_button = gr.Button('运行生成', variant="primary")
                                
                        with gr.Column():
                            with gr.Group():
                                width = gr.Slider(512,
                                                    960,
                                                    step=64,
                                                    value=960,
                                                    label='图像宽度')
                                height = gr.Slider(512,
                                                    576,
                                                    step=64,
                                                    value=512,
                                                    label='图像高度')
                                steps = gr.Slider(20,
                                                    200,
                                                    step=1,
                                                    value=50,
                                                    label='步数')
                                scale = gr.Slider(1.0,
                                                    20.0,
                                                    step=0.5,
                                                    value=7.5,
                                                    label='尺度')
                                strength = gr.Slider(0.0,
                                                        0.99,
                                                        step=0.01,
                                                        value=0.9,
                                                        label='强度')
                                seed = gr.Slider(0,
                                                    1000,
                                                    step=1,
                                                    value=42,
                                                    label='种子数')
                                num_images = gr.Slider(1,
                                                        9,
                                                        step=1,
                                                        value=4,
                                                        label='图像数量')   
                with gr.Column():
                    with gr.Group():
                        #translated_text = gr.Textbox(label='Translated Text')
                        with gr.Tabs() as result_tabs_variance:
                            
                            with gr.TabItem('结果预览'):
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown('选择一张图片并点击 “超分” 生成高清图像')
                                    with gr.Row():
                                        hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                result_gallery = gr.Gallery(labels="Images", elem_id="variance_outputs").style(grid=[2,2])
                            with gr.TabItem('超分辨率', id='hr_tab'):
                                with gr.TabItem('高清图像'):
                                    hr_image = gr.Image(labels="高清图像", show_label=True)
                                with gr.TabItem('v.s. 原图'):
                                    low_image = gr.Image(labels="原图", show_label=True)
                            with gr.TabItem('网格展示'):
                                result_grid = gr.Image(show_label=False)
                run_button.click(fn=variance_predict,
                                inputs=[
                                    text_prompt,
                                    width,
                                    height,
                                    steps,
                                    scale,
                                    strength,
                                    seed,
                                    num_images
                                ],
                                outputs=[
                                    result_grid,
                                    result_gallery,
                                    hr_button
                                ])     
                hr_button.click(fn=SR_model.run,
                                inputs=[result_gallery],
                                outputs=[low_image, hr_image, result_tabs_variance],
                                _js=call_JS("moveImageFromGallery",
                                            fromId="variance_outputs",
                                            toId="hr_outputs")
                                )                       
            
            with gr.TabItem('简笔画'):
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Tabs():
                                    with gr.TabItem('图像输入'):
                                        init_img = gr.Image(type="pil", label="初始图像", source="canvas")
                                    with gr.TabItem('文本输入'):
                                        text_prompt = gr.Textbox(lines=12, label='输入文本')
                                with gr.Row():
                                    run_button = gr.Button('运行生成', variant="primary")
                        with gr.Column():
                            with gr.Group():
                                width = gr.Slider(512,
                                                    960,
                                                    step=64,
                                                    value=960,
                                                    label='图像宽度')
                                height = gr.Slider(512,
                                                    576,
                                                    step=64,
                                                    value=512,
                                                    label='图像高度')
                                steps = gr.Slider(20,
                                                    200,
                                                    step=1,
                                                    value=50,
                                                    label='步数')
                                scale = gr.Slider(1.0,
                                                    20.0,
                                                    step=0.5,
                                                    value=7.5,
                                                    label='尺度')
                                strength = gr.Slider(0.0,
                                                        0.99,
                                                        step=0.01,
                                                        value=0.9,
                                                        label='文本权重')
                                seed = gr.Slider(0,
                                                    1000,
                                                    step=1,
                                                    value=42,
                                                    label='种子数')
                                num_images = gr.Slider(1,
                                                        9,
                                                        step=1,
                                                        value=4,
                                                        label='图像数量')

                with gr.Column():
                    with gr.Group():
                        #translated_text = gr.Textbox(label='Translated Text')
                        with gr.Tabs() as result_tabs_sketch:
                            with gr.TabItem('结果预览'):
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown('选择一张图片并点击 “超分” 生成高清图像')
                                    with gr.Row():
                                        confrim_button = gr.Button(visible=False)
                                        hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                result_gallery = gr.Gallery(labels="Images", elem_id="sketch_outputs").style(grid=[2,2])
                            with gr.TabItem('超分辨率', id='hr_tab'):
                                with gr.TabItem('高清图像'):
                                    hr_image = gr.Image(labels="高清图像", show_label=True)
                                with gr.TabItem('v.s. 原图'):
                                    low_image = gr.Image(labels="原图", show_label=True)
                            with gr.TabItem('网格展示'):
                                result_grid = gr.Image(show_label=False)
                run_button.click(fn=multi2image_predict,
                                inputs=[
                                    None if not init_img else init_img,
                                    text_prompt,
                                    width,
                                    height,
                                    steps,
                                    scale,
                                    strength,
                                    seed,
                                    num_images
                                ],
                                outputs=[
                                    result_grid,
                                    result_gallery,
                                    run_button,
                                    hr_button,
                                    confirm_button
                                ])
                
                hr_button.click(fn=SR_model.run,
                                inputs=[result_gallery],
                                outputs=[low_image, hr_image, result_tabs_sketch],
                                _js=call_JS("moveImageFromGallery",
                                            fromId="sketch_outputs",
                                            toId="hr_outputs")
                                )
        
            with gr.TabItem('迭代创作'):
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Tabs():
                                    with gr.TabItem('图像输入'):
                                        init_img = gr.Image(type="pil", label="初始图像", id='input_img')
                                    with gr.TabItem('文本输入'):
                                        text_prompt = gr.Textbox(lines=12, label='输入文本')
                                with gr.Row():
                                    run_button = gr.Button('运行生成', variant="primary")
                                    confirm_button = gr.Button('确认', variant="primary", visible=False)
                        with gr.Column():
                            with gr.Group():
                                
                                width = gr.Slider(512,
                                                    960,
                                                    step=64,
                                                    value=512,
                                                    label='图像宽度')
                                height = gr.Slider(512,
                                                    576,
                                                    step=64,
                                                    value=512,
                                                    label='图像高度')
                                steps = gr.Slider(20,
                                                    200,
                                                    step=1,
                                                    value=50,
                                                    label='步数')
                                scale = gr.Slider(1.0,
                                                    20.0,
                                                    step=0.5,
                                                    value=7.5,
                                                    label='尺度')
                                strength = gr.Slider(0.0,
                                                        0.99,
                                                        step=0.01,
                                                        value=0.9,
                                                        label='文本权重')
                                seed = gr.Slider(0,
                                                    1000,
                                                    step=1,
                                                    value=42,
                                                    label='种子数')
                                num_images = gr.Slider(1,
                                                        9,
                                                        step=1,
                                                        value=9,
                                                        label='图像数量')
                with gr.Column():
                    with gr.Group():
                        with gr.Tabs() as result_tabs_iteration:
                            with gr.TabItem('生成结果'):
                                with gr.Group():
                                    with gr.Row():
                                        confirm_button = gr.Button('选择', variant="primary", visible=False)
                                        hr_button = gr.Button('生成高清图像', variant="primary", visible=False) 
                                result_gallery = gr.Gallery(labels="Images", elem_id="sd_outputs").style(grid=[3,3])
                            with gr.TabItem('超分辨率', id='hr_tab'):
                                with gr.TabItem('高清图像'):
                                    hr_image = gr.Image(labels="高清图像", show_label=True, id='hr_outputs') 
                                with gr.TabItem('v.s. 原图'):
                                    low_image = gr.Image(labels="原图", show_label=True)
                            with gr.TabItem('网格展示'):
                                result_grid = gr.Image(show_label=False)
                run_button.click(fn=multi2image_predict,
                            inputs=[
                                init_img,
                                text_prompt,
                                width,
                                height,
                                steps,
                                scale,
                                strength,
                                seed,
                                num_images
                            ],
                            outputs=[
                                result_grid,
                                result_gallery,
                                run_button,
                                confirm_button,
                                hr_button
                            ])
            
                confirm_button.click(fn=cp_one_image_from_gallery_to_original,
                                    inputs=[
                                        result_gallery
                                    ],
                                    outputs=[
                                        init_img,
                                        run_button,
                                        confirm_button,
                                        hr_button
                                    ],
                                    _js=call_JS("moveImageFromGallery",
                                                fromId="sd_outputs",
                                                toId="input_img")
                                    )
                
                hr_button.click(fn=SR_model.run,
                                inputs=[result_gallery],
                                outputs=[low_image, hr_image, result_tabs_iteration],
                                _js=call_JS("moveImageFromGallery",
                                            fromId="sd_outputs",
                                            toId="hr_outputs")
                                ) 
           
            # with gr.TabItem('局部编辑'):
            #     with gr.Row():
            #         with gr.Row():
            #             with gr.Column():
            #                 with gr.Group():
            #                     with gr.Tabs():
            #                         with gr.TabItem('图像输入'):
            #                             init_img = gr.Image(type="pil", label="初始图像最大边 768", tool='sketch')
            #                         with gr.TabItem('文本输入'):
            #                             text_prompt = gr.Textbox(lines=12, label='输入文本')
            #                     with gr.Row():
            #                         run_button = gr.Button('运行生成', variant="primary")
            #             with gr.Column():
            #                 with gr.Group():

            #                     steps = gr.Slider(20,
            #                                         200,
            #                                         step=1,
            #                                         value=50,
            #                                         label='步数')
            #                     scale = gr.Slider(1.0,
            #                                         20.0,
            #                                         step=0.5,
            #                                         value=7.5,
            #                                         label='尺度')
            #                     strength = gr.Slider(0.0,
            #                                             0.99,
            #                                             step=0.01,
            #                                             value=0.9,
            #                                             label='文本权重')
            #                     seed = gr.Slider(0,
            #                                         1000,
            #                                         step=1,
            #                                         value=42,
            #                                         label='种子数')
            #                     num_images = gr.Slider(1,
            #                                             9,
            #                                             step=1,
            #                                             value=4,
            #                                             label='图像数量')

            #     with gr.Column():
            #         with gr.Group():
            #             #translated_text = gr.Textbox(label='Translated Text')
            #             with gr.Tabs() as result_tabs_iteration:
            #                 with gr.TabItem('生成结果'):
            #                     with gr.Group():
            #                         with gr.Row():
            #                             gr.Markdown('选择一张图片并点击 “超分” 生成高清图像')
            #                         with gr.Row():
            #                             confirm_button = gr.Button('选择', variant="primary", visible=False)
            #                             hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
            #                     result_gallery = gr.Gallery(labels="Images", elem_id="inpaiting_outputs").style(grid=[2,2])
            #                 with gr.TabItem('超分辨率', id='hr_tab'):
            #                     with gr.TabItem('高清图像'):
            #                         hr_image = gr.Image(labels="高清图像", show_label=True)
            #                     with gr.TabItem('v.s. 原图'):
            #                         low_image = gr.Image(labels="原图", show_label=True)
            #                 with gr.TabItem('网格展示'):
            #                     result_grid = gr.Image(show_label=False)
            #     run_button.click(fn=inpaint_predict,
            #                     inputs=[
            #                         None if not init_img else init_img,
            #                         text_prompt,
            #                         steps,
            #                         scale,
            #                         strength,
            #                         seed,
            #                         num_images
            #                     ],
            #                     outputs=[
            #                         result_grid,
            #                         result_gallery,
            #                         run_button,
            #                         # confirm_button,
            #                         hr_button,
            #                     ])
                
            #     # confirm_button.click(fn=cp_one_image_from_gallery_to_original,
            #     #                     inputs=[
            #     #                         result_gallery
            #     #                     ],
            #     #                     outputs=[
            #     #                         init_img,
            #     #                         run_button,
            #     #                         confirm_button,
            #     #                         hr_button
            #     #                     ],
            #     #                     _js=call_JS("moveImageFromGallery",
            #     #                                 fromId="sd_outputs",
            #     #                                 toId="input_img")
            #     #                     )
                
            #     hr_button.click(fn=SR_model.run,
            #                     inputs=[result_gallery],
            #                     outputs=[low_image, hr_image, result_tabs_iteration],
            #                     _js=call_JS("moveImageFromGallery",
            #                                 fromId="inpaiting_outputs",
            #                                 toId="hr_outputs")
            #                     ) 
                
            with gr.TabItem('局部编辑'):
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Tabs():
                                    with gr.TabItem('图像输入'):
                                        init_img = gr.Image(type="pil", label="初始图像最大边 768", tool='sketch')
                                with gr.Row():
                                    run_button = gr.Button('运行生成', variant="primary")
                                    confirm_button = gr.Button('确认', variant="primary", visible=False)
                        with gr.Column():
                            with gr.Tabs():
                                with gr.TabItem('文本输入'):
                                    text_prompt = gr.Textbox(lines=3, label='输入文本')
                                
                            with gr.Group():

                                steps = gr.Slider(20,
                                                    200,
                                                    step=1,
                                                    value=50,
                                                    label='步数')
                                scale = gr.Slider(1.0,
                                                    20.0,
                                                    step=0.5,
                                                    value=7.5,
                                                    label='尺度')
                                strength = gr.Slider(0.0,
                                                        0.99,
                                                        step=0.01,
                                                        value=0.9,
                                                        label='文本权重')
                                seed = gr.Slider(0,
                                                    1000,
                                                    step=1,
                                                    value=42,
                                                    label='种子数')
                with gr.Column():
                    with gr.Group():
                        with gr.Tabs() as result_tabs_outpainting:
                            with gr.TabItem('生成结果'):
                                with gr.Group():
                                    with gr.Row():
                                        confirm_button = gr.Button('确认', variant="primary", visible=False)
                                        hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                result_image = gr.Image(show_label=False, interactive=False)   
                            with gr.TabItem('超分辨率', id='hr_tab'):
                                with gr.TabItem('高清图像'):
                                    hr_image = gr.Image(labels="高清图像", show_label=True) 
                                with gr.TabItem('v.s. 原图'):
                                    low_image = gr.Image(labels="原图", show_label=True)
            
                run_button.click(fn=inpaint_predict,
                            inputs=[
                                None if not init_img else init_img,
                                text_prompt,
                                steps,
                                scale,
                                strength,
                                seed
                            ],
                            outputs=[
                                result_image,
                                run_button,
                                confirm_button,
                                hr_button
                            ])
            
                confirm_button.click(fn=cp_to_original,
                                    inputs=[
                                        result_image
                                    ],
                                    outputs=[
                                        init_img,
                                        run_button,
                                        confirm_button,
                                        hr_button
                                    ]
                                    )
                
                hr_button.click(fn=SR_model.run,
                                    inputs=[result_image],
                                    outputs=[low_image, hr_image, result_tabs_outpainting]
                                    ) 
                
            with gr.TabItem('边缘扩展'):
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Tabs():
                                    with gr.TabItem('图像输入'):
                                        init_img = gr.Image(type="pil", label="初始图像最大边 768")
                                with gr.Row():
                                    run_button = gr.Button('运行生成', variant="primary")
                                    confirm_button = gr.Button('确认', variant="primary", visible=False)
                        with gr.Column():
                            with gr.Tabs():
                                with gr.TabItem('文本输入'):
                                        text_prompt = gr.Textbox(lines=3, label='输入文本')
                            with gr.Group():
                                direction = gr.Radio(
                                     choices=['左', '右', '上', '下'],
                                     value='左', 
                                     label="请选择一个扩展方向"
                                     )
                                expand_lenth = gr.Slider(
                                    0, 
                                    512, 
                                    step=64, 
                                    label='Outpainting Size')
                                steps = gr.Slider(20,
                                                    200,
                                                    step=1,
                                                    value=50,
                                                    label='步数')
                                scale = gr.Slider(1.0,
                                                    20.0,
                                                    step=0.5,
                                                    value=7.5,
                                                    label='尺度')
                                strength = gr.Slider(0.0,
                                                        0.99,
                                                        step=0.01,
                                                        value=0.9,
                                                        label='文本权重')
                                seed = gr.Slider(0,
                                                    1000,
                                                    step=1,
                                                    value=42,
                                                    label='种子数')
                with gr.Column():
                    with gr.Group():
                        with gr.Tabs() as result_tabs_outpainting:
                            with gr.TabItem('生成结果'):
                                with gr.Group():
                                    with gr.Row():
                                        confirm_button = gr.Button('确认', variant="primary", visible=False)
                                        hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                result_image = gr.Image(show_label=False, interactive=False)   
                            with gr.TabItem('超分辨率', id='hr_tab'):
                                with gr.TabItem('高清图像'):
                                    hr_image = gr.Image(labels="高清图像", show_label=True) 
                                with gr.TabItem('v.s. 原图'):
                                    low_image = gr.Image(labels="原图", show_label=True)
            
                run_button.click(fn=outpaint_predict,
                            inputs=[
                                init_img,
                                text_prompt,
                                direction,
                                expand_lenth,
                                steps,
                                scale,
                                strength,
                                seed
                            ],
                            outputs=[
                                result_image,
                                run_button,
                                confirm_button,
                                hr_button
                            ])
            
                confirm_button.click(fn=cp_to_original,
                                    inputs=[
                                        result_image
                                    ],
                                    outputs=[
                                        init_img,
                                        run_button,
                                        confirm_button,
                                        hr_button
                                    ]
                                    )
                
                hr_button.click(fn=SR_model.run,
                                    inputs=[result_image],
                                    outputs=[low_image, hr_image, result_tabs_outpainting]
                                    ) 


        

        
        
            with gr.TabItem('超分辨率', id="sr_tab"):
                    with gr.Column():
                        SR_input_img = gr.Image(type="pil", label="SR input image here", elem_id="sr_input")
                        SR_button = gr.Button('Run SR', variant="primary")
                    with gr.Column():
                        with gr.Tabs() as result_tabs_hr:
                            with gr.TabItem('高清图像', id='hr_tab'):
                                SR_result = gr.Image(show_label=False)
                            with gr.TabItem('v.s. 原图'):
                                low_image = gr.Image(show_label=False)
                                
    
                    SR_button.click(fn=SR_model.run,
                                    inputs=[
                                        SR_input_img,
                                    ],
                                    outputs=[
                                        low_image,
                                        SR_result,
                                        result_tabs_hr
                                    ])
                
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js())
        demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
        
    demo.launch(
        server_name="0.0.0.0",
        server_port=8082,
        enable_queue=True,
        share=args.share,
    )


if __name__ == '__main__':
    main()
