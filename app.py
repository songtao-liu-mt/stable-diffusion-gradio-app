#!/usr/bin/env python

from __future__ import annotations

import argparse, os, sys, glob, re
from pickle import FALSE
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
from diffusers_func import inpaint_predict, outpaint_predict, multi2image_predict, variance_predict, RealESGAN_warp

DESCRIPTION = '''# 摩尔线程马良 AIGC 创作平台
'''

GUIDENCE = '''为了更好地生成文本提示，从而生成优美的图片。文本输入通常遵循一定的原则。本篇将带领大家生成一个漂亮的文本输入。

注意：为了净化网络环境，请慎重输入词汇，避免 **敏感词**。

#### 文本核心描述（必写）
文本核心描述为文本提示最核心的部分，也是必不可少的部分。一般这个不分需要至少一个核心主体名词加上一定数量的形容词汇，或者动作描述。例如：
1. 一个正在跑步的女孩
2. 一只聪明的熊猫在喝可口可乐

#### 类型（必写）
类型也是文本必不可少的内容。通常来说，类型包括但不限于 照片、肖像、油画、草图、素描、电影海报、广告、3D渲染、图案等等

以上两部分就可以醉成最简单的文本描述。例如：
1. 一个正在跑步的女孩的照片
2. 一只聪明的熊猫在喝可口可乐的电影海报

除了以上两部分为必写内容，可以按照自己的想法加入图像的风格，主体的颜色，图像的纹理，图像分辨率，图像的类型，任务的情绪，所处的时代等等。简单的形式可以用逗号的形式分开。不同的顺序也会产生不同的结果。但是，每一个句子里的中文单字 + 英文单词 建议不要超过 77 个。

在「马良生成」「文本生图」「文本输入」下方有一些中英文例子。


'''
SUB_GUIDENCE = '为了净化网络环境，请慎重输入词汇，避免 **敏感词**。建议先阅读「马良小助手」中的「文本输入引导」'
TEXT_EXAMPLES = [
            '一个正在跑步的女孩的照片，穿着红色裙子，8k高清的图片，背景虚化的拍照风格',
            '一只聪明的熊猫在喝可口可乐的电影海报， 戴着极光蓝的帽子，8k高清图片，在宋朝时期',
            'low wide angle shot of dilapidated fallout 5 miami, tropical coastal city, desolate, dilapidated neon signs, few rusted retro futuristic vintage parked vehicles like cars, buses, trucks, trams, volumetric lighting, photorealistic, foggy, rain daytime, autumn, overcast weather, sharp focus, ultra detailed, 8k',
            'magic fluffy Persian carpet dimension, by Greg Rutkowski and Gaston Bussiere, dim lighting, beautiful volumetric-lighting-style atmosphere, surreal atmosphere, intricate, detailed, photorealistic imagery, artstation',
        ]

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
        img_update = gr.update(value=processed_image)
        button_update_1 = gr.Button.update(value='运行生成')
        button_update_2 = gr.Button.update(visible=False)
        button_update_3 = gr.Button.update(visible=True)
        return img_update, button_update_1, button_update_2, button_update_3
    except IndexError:
        return [None] * 5


def cp_to_original(img):
    try:
        button_update_1 = gr.Button.update(value='运行生成')
        button_update_2 = gr.Button.update(visible=False)
        button_update_3 = gr.Button.update(visible=True)
        result_img = gr.Image.update(value=img)
        return result_img, button_update_1, button_update_2, button_update_3
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



def main():
    gr.close_all()
    args = parse_args()

    SR_model = RealESGAN_warp(load_RealESRGAN())

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)
        # gr.Markdown(GUIDENCE)
        with gr.Tabs() as tabs:
            with gr.TabItem('马良生成'):
                with gr.TabItem('文本生图'):
                    with gr.Row():
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Tabs():
                                        with gr.TabItem('文本输入'):
                                            lang = gr.Radio(value='英文', choices=['英文', '中文'], show_label=False)
                                            text_prompt = gr.Textbox(value='', label='输入文本前，请先阅读「马良小助手」「文本输入引导」', lines=6, interactive=True)
                                            gr.Examples(
                                                label='文本示例',
                                                examples=TEXT_EXAMPLES,
                                                inputs=text_prompt,
                                                
                                            )
                                        with gr.TabItem('图像输入'):
                                            init_img = gr.Image(type="pil", label="初始图像")
                                            gr.Examples(
                                                label='图像示例',
                                                examples=[
                                                            os.path.join(os.path.dirname(__file__), "mt_images/logo.jpeg")
                                                        ],
                                                inputs=init_img,
                                            )
                                        
                                    with gr.Row():
                                        run_button = gr.Button('运行生成', variant="primary")
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
                                                            value=4,
                                                            label='图像数量')

                    with gr.Column():
                        with gr.Group():
                            with gr.Tabs() as result_tabs_sd:
                                with gr.TabItem('结果预览'):
                                    with gr.Group():
                                        with gr.Row():
                                            gr.Markdown('选择一张图片并点击 “超分” 生成高清图像')
                                        with gr.Row():
                                            confirm_button = gr.Button(visible=False)
                                            hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                    result_gallery_ori = gr.Gallery(labels="Images", elem_id="ori_outputs", show_label=False, visible=False)
                                    result_gallery = gr.Gallery(labels="Images", elem_id="sd_outputs", show_label=False).style(grid=[2,2])
                                with gr.TabItem('超分辨率', id='hr_tab'):
                                    with gr.TabItem('高清图像'):
                                        hr_image = gr.Image(labels="高清图像", show_label=False, id='hr_outputs')
                                    with gr.TabItem('v.s. 原图'):
                                        low_image = gr.Image(labels="原图", show_label=False)
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
                                        num_images,
                                        lang
                                    ],
                                    outputs=[
                                        result_grid,
                                        result_gallery_ori,
                                        result_gallery,
                                        run_button,
                                        hr_button,
                                        confirm_button
                                    ],
                                    )
                    
                    hr_button.click(fn=SR_model.run,
                                    inputs=[result_gallery_ori],
                                    outputs=[low_image, hr_image, result_tabs_sd],
                                    _js=call_JS("moveImageFromGallery",
                                                fromId="sd_outputs",
                                                toId="hr_outputs")
                                    )
            
                with gr.TabItem('相似生成'):
                    with gr.Row():
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    
                                        
                                    with gr.Tabs():
                                        with gr.TabItem('文本输入'):
                                            lang = gr.Radio(value='英文', choices=['英文', '中文'], show_label=False)
                                            text_prompt = gr.Textbox(value="",
                                                                lines=12, label=SUB_GUIDENCE)
                                    with gr.Row():
                                        run_button = gr.Button('运行生成', variant="primary")
                                    
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
                            with gr.Tabs() as result_tabs_variance:
                                
                                with gr.TabItem('结果预览'):
                                    with gr.Group():
                                        with gr.Row():
                                            gr.Markdown('选择一张图片并点击 “超分” 生成高清图像')
                                        with gr.Row():
                                            hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                    result_gallery_ori = gr.Gallery(labels="Images", elem_id="ori_variance_outputs", show_label=False, visible=False)
                                    result_gallery = gr.Gallery(labels="Images", elem_id="variance_outputs", show_label=False).style(grid=[2,2])
                                with gr.TabItem('超分辨率', id='hr_tab'):
                                    with gr.TabItem('高清图像'):
                                        hr_image = gr.Image(labels="高清图像", show_label=False, id='hr_outputs')
                                    with gr.TabItem('v.s. 原图'):
                                        low_image = gr.Image(labels="原图", show_label=False)
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
                                        num_images,
                                        lang
                                    ],
                                    outputs=[
                                        result_grid,
                                        result_gallery_ori,
                                        result_gallery,
                                        hr_button
                                    ])     
                    hr_button.click(fn=SR_model.run,
                                    inputs=[result_gallery_ori],
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
                                            init_img = gr.Image(type="pil", label="初始图像", source="canvas", tool="color-sketch")
                                        with gr.TabItem('文本输入'):
                                            lang = gr.Radio(value='英文', choices=['英文', '中文'], show_label=False)
                                            text_prompt = gr.Textbox(lines=12, label=SUB_GUIDENCE)
                                    with gr.Row():
                                        run_button = gr.Button('运行生成', variant="primary")
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
                                                            value=4,
                                                            label='图像数量')

                    with gr.Column():
                        with gr.Group():
                            with gr.Tabs() as result_tabs_sketch:
                                with gr.TabItem('结果预览'):
                                    with gr.Group():
                                        with gr.Row():
                                            gr.Markdown('选择一张图片并点击 “超分” 生成高清图像')
                                        with gr.Row():
                                            confrim_button = gr.Button(visible=False)
                                            hr_button = gr.Button('生成高清图像', variant="primary", visible=False)
                                    result_gallery_ori = gr.Gallery(labels="Images", elem_id="ori_sketch_outputs", show_label=False, visible=False)
                                    result_gallery = gr.Gallery(labels="Images", elem_id="sketch_outputs", show_label=False).style(grid=[2,2])
                                with gr.TabItem('超分辨率', id='hr_tab'):
                                    with gr.TabItem('高清图像'):
                                        hr_image = gr.Image(labels="高清图像", show_label=False, id='hr_outputs')
                                    with gr.TabItem('v.s. 原图'):
                                        low_image = gr.Image(labels="原图", show_label=False)
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
                                        num_images,
                                        lang
                                    ],
                                    outputs=[
                                        result_grid,
                                        result_gallery_ori,
                                        result_gallery,
                                        run_button,
                                        hr_button,
                                        confirm_button
                                    ])
                    
                    hr_button.click(fn=SR_model.run,
                                    inputs=[result_gallery_ori],
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
                                            lang = gr.Radio(value='英文', choices=['英文', '中文'], show_label=False)
                                            text_prompt = gr.Textbox(lines=12, label=SUB_GUIDENCE)
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
                                            confirm_button = gr.Button('确定', variant="primary", visible=False)
                                            hr_button = gr.Button('生成高清图像', variant="primary", visible=False) 
                                    result_gallery_ori = gr.Gallery(labels="Images", elem_id="ori_iterative_outputs", show_label=False, visible=False)
                                    result_gallery = gr.Gallery(labels="Images", elem_id="iterative_outputs", show_label=False).style(grid=[3,3])
                                with gr.TabItem('超分辨率', id='hr_tab'):
                                    with gr.TabItem('高清图像'):
                                        hr_image = gr.Image(labels="高清图像", show_label=False, id='hr_outputs') 
                                    with gr.TabItem('v.s. 原图'):
                                        low_image = gr.Image(labels="原图", show_label=False)
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
                                    num_images,
                                    lang
                                    
                                ],
                                outputs=[
                                    result_grid,
                                    result_gallery_ori,
                                    result_gallery,
                                    run_button,
                                    confirm_button,
                                    hr_button
                                ])
                
                    confirm_button.click(fn=cp_one_image_from_gallery_to_original,
                                        inputs=[
                                            result_gallery_ori
                                        ],
                                        outputs=[
                                            init_img,
                                            run_button,
                                            confirm_button,
                                            hr_button
                                        ],
                                        _js=call_JS("moveImageFromGallery",
                                                    fromId="iterative_outputs",
                                                    toId="input_img")
                                        )
                    
                    hr_button.click(fn=SR_model.run,
                                    inputs=[result_gallery_ori],
                                    outputs=[low_image, hr_image, result_tabs_iteration],
                                    _js=call_JS("moveImageFromGallery",
                                                fromId="iterative_outputs",
                                                toId="hr_outputs")
                                    ) 
                    
                with gr.TabItem('局部编辑'):
                    with gr.Row():
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Tabs():

                                        with gr.TabItem('图像输入'):
                                            init_img = gr.Image(type="pil", label="初始图像最大边 768", tool='sketch', source='upload')
                                    with gr.Row():
                                        run_button = gr.Button('运行生成', variant="primary")
                                        confirm_button = gr.Button('确认', variant="primary", visible=False)
                            with gr.Column():
                                with gr.Tabs():
                                    with gr.TabItem('文本输入'):
                                        lang = gr.Radio(value='英文', choices=['英文', '中文'], show_label=False)
                                        text_prompt = gr.Textbox(lines=3, label=SUB_GUIDENCE)
                                    
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
                                    result_image_ori = gr.Image(show_label=False, interactive=False, visible=False)  
                                    result_image = gr.Image(show_label=False, interactive=False)
                                with gr.TabItem('超分辨率', id='hr_tab'):
                                    with gr.TabItem('高清图像'):
                                        hr_image = gr.Image(labels="高清图像", show_label=False, id='hr_outputs') 
                                    with gr.TabItem('v.s. 原图'):
                                        low_image = gr.Image(labels="原图", show_label=False)
                
                    run_button.click(fn=inpaint_predict,
                                inputs=[
                                    None if not init_img else init_img,
                                    text_prompt,
                                    steps,
                                    scale,
                                    strength,
                                    seed,
                                    lang
                                ],
                                outputs=[
                                    result_image_ori,
                                    result_image,
                                    run_button,
                                    confirm_button,
                                    hr_button
                                ])
                
                    confirm_button.click(fn=cp_to_original,
                                        inputs=[
                                            result_image_ori
                                        ],
                                        outputs=[
                                            init_img,
                                            run_button,
                                            confirm_button,
                                            hr_button
                                        ]
                                        )
                    
                    hr_button.click(fn=SR_model.run,
                                        inputs=[result_image_ori],
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
                                        lang = gr.Radio(value='英文', choices=['英文', '中文'], show_label=False)
                                        text_prompt = gr.Textbox(lines=3, label=SUB_GUIDENCE)
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
                                        label='扩展尺寸')
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
                                    result_image_ori = gr.Image(show_label=False, interactive=False, visible=False) 
                                    result_image = gr.Image(show_label=False, interactive=False)   
                                with gr.TabItem('超分辨率', id='hr_tab'):
                                    with gr.TabItem('高清图像'):
                                        hr_image = gr.Image(labels="高清图像", show_label=False, id='hr_outputs') 
                                    with gr.TabItem('v.s. 原图'):
                                        low_image = gr.Image(labels="原图", show_label=False)
                
                    run_button.click(fn=outpaint_predict,
                                inputs=[
                                    None if not init_img else init_img,
                                    text_prompt,
                                    direction,
                                    expand_lenth,
                                    steps,
                                    scale,
                                    strength,
                                    seed,
                                    lang
                                ],
                                outputs=[
                                    result_image_ori, 
                                    result_image,
                                    run_button,
                                    confirm_button,
                                    hr_button
                                ])
                
                    confirm_button.click(fn=cp_to_original,
                                        inputs=[
                                            result_image_ori
                                        ],
                                        outputs=[
                                            init_img,
                                            run_button,
                                            confirm_button,
                                            hr_button
                                        ]
                                        )
                    
                    hr_button.click(fn=SR_model.run,
                                        inputs=[result_image_ori],
                                        outputs=[low_image, hr_image, result_tabs_outpainting]
                                        ) 
            
                with gr.TabItem('超分辨率', id="sr_tab"):
                        with gr.Column():
                            SR_input_img = gr.Image(type="pil", label="低分辨率图像", elem_id="sr_input")
                            SR_button = gr.Button('生成高清图像', variant="primary")
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
                
            with gr.TabItem('马良小助手'):
                with gr.TabItem('流程演示'):
                    gr.Video(value='demos/maliang_demo.mp4',
                            format='mp4',
                            label='请点击播放，观看完整流程教学视频（建议打开全屏模式）。',
                            show_label=True,
                            interactive=False
                            )
                with gr.TabItem('文本输入引导'):
                    gr.Markdown(GUIDENCE)
                
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js())
        demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
        
    demo.launch(
        server_name="0.0.0.0",
        server_port=7800,
        enable_queue=True,
        share=args.share,
    )


if __name__ == '__main__':
    main()
