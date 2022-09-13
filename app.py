#!/usr/bin/env python

from __future__ import annotations

import argparse, os, sys, glob, re
from os import path
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import mimetypes
import base64
import json
import torch

import gradio as gr

from model import AppModel

DESCRIPTION = '''# text2image New version with Stable Diffusion!
Note: This application accepts ONLY English as input.
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

class RealESGAN_warp:
    def __init__(self, model):
        self.model = model

    def run(self, image):
        assert image is not None, "Input image should not be None!"
        image = image.convert("RGB")
        output, img_mode = self.model.enhance(np.array(image, dtype=np.uint8))
        res = Image.fromarray(output)
        return res


def main():
    args = parse_args()
    model = AppModel()

    SR_model = RealESGAN_warp(load_RealESRGAN())

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs() as tabs:
            with gr.TabItem('Stable Dissusion Generation!'):
                with gr.Row():
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown('Once generation finished, select one image in Gallery and click \'Sent to SR model\' to adapt super resolution!')
                                with gr.Row():
                                    run_button = gr.Button('Run Generation', variant="primary")
                                    copy_button = gr.Button('Sent to SR model', variant="primary")
                                with gr.Tabs():
                                    with gr.TabItem('Text Prompt'):
                                        text = gr.Textbox(value="A fantasy landscape, trending on artstation.",
                                                            lines=6, label='Input Text')
                                    with gr.TabItem('Image Prompt'):
                                        init_img = gr.Image(type="pil", label="Initial image here")
                                        gr.Examples(
                                            examples = [os.path.join(os.path.dirname(__file__), "assets/stable-samples/img2img/sketch-mountains-input.jpg"),
                                                        os.path.join(os.path.dirname(__file__), "assets/logo.jpeg")],
                                            inputs = init_img,
                                        )
                        with gr.Column():
                            with gr.Group():
                                seed = gr.Slider(0,
                                                    1000,
                                                    step=1,
                                                    value=42,
                                                    label='Seed')
                                width = gr.Slider(512,
                                                    960,
                                                    step=64,
                                                    value=960,
                                                    label='width')
                                height = gr.Slider(512,
                                                    576,
                                                    step=64,
                                                    value=512,
                                                    label='height')

                                scale = gr.Slider(1.0,
                                                    20.0,
                                                    step=0.5,
                                                    value=7.5,
                                                    label='scale')
                                steps = gr.Slider(20,
                                                    200,
                                                    step=1,
                                                    value=50,
                                                    label='steps')
                                num_images = gr.Slider(1,
                                                        1,
                                                        step=1,
                                                        value=1,
                                                        label='Number of Images')
                                strength = gr.Slider(0.0,
                                                        0.99,
                                                        step=0.01,
                                                        value=0.9,
                                                        label='Weight of Prompt Text')


                with gr.Column():
                    with gr.Group():
                        #translated_text = gr.Textbox(label='Translated Text')
                        with gr.Tabs():
                            with gr.TabItem('Output (Gallery)'):
                                result_gallery = gr.Gallery(labels="Images", elem_id="sd_outputs").style(grid=[2,2])
                            with gr.TabItem('Output (Grid View)'):
                                result_grid = gr.Image(show_label=False)

                run_button.click(fn=model.run_with_prompt,
                                inputs=[
                                    seed,
                                    text,
                                    num_images,
                                    width,
                                    height,
                                    scale,
                                    steps,
                                    strength,
                                    None if not init_img else init_img,
                                ],
                                outputs=[
                                    result_grid,
                                    result_gallery,
                                ])

            with gr.TabItem('Super Resolution!', id="sr_tab"):
                with gr.Column():
                    SR_input_img = gr.Image(type="pil", label="SR input image here", elem_id="sr_input")
                    SR_button = gr.Button('Run SR', variant="primary")
                with gr.Column():
                    SR_result = gr.Image(show_label=False)
 
                SR_button.click(fn=SR_model.run,
                                inputs=[
                                    SR_input_img,
                                ],
                                outputs=[
                                    SR_result,
                                ])
                copy_button.click(fn=copy_img_to_sr,
                                inputs=[result_gallery],
                                outputs=[SR_input_img, tabs],
                                _js=call_JS("moveImageFromGallery",
                                            fromId="sd_outputs",
                                            toId="sr_input")
                                )
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js())
        demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)

    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        enable_queue=True,
        share=args.share,
    )


if __name__ == '__main__':
    main()
