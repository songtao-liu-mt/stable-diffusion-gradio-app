#!/usr/bin/env python

from __future__ import annotations

import argparse

import gradio as gr
import os

from model import AppModel

DESCRIPTION = '''# text2image New version with Stable Diffusion!
Note: This application accepts ONLY English as input.
'''

#This application accepts ONLY English as input.
#But the translation model may mistranslate and the results could be poor.
#So, it is also a good idea to input the translation results from other translation services.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    model = AppModel()

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(value="an attractive young female cyborg, highly detailed oil painting, \
                                            impasto brush strokes, soft light, 8 k, cyberpunk colour palette, \
                                            dramatic composition, dramatic lighting, sharp focus, concept art,\
                                             masterpiece, by adrian ghenie and jenny saville and zhang jingna",
                                      lines=6, label='Input Text')
                    #translate = gr.Checkbox(label='Translate to Chinese',
                                            #value=False)
                    
                    init_img = gr.Image(type="pil", label="Initial image here")
                    gr.Examples(
                        examples = [os.path.join(os.path.dirname(__file__), "assets/stable-samples/img2img/sketch-mountains-input.jpg"),
                                    os.path.join(os.path.dirname(__file__), "assets/logo.jpeg")],
                        inputs = init_img,
                        #cache_examples=True,
                    )
                with gr.Column():
                    seed = gr.Slider(0,
                                        1000,
                                        step=1,
                                        value=42,
                                        label='Seed')
                    width = gr.Slider(512,
                                        960,
                                        step=32,
                                        value=512,
                                        label='width')
                    height = gr.Slider(512,
                                        1024,
                                        step=32,
                                        value=512,
                                        label='height')

                with gr.Column():

                    scale = gr.Slider(1.0,
                                        10.0,
                                        step=0.1,
                                        value=7.5,
                                        label='scale')

                    steps = gr.Slider(20,
                                        200,
                                        step=1,
                                        value=50,
                                        label='steps')

                    num_images = gr.Slider(1,
                                            16,
                                            step=1,
                                            value=4,
                                            label='Number of Images')

                    strength = gr.Slider(0.0,
                                            0.99,
                                            step=0.01,
                                            value=0.9,
                                            label='Weight of Prompt Text')

        run_button = gr.Button('Run')

                    #with open('samples.txt') as f:
                        #samples = [
                            #line.strip().split('\t') for line in f.readlines()
                        #]
                    #examples = gr.Dataset(components=[text, style],
                                          #samples=samples)

        with gr.Column():
            with gr.Group():
                #translated_text = gr.Textbox(label='Translated Text')
                with gr.Tabs():
                    with gr.TabItem('Output (Grid View)'):
                        result_grid = gr.Image(show_label=False)
                    with gr.TabItem('Output (Gallery)'):
                        result_gallery = gr.Gallery(show_label=False)

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
        #examples.click(fn=set_example_text,
                       #inputs=examples,
                       #outputs=examples.components)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7899,
        enable_queue=True,
        share=args.share,
    )


if __name__ == '__main__':
    main()
