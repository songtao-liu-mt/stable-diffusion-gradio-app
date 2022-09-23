import re
from PIL import Image, ImageFilter
from io import BytesIO
import base64
import gradio as gr
import math
import torch


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
        
def copy_res_to_img(img, index):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img[index - 1])
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        return gr.update(value=processed_image)
    except IndexError:
        print("index exceeds range...")
        return None
    

MAX_SIDE = 960
def deal_width_exceed_maxside(image):
    w, h = image.size
    max_side = max(w, h)

    if max_side > MAX_SIDE:
        if max_side == w:
            new_w = MAX_SIDE
            new_h = int(new_w / w * h)
        else:
            new_h = MAX_SIDE
            new_w = int(new_h / h * w)
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)
    w, h = image.size
    w, h = map(lambda x: x - x % 96, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    return image



        
def image_grid(imgs, num_images):
    w, h = imgs[0].size
    rows = math.floor(num_images ** 0.5)
    cols = rows
    grids = Image.new('RGB', size=(cols * w, rows*h))
    grid_w, grid_h = grids.size
    for i, img in enumerate(imgs):
        grids.paste(img, box=(i % cols * w, i // cols * h))
    return grids