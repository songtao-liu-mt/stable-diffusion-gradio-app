import re
from PIL import Image, ImageFilter
from io import BytesIO
import base64
import gradio as gr
import math
import torch
        
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
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    return image



        
def image_grid(imgs):
    num_images = len(imgs)
    w, h = imgs[0].size
    rows = math.floor(num_images ** 0.5)
    cols = rows
    grids = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grids.paste(img, box=(i % cols * w, i // cols * h))
    return grids

def is_contain_chinese(text):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    
    return len(text.encode('utf-8')) > len(text)