import re
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os
import sys



class RealESGANWrapper:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()
        
    def load_model(self, checking=False):
        
        RealESRGAN_models = {
            'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        }
        model_name = self.model_path.split('/')[-1].split('.')[0]
        if not os.path.isfile(self.model_path):
            raise Exception(model_name+".pth not found at path " + self.model_path)
        if checking == True:
            return True

        instance = RealESRGANer(scale=2, model_path=self.model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=True)
        instance.model.name = model_name
        return instance

    def __call__(self, image):
        
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
            
        return res