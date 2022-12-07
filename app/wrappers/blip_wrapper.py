import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))

from modules.BLIP.models.blip import blip_decoder
blip_dir = os.path.join(__dir__, '../modules/BLIP')

class BlipWrapper:
    def __init__(self, model_path, device, image_size=384):
        self.device = device
        self.image_size = image_size
        print("loading blip model...")
        self.blip_model = blip_decoder(pretrained=model_path, 
                                       image_size=self.image_size, 
                                       med_config=os.path.join(blip_dir, 'configs/med_config.json')
                                       )
        self.blip_model.eval()
        self.blip_model.to(self.device)
        
        self.transform = transforms.Compose([
                        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        
    def __call__(self, image):
        with torch.no_grad():
            image = self.load_image(image=image, image_size=self.image_size)
            caption = self.blip_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)[0]
        caption = caption.strip("'")
        return caption
    
    def load_image(self, image):
        image = image.convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image