import torch
from torch import nn
from torchvision import transforms
import cv2
from PIL import Image
import os
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(__dir__, '../'))
from clipseg.model import CLIPDensePredT

class CLIPSegWrapper():
    def __init__(self, device):
        self.device = device
        self.model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512)),
        ])
        print("loading the clipseg models...")
        self.load_models()
    def load_models(self):
        self.model.load_state_dict(torch.load('models/clipseg/rd64-uni.pth', map_location='cpu'), strict=False)
    
    def get_mask_edge(self, mask):
        mask = np.array(mask)
        (thresh, bw_image) = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        edge = cv2.Canny(bw_image, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        edge = cv2.dilate(edge, kernel)
        
        mask = Image.fromarray(bw_image)
        edge = Image.fromarray(edge)
        return mask, edge
        
    @torch.no_grad()
    def __call__(self, prompts, image):
        if isinstance(prompts, str):
            prompts = [prompts]
        image = image.resize((512, 512)).convert('RGB')
        image = self.transforms(image).unsqueeze(0)
        preds = torch.sigmoid(self.model(image.repeat(len(prompts), 1, 1, 1), prompts)[0])
        preds = preds[0][0].mul(255).to('cpu', torch.uint8).numpy()
        preds = Image.fromarray(preds)
        
        return self.get_mask_edge(preds)
    
if __name__ == "__main__":
    clipseg_wrapper = CLIPSegWrapper(device="cuda:1")
    image = Image.open("mt_images/forbidden.jpg")
    prompt = "shirt"
    mask = clipseg_wrapper(prompt, image)
    mask.save('tmp.jpg')