import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from modules.adain.net import vgg, decoder
from modules.adain.function import adaptive_instance_normalization, coral

class AdaINWrapper():
    def __init__(self, device, vgg_path, decoder_path, preserve_color=False, alpha=1.0):
        self.preserve_color = preserve_color
        self.alpha = alpha
        self.vgg = vgg
        self.decoder = decoder
        self.device = device
        self.vgg_path = vgg_path
        self.decoder_path = decoder_path
        self.transforms = transforms.Compose([
                                              transforms.ToTensor(),
                ])
        print('loading the adain vgg and decoder models...')
        self.load_models()
        
    def load_models(self):
        self.vgg.load_state_dict(torch.load(self.vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        self.decoder.load_state_dict(torch.load(self.decoder_path))
        self.vgg.to(self.device)
        self.decoder.to(self.device)
    
    def style_transfer(self, content, style):
        assert (0.0 <= self.alpha <= 1.0)
        content_style_f = self.vgg(torch.cat([content, style], dim=0))
        feat, content_f = adaptive_instance_normalization(content_style_f)
        feat = feat * self.alpha + content_f * (1 - self.alpha)
        return self.decoder(feat)
    
    @torch.no_grad()
    def __call__(self, content_pil, style_pil):
        style_pil = style_pil.resize(content_pil.size)
        content = self.transforms(content_pil.convert('RGB'))
        style = self.transforms(style_pil.convert('RGB'))
        if self.preserve_color:
            style = coral(style, content)
        style = style.to(self.device).unsqueeze(0)
        content = content.to(self.device).unsqueeze(0)
        output = self.style_transfer(content, style)
        output = output[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return Image.fromarray(output)    