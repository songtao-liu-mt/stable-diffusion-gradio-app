from pickle import NONE
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from transformers import BertTokenizer, BertForSequenceClassification
import torch

device = "cuda:6"

class DummySafeChecker():
    def __init__(self,):
        pass

    def __call__(self, clip_input, images):
        return images, None
    
class DummyFeatureExtractor():
    def __init__(self,):
        pass
    def __call__(self, image):
        return image, None
    
print('prepare diffusers models...')
paint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=False
).to(device)

paint_pipeline.safety_checker = DummySafeChecker()
paint_pipeline.feature_extractor = DummyFeatureExtractor()


image2image_pipeline = StableDiffusionImg2ImgPipeline(
    vae = paint_pipeline.vae,
    text_encoder = paint_pipeline.text_encoder,
    tokenizer = paint_pipeline.tokenizer,
    unet = paint_pipeline.unet,
    scheduler = paint_pipeline.scheduler,
    safety_checker = paint_pipeline.safety_checker,
    feature_extractor = paint_pipeline.feature_extractor,
).to(device)

text2image_pipeline = StableDiffusionPipeline(
    vae = paint_pipeline.vae,
    text_encoder = paint_pipeline.text_encoder,
    tokenizer = paint_pipeline.tokenizer,
    unet = paint_pipeline.unet,
    scheduler = paint_pipeline.scheduler,
    safety_checker = paint_pipeline.safety_checker,
    feature_extractor = paint_pipeline.feature_extractor,
).to(device)

# torch.cuda.empty_cache()
print('finish loading the diffusers models')