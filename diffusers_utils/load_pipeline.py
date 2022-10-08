from pickle import NONE
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gc
import logging

device = "cuda"

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
    use_auth_token=True
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

class Translator:
    def __init__(self, model_path):
        logging.info('init...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cpu')
        logging.info('finish load translator')
    def __call__(self, text):
        tokenized_text = self.tokenizer.prepare_seq2seq_batch([text], return_tensors='pt').to('cpu')
        translation = self.model.generate(**tokenized_text)
        return self.tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
translator_zh2en = Translator(model_path='models/opus_mt')

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print('finish loading the diffusers models')
