import argparse, os, sys, glob, re
import torch
import PIL
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from mt_filter.sensewords import ChineseFilter, EnglishFilter
import logging
import musa_torch_extension

logging.basicConfig(filename="./filter.log", level=logging.INFO)
#logger = logging.getLogger("app_filter_log.txt")
forbidden_pil = Image.open('mt_images/forbidden_2048_1536_logo.jpg').resize((512, 384)).convert('RGB')
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

class Translator:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cpu')
    def __call__(self, text):
        if zhPattern.search(text):
            tokenized_text = self.tokenizer.prepare_seq2seq_batch([text], return_tensors='pt').to('cpu')
            translation = self.model.generate(**tokenized_text)
            return self.tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
        else:
            return text

translator_zh2en = Translator(model_path='models/opus_mt')

class TextFilter:
    def __init__(self):
        self.zh_filter = ChineseFilter
        self.en_filter = EnglishFilter

    def __call__(self, text):
        if zhPattern.search(text):
            filter_dict = self.zh_filter.filter(text)
        else:
            filter_dict = self.en_filter.filter(text)

        if '1' in filter_dict:
            logging.info(filter_dict['1']+'  '+text)
            return forbidden_pil
        else:
            return None

prompt_filter = TextFilter()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

def load_img(image, W, H):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    image = image.resize((int(W), int(H)), resample=PIL.Image.LANCZOS)
    print(f"resize input image to size ({W}, {H})")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

class AppModel():
    def __init__(self,):
        self.config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
        self.model = load_model_from_config(self.config, "weights/sd-v1-4.ckpt")

        device = "mtgpu"
        #device = "cpu"
        self.device = device
        self.model = self.model.to(device)
        self.model.cond_stage_model.device = device
        # self.model.model.diffusion_model = torch.jit.load('weights/unet.pt')

        self.sampler = PLMSSampler(self.model)
        self.img_sampler = DDIMSampler(self.model)

        self.C = 4 # latent channels
        self.f = 8 # downsampling factors

    def run_with_prompt(self, seed, prompt, n_samples, W, H, scale, ddim_steps,  strength=0., init_img=None, call_back=None, log_t=1):
        seed_everything(seed)
        ddim_eta=0.0

        assert prompt is not None
        fb_img = prompt_filter(prompt)
        if fb_img is not None:
            return fb_img
        prompt = translator_zh2en(prompt)
        fb_img2 = prompt_filter(prompt)
        if fb_img2 is not None:
            return fb_img2
        print(f"Prompt: {prompt}")
        batch_size = n_samples
        data = [batch_size * [prompt]]

        start_code = torch.randn([n_samples, self.C, H // self.f, W // self.f]).to(self.device)

        if not self.sampler.schedule == "ddim":
            if init_img is not None:
                init_image = load_img(init_img, W, H)
                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                init_image = init_image.to(self.device)
                with torch.no_grad():
                    with torch.autograd.inference_mode(mode=True):
                        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

                assert 0. <= strength < 1., 'can only work with strength in [0.0, 1.0)'
                t_enc = int(strength * ddim_steps)
            else:
                init_latent = None
                t_enc = 1
                
            with torch.no_grad():
                with torch.autograd.inference_mode(mode=True):
                    with self.model.ema_scope():
                        all_samples = list()
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [self.C, H // self.f, W // self.f]
                            samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=start_code, 
                                                            img_callback=call_back,
                                                            log_every_t=int(log_t),
                                                            init_latent=init_latent,
                                                            t_enc=t_enc)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                image = Image.fromarray(x_sample.astype(np.uint8))
                                all_samples.append(image)

                        return all_samples[0]
        else:
            init_image = load_img(init_img, W, H)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_image = init_image.to(self.device)
            with torch.autograd.inference_mode(mode=True):
                init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

            sampler = self.img_sampler

            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

            assert 0. <= strength < 1., 'can only work with strength in [0.0, 1.0)'
            t_enc = int(strength * ddim_steps)
            print(f"target t_enc is {t_enc} steps")

            with torch.no_grad():
                with torch.autograd.inference_mode(mode=True):
                    with self.model.ema_scope():
                        all_samples = list()
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,)

                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                image = Image.fromarray(x_sample.astype(np.uint8))
                                all_samples.append(image)

                        return all_samples[0]
            



if __name__ == "__main__":
    main()
