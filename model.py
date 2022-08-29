import argparse, os, sys, glob
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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


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

    model.cuda()
    model.half()
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
        self.model = load_model_from_config(self.config, "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.model = self.model.to(device)

        self.sampler = PLMSSampler(self.model)
        self.img_sampler = DDIMSampler(self.model)

        self.C = 4 # latent channels
        self.f = 8 # downsampling factors

    def run_with_prompt(self, seed, prompt, n_samples, W, H, scale, ddim_steps,  strength=0., init_img=None):
        seed_everything(seed)
        ddim_eta=0.0

        assert prompt is not None
        print(f"Prompt: {prompt}")
        batch_size = n_samples
        data = [batch_size * [prompt]]

        start_code = None

        n_rows = int(n_samples**0.5)

        precision_scope = autocast 
        if init_img is None:
            with torch.no_grad():
                with precision_scope(device_type='cuda', dtype=torch.float16):
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
                                                            x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                all_samples.append(x_sample.astype(np.uint8))

                        # additionally, grid image
                        grid = torch.stack([x_samples_ddim], 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        grid = grid.astype(np.uint8)

                        return grid, all_samples
        else:
            init_image = load_img(init_img, W, H).to(self.device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            with precision_scope(device_type='cuda', dtype=torch.float16):
                init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

            sampler = self.img_sampler

            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

            assert 0. <= strength < 1., 'can only work with strength in [0.0, 1.0)'
            t_enc = int(strength * ddim_steps)
            print(f"target t_enc is {t_enc} steps")

            with torch.no_grad():
                with precision_scope(device_type='cuda', dtype=torch.float16):
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
                                all_samples.append(x_sample.astype(np.uint8))

                        # additionally, save as grid
                        grid = torch.stack([x_samples], 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        grid = grid.astype(np.uint8)

                        return grid, all_samples
            



if __name__ == "__main__":
    main()
