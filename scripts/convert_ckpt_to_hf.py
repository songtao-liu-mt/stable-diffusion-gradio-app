
import argparse
import os.path as osp

import torch
import os


# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (HF Diffusers, stable-diffusion)
    ("time_embedding.linear_1.weight", "time_embed.0.weight"),
    ("time_embedding.linear_1.bias", "time_embed.0.bias"),
    ("time_embedding.linear_2.weight", "time_embed.2.weight"),
    ("time_embedding.linear_2.bias", "time_embed.2.bias"),
    ("conv_in.weight", "input_blocks.0.0.weight"),
    ("conv_in.bias", "input_blocks.0.0.bias"),
    ("conv_norm_out.weight", "out.0.weight"),
    ("conv_norm_out.bias", "out.0.bias"),
    ("conv_out.weight", "out.2.weight"),
    ("conv_out.bias", "out.2.bias"),
]

unet_conversion_map_resnet = [
    # (HF Diffusers, stable-diffusion)
    ("norm1", "in_layers.0"),
    ("conv1", "in_layers.2"),
    ("norm2", "out_layers.0"),
    ("conv2", "out_layers.3"),
    ("time_emb_proj", "emb_layers.1"),
    ("conv_shortcut", "skip_connection"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((hf_down_res_prefix, sd_down_res_prefix))
        
        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((hf_down_atn_prefix, sd_down_atn_prefix))
    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((hf_up_res_prefix, sd_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((hf_up_atn_prefix, sd_up_atn_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((hf_mid_atn_prefix, sd_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((hf_mid_res_prefix, sd_mid_res_prefix))
    
    
def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for hf_name, sd_name in unet_conversion_map:
        mapping[sd_name] = hf_name
    for k, v in mapping.items():
        if "resnets" in k:
            for hf_part, sd_part in unet_conversion_map_resnet:
                v = v.replace(sd_part, hf_part)
            mapping[k] = v
    for k, v in mapping.items():
        for hf_part, sd_part in unet_conversion_map_layer:
            v = v.replace(sd_part, hf_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (HF Diffusers, stable-diffusion)
    ("conv_shortcut", "nin_shortcut"),
    ("conv_norm_out", "norm_out"),
    ("mid_block.attentions.0.", "mid.attn_1."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((hf_down_prefix, sd_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((hf_downsample_prefix, sd_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3-i}.upsample."
        vae_conversion_map.append((hf_upsample_prefix, sd_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
        vae_conversion_map.append((hf_up_prefix, sd_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i+1}."
    vae_conversion_map.append((hf_mid_res_prefix, sd_mid_res_prefix))
    
vae_conversion_map_attn = [
    # (HF Diffusers, stable-diffusion)
    (".group_norm.", ".norm."),
    (".query.", ".q."),
    (".key.", ".k."),
    (".value.", ".v."),
    (".proj_attn.", ".proj_out."),
]

def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    return w.reshape(*w.shape[:-2])


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for hf_part, sd_part in vae_conversion_map:
            v = v.replace(sd_part, hf_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in v:
            for hf_part, sd_part in vae_conversion_map_attn:
                v = v.replace(sd_part, hf_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["query", "key", "value", "proj_attn"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid_block.attentions.0.{weight_name}.weight" in k:
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#
# pretty much a no-op


def convert_text_enc_state_dict(text_enc_dict):
    return text_enc_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    
    args = parser.parse_args()

    assert args.model_path is not None, "Must provide a model path!"

    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"
    os.makedirs(args.model_path, exist_ok=True)
    
    unet_path = osp.join(args.model_path, "unet", "diffusion_pytorch_model.bin")
    os.makedirs(osp.join(args.model_path, "unet"), exist_ok=True)
    
    vae_path = osp.join(args.model_path, "vae", "diffusion_pytorch_model.bin")
    os.makedirs(osp.join(args.model_path, "vae"), exist_ok=True)
    
    text_enc_path = osp.join(args.model_path, "text_encoder", "pytorch_model.bin")
    os.makedirs(osp.join(args.model_path, "text_encoder"), exist_ok=True)
    
    # loading the ckpt
    print("loading ckpt...")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}
        
    unet_state_dict = {}
    vae_state_dict = {}
    text_enc_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith("model.diffusion_model."):
            unet_state_dict[k] = v
        elif k.startswith("first_stage_model."):
            vae_state_dict[k] = v
        elif k.startswith("cond_stage_model.transformer."):
            text_enc_state_dict[k] = v
            
    
    # Convert the UNet model
    print("start converting unet...")
    unet_state_dict = {k[22:]: v for k, v in unet_state_dict.items()}
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    torch.save(unet_state_dict, unet_path)
    
    # Convert the VAE model
    print("start converting vae...")
    vae_state_dict = {k[18:]: v for k, v in vae_state_dict.items()}
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    torch.save(vae_state_dict, vae_path)
    
     # Convert the text encoder model
    print("start converting text encoder...")
    text_enc_state_dict = {k[29:]: v for k, v in text_enc_state_dict.items()}
    text_enc_state_dict = convert_text_enc_state_dict(text_enc_state_dict)
    torch.save(text_enc_state_dict, text_enc_path)
    
    print('done')
    