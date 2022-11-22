import torch
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--output_dir", default='./', type=str, required=True, help="Path to output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    torch.save({'state_dict': unet_state_dict}, os.path.join(args.output_dir, 'unet.ckpt'))
    torch.save({'state_dict': vae_state_dict}, os.path.join(args.output_dir, 'vae.ckpt'))
    torch.save({'state_dict': text_enc_state_dict}, os.path.join(args.output_dir, 'text_encoder.ckpt'))
    
    