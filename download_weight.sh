mkdir weights
cd weights
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
git clone https://huggingface.co/openai/clip-vit-large-patch14

mv stable-diffusion-v-1-4-original/sd-v1-4.ckpt .

echo Done.

