#MUSA_LAUNCH_BLOCKING=1 python scripts/txt2img_mt.py --prompt "a photograph of an astronaut riding a horse" --plms
#MTGPU_MAX_MEM_USAGE_GB=16 python scripts/txt2img_mt.py --prompt "A fantasy landscape, trending on artstation." --plms --H 512 --ddim_steps 50
MTGPU_MAX_MEM_USAGE_GB=16 python scripts/txt2img_mt.py --prompt "a photograph of an astronaut riding a horse" --plms --ddim_steps 50 $@
