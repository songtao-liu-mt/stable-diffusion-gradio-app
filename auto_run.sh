#!/bin/bash
conda activate pytorch
cd /home/ubuntu/stable-diffusion-gradio-app
nohup python app.py > logger.log 2>&1 &