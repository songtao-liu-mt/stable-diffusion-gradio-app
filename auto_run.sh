#!/bin/bash
conda activate pytorch
cd /home/ubuntu/stable-diffusion-gradio-app
nohup /opt/conda/envs/pytorch/bin/python app.py > logger.log 2>&1 &