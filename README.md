# Stable Diffusion 运行脚本
基于mtGPU的Stable Diffusion运行脚本

# 环境准备
### 数据
在hugging face上注册账号：[link](https://huggingface.co/join)，并同意Stable Diffusion上的用户协议：[link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original).

然后运行download_weight.sh 脚本：
```
bash download_weight.sh
```

中间需要多次输入自己的hugging face账户和密码

等待预训练模型下载完毕(大概有15个G，需要1个多小时，也可以联系liusongtao获取weight)

### 环境
编译安装好mtPytorch和torchvision (**torchvision 一定不能用 pip install 安装！**
再执行：
```
pip install -r requiements.txt
```

# 运行脚本 
```
bash run.sh
```
以上命令就可以在mtgpu上开始运行stable-diffusion，生成的图片会默认保存在outputs/text2img-samples/samples
