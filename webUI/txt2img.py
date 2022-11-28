# This file is part of stable-diffusion-webui (https://github.com/sd-webui/stable-diffusion-webui/).

# Copyright 2022 sd-webui team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# base webui import and utils.
from sd_utils import *
from sd_utils import generation_callback
import hydralit as st


# streamlit imports
from streamlit import StopException, StreamlitAPIException

#streamlit components section
from streamlit_server_state import server_state, server_state_lock
import hydralit_components as hc

# streamlit imports
from streamlit import StopException
#from streamlit.elements import image as STImage
import streamlit.components.v1 as components
from streamlit.runtime.media_file_manager  import media_file_manager
from streamlit.elements.image import image_to_url

#other imports
import uuid
from typing import Union
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import torch
from torchvision import transforms

from model import AppModel

os.environ["MTGPU_MAX_MEM_USAGE_GB"]="15"

# Temp imports


# end of imports
#---------------------------------------------------------------------------------------------------------------


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

#
# Dev mode (server)
# _component_func = components.declare_component(
#         "sd-gallery",
#         url="http://localhost:3001",
#     )

LEXAMPLES = [
"超写实老虎，黑纸，元素，白金，青色， 巴洛克， 洛可可， 水墨，细致，错综复杂的水墨插画",

"一幅中国古代宫殿风景画的写实水彩画，远处是中国古代城墙的入口，背景是雾蒙蒙的山脉",

"gta9游戏玩法，16k分辨率，超级未来主义的图像",

"古埃及，繁荣，青翠的高科技城市，由greg rutkowski, alex grey创作的数字绘画，4k高清",

"红海世界之树的美丽画作， 詹姆斯格尼印象派风格。 非常详细的细节。 幻想，超详细，清晰的焦点，柔和的光线。 光线追踪，由ivan aivazovsky 和 greg rutkowski 绘制",

"Concept art painting of a cozy village in a mountainous forested valley, historic english and japanese architecture, realistic, detailed, cel shaded, in the style of makoto shinkai and greg rutkowski and james gurney",

"Amazon valkyrie athena, d & d, fantasy, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and magali villeneuve",

"Black haired little girl. her hair is very windy. she is in a black and white dress, that white dress has black triangles as a pattern. she is carrying a big glistening green diamond shaped crystal on her hands. art by artgerm and greg rutkowski and alphonse mucha and ian sprigger and wlop and krenz cushart",

]

GUIDENCE = '''为了更好地生成文本提示，从而生成优美的图片。文本输入通常遵循一定的原则。本篇将带领大家生成一个漂亮的文本输入。
注意：为了净化网络环境，请慎重输入词汇，避免 **敏感词**。
* **文本核心描述（必写）**
文本核心描述为文本提示最核心的部分，也是必不可少的部分。一般这个不分需要至少一个核心主体名词加上一定数量的形容词汇，或者动作描述。例如：

一只聪明的熊猫在喝可口可乐
* **类型（必写）**
类型也是文本必不可少的内容。通常来说，类型包括但不限于 照片、肖像、油画、草图、素描、电影海报、广告、3D渲染、图案等等
以上两部分就可以醉成最简单的文本描述。例如：

一只聪明的熊猫在喝可口可乐的电影海报

除了以上两部分为必写内容，可以按照自己的想法加入图像的风格，主体的颜色，图像的纹理，图像分辨率，图像的类型，任务的情绪，所处的时代等等。简单的形式可以用逗号的形式分开。不同的顺序也会产生不同的结果。但是，每一个句子里的中文单字 + 英文单词 建议不要超过 77 个。

'''

# Init Vuejs component
_component_func = components.declare_component(
    "sd-gallery", "./frontend/dists/sd-gallery/dist")

def sdGallery(images=[], key=None):
    component_value = _component_func(images=imgsToGallery(images), key=key, default="")
    return component_value

def imgsToGallery(images):
    urls = []
    for i in images:
        # random string for id
        random_id = str(uuid.uuid4())
        url = image_to_url(
            image=i,
            image_id= random_id,
            width=i.width,
            clamp=False,
            channels="RGB",
            output_format="PNG"
        )
        # image_io = BytesIO()
        # i.save(image_io, 'PNG')
        # width, height = i.size
        # image_id = "%s" % (str(images.index(i)))
        # (data, mimetype) = STImage._normalize_to_bytes(image_io.getvalue(), width, 'auto')
        # this_file = media_file_manager.add(data, mimetype, image_id)
        # img_str = this_file.url
        urls.append(url)

    return urls


class plugin_info():
    plugname = "txt2img"
    description = "Text to Image"
    isTab = True
    displayPriority = 1


def layout():
    with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders,index=[0]):
        if "model" in server_state:
            print("Already loaded model")
            model = server_state["model"]
        else:
            model = AppModel()
            server_state["model"] = model

    st.session_state["generation_mode"] = "txt2img"
    #txt_tab,img_tab = st.tabs(["文字输入","图片输入"])
    input_col1_txt, img_col = st.columns([1, 1], gap="large")
    def example(index):
        st.session_state["text_v"] = LEXAMPLES[index]
        #st.write(st.session_state.text_v)


    with input_col1_txt:
        #prompt = st.text_area("Input Text","")
        #example = st.selectbox("输入示例",
                                #["无",
                                #"英文示例1","英文示例2","英文示例3","英文示例4",
                                #"中文示例1","中文示例2","中文示例3","中文示例4",],
                                #index=0,)
        #if example == "无":
            #text_v = ""
        #else:
            #text_v = EXAMPLES[example]
        text_tab, img_tab = st.tabs(["文字输入","图像输入"])
        with text_tab:
            disable_text = False
            if not "text_v" in st.session_state:
                st.session_state["text_v"] = ""

            prompt = st.text_area("文字输入", st.session_state["text_v"], placeholder="支持英文或者中文输入, 推荐使用右边示例或阅读引导！", 
                                   label_visibility="collapsed", height=190, disabled=disable_text)

        with img_tab:
            uploaded_images = None
            uploaded_images = st.file_uploader(
                        "图像输入", accept_multiple_files=False, type=["png", "jpg", "jpeg", "webp"],
                        help="Upload an image which will be used for the image to image generation.", label_visibility="collapsed",
                    )
            img_h, scale, _ = st.columns([2, 1, 0.3], )
            with scale:
                if uploaded_images:
                    st.session_state["strength"] = st.slider("文本权重", min_value=0.1,
                                                    max_value=0.99,
                                                    step=0.01,
                                                    value=0.8,
                                                    disabled=disable_text,
                                                    help="How strongly the image should follow the prompt.")
            with img_h:
                image_holder = st.empty()
            if uploaded_images:
                image = Image.open(uploaded_images).convert('RGB')
                new_img = image.resize((128, 128))
                image_holder.image(new_img)
            else:
                st.session_state["strength"] = -1
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")

        #with st.form("txt2img-outputs"):
        la, qua, generate,stop_col = st.columns([1, 6, 1.2, 0.8])
        with la:
            st.write("生成质量:")
            #st.markdown("""
            #<style>
            #.big-font {
                #font-size:20px !important;
            #}
            #</style>
            #""", unsafe_allow_html=True)

            #st.markdown('<p class="big-font">生成质量:</p>', unsafe_allow_html=True)
        with qua:
            qlist = ["低（速度快）", "中（默认）", "高（速度慢）"]
            quality = st.radio("图片质量", ["低（速度快）", "中（默认）", "高（速度慢）"], index=1, horizontal=True, label_visibility="collapsed")
            st.session_state.sampling_steps = qlist.index(quality) * 20 + 10

        #generate_button = generate.form_submit_button("开始生成")
        generate_button = generate.button("开始生成")
        stop_button = stop_col.button("中止")

        ex_tab, guide_tab = st.tabs(["输入示例","输入引导"])
        with ex_tab:
            ex1 = st.button(LEXAMPLES[0], on_click=example, kwargs={'index': 0})
            ex2 = st.button(LEXAMPLES[1], on_click=example, kwargs={'index': 1})
            ex3 = st.button(LEXAMPLES[2], on_click=example, kwargs={'index': 2})
            ex4 = st.button(LEXAMPLES[3], on_click=example, kwargs={'index': 3})
            ex5 = st.button(LEXAMPLES[4], on_click=example, kwargs={'index': 4})
            ex6 = st.button(LEXAMPLES[5], on_click=example, kwargs={'index': 5})
            ex7 = st.button(LEXAMPLES[6], on_click=example, kwargs={'index': 6})
            ex8 = st.button(LEXAMPLES[7], on_click=example, kwargs={'index': 7})

        with guide_tab:
            st.markdown(GUIDENCE)
        st.markdown(
            """
            <style>
            .css-6kekos.edgvbvh9{
                text-align: left;
            }
            </style>
            """,unsafe_allow_html=True
        )

    with img_col:
        #preview_tab, gallery_tab = st.tabs(["Preview", "Gallery"])
        preview_tab,  = st.tabs(["图片展示", ])

        with preview_tab:
            #st.write("Image")
            #Image for testing
            #image = Image.open(requests.get("https://icon-library.com/images/image-placeholder-icon/image-placeholder-icon-13.jpg", stream=True).raw).convert('RGB')
            #new_image = image.resize((175, 240))
            #preview_image = st.image(image)

            # create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
            st.session_state["preview_image"] = st.empty()

            st.session_state["progress_bar_text"] = st.empty()
            st.session_state["progress_bar_text"].info("暂时没有结果展示, 请先开始生成图片")

            st.session_state["progress_bar"] = st.empty()

            message = st.empty()

    sampler_name = "k_euler"
    seed = None
    st.session_state["update_preview"] = True
    st.session_state["update_preview_frequency"] = 5


    # creating the page layout using columns
    #col1, col2, col3 = st.columns([1,2,1], gap="large")


    #example_col, _, = st.columns([1,1], gap="large")

    #with example_col:
        #st.session_state.sampling_steps = st.number_input("生成步数", 
                                                        #value=st.session_state.defaults.txt2img.sampling_steps.value,
                                                        #min_value=st.session_state.defaults.txt2img.sampling_steps.min_value,
                                                        #step=st.session_state['defaults'].txt2img.sampling_steps.step,
                                                        #disabled=disable_text,
                                                        #help="Set the default number of sampling steps to use. Default is: 30 (with k_euler)")

        #sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "DDIM"]
        #sampler_name = st.selectbox("采样方法", sampler_name_list,
                                    #disabled=disable_text,
                                    #index=sampler_name_list.index(st.session_state['defaults'].txt2img.default_sampler), help="Sampling method to use. Default: k_euler")
        #seed = st.text_input("随机种子:", value=st.session_state['defaults'].txt2img.seed, 
                            #disabled=disable_text,
                            #help=" The seed to use, if left blank a random seed will be generated.")

        #st.session_state["update_preview"] = st.session_state["defaults"].general.update_preview
        #st.session_state["update_preview_frequency"] = st.number_input("预览频率",
                                                                        #min_value=1,
                                                                        #max_value=30,
                                                                        #value=st.session_state['defaults'].txt2img.update_preview_frequency,
                                                                        #step=5,
                                                                        #disabled=disable_text,
                                                                        #help="Frequency in steps at which the the preview image is updated. By default the frequency \
                                                                        #is set to 10 step.")

    if generate_button:

        with img_col:

            pil_image = transforms.ToPILImage()(torch.ones(3, 384, 512))
            st.session_state["preview_image"].image(pil_image)
            st.session_state["progress_bar_text"].text(
                        f"""运行中: 0/{st.session_state.sampling_steps} 0%""")
            st.session_state["progress_bar"].progress(0)

            if sampler_name == 'PLMS':
                sampler = PLMSSampler(server_state["model"].model)
            elif sampler_name == 'DDIM':
                sampler = DDIMSampler(server_state["model"].model)
            elif sampler_name == 'k_dpm_2_a':
                sampler = KDiffusionSampler(server_state["model"].model,'dpm_2_ancestral')
            elif sampler_name == 'k_dpm_2':
                sampler = KDiffusionSampler(server_state["model"].model,'dpm_2')
            elif sampler_name == 'k_euler_a':
                sampler = KDiffusionSampler(server_state["model"].model,'euler_ancestral')
            elif sampler_name == 'k_euler':
                sampler = KDiffusionSampler(server_state["model"].model,'euler')
            elif sampler_name == 'k_heun':
                sampler = KDiffusionSampler(server_state["model"].model,'heun')
            elif sampler_name == 'k_lms':
                sampler = KDiffusionSampler(server_state["model"].model,'lms')
            else:
                raise Exception("Unknown sampler: " + sampler_name)
            model.sampler = sampler
            seed = seed_to_int(seed)
            output_image = model.run_with_prompt(seed, prompt, 1, 512, 384, 7.5, 
                                                st.session_state.sampling_steps, st.session_state["strength"], image if uploaded_images else None, 
                                                generation_callback, st.session_state.update_preview_frequency)
            st.session_state["preview_image"].image(output_image)

            message.success('生成完毕！', icon="✅")


        #history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont = st.session_state['historyTab']

        #if 'latestImages' in st.session_state:
            #for i in output_images:
                ##push the new image to the list of latest images and remove the oldest one
                ##remove the last index from the list\
                #st.session_state['latestImages'].pop()
                ##add the new image to the start of the list
                #st.session_state['latestImages'].insert(0, i)
            #PlaceHolder.empty()
            #with PlaceHolder.container():
                #col1, col2, col3 = st.columns(3)
                #col1_cont = st.container()
                #col2_cont = st.container()
                #col3_cont = st.container()
                #images = st.session_state['latestImages']
                #with col1_cont:
                    #with col1:
                        #[st.image(images[index]) for index in [0, 3, 6] if index < len(images)]
                #with col2_cont:
                    #with col2:
                        #[st.image(images[index]) for index in [1, 4, 7] if index < len(images)]
                #with col3_cont:
                    #with col3:
                        #[st.image(images[index]) for index in [2, 5, 8] if index < len(images)]
                #historyGallery = st.empty()

            ## check if output_images length is the same as seeds length
            #with gallery_tab:
                #st.markdown(createHTMLGallery(output_images,seeds), unsafe_allow_html=True)


                #st.session_state['historyTab'] = [history_tab,col1,col2,col3,PlaceHolder,col1_cont,col2_cont,col3_cont]

        #with gallery_tab:
            #print(seeds)


        #except (StopException, KeyError):
            #print(f"Received Streamlit StopException")

            # this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
            # use the current col2 first tab to show the preview_img and update it as its generated.
            #preview_image.image(output_images)


