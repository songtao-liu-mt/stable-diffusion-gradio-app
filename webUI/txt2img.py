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

from model import AppModel

os.environ["MTGPU_MAX_MEM_USAGE_GB"]="16"

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

#
def txt2img(prompt: str, ddim_steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, seed: Union[int, str, None],
            height: int = 512, width: int= 512, separate_prompts:bool = False, normalize_prompt_weights:bool = True,
            save_individual_images: bool = True, save_grid: bool = True, group_by_prompt: bool = True,
            save_as_jpg: bool = True, use_GFPGAN: bool = True, GFPGAN_model: str = 'GFPGANv1.3', use_RealESRGAN: bool = False,
            RealESRGAN_model: str = "RealESRGAN_x4plus_anime_6B", use_LDSR: bool = True, LDSR_model: str = "model",
            fp = None, variant_amount: float = None,
            variant_seed: int = None, ddim_eta:float = 0.0, write_info_files:bool = True):

    outpath = st.session_state['defaults'].general.outdir_txt2img

    seed = seed_to_int(seed)

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(server_state["model"])
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(server_state["model"])
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(server_state["model"],'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(server_state["model"],'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(server_state["model"],'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(server_state["model"],'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(server_state["model"],'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(server_state["model"],'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale,
                                         unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x, img_callback=generation_callback,
                                                 log_every_t=int(st.session_state.update_preview_frequency))

        return samples_ddim

    #try:
    output_images, seed, info, stats = process_images(
        outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_name=sampler_name,
                save_grid=save_grid,
                batch_size=batch_size,
                n_iter=n_iter,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=separate_prompts,
                use_GFPGAN=st.session_state["use_GFPGAN"],
                GFPGAN_model=st.session_state["GFPGAN_model"],
                use_RealESRGAN=st.session_state["use_RealESRGAN"],
                realesrgan_model_name=RealESRGAN_model,
                use_LDSR=st.session_state["use_LDSR"],
                LDSR_model_name=LDSR_model,
                ddim_eta=ddim_eta,
                normalize_prompt_weights=normalize_prompt_weights,
                save_individual_images=save_individual_images,
                sort_samples=group_by_prompt,
                write_info_files=write_info_files,
                jpg_sample=save_as_jpg,
                variant_amount=variant_amount,
                variant_seed=variant_seed,
    )

    del sampler

    return output_images, seed, info, stats

    #except RuntimeError as e:
        #err = e
        #err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        #stats = err_msg
        #return [], seed, 'err', stats

#
def layout():
    with st.form("txt2img-inputs"):
        st.session_state["generation_mode"] = "txt2img"
        txt_tab,img_tab = st.tabs(["文字输入","图片输入"])
        with txt_tab:
            input_col1_txt, _,_ = st.columns([5, 5, 2])

            with input_col1_txt:
                #prompt = st.text_area("Input Text","")
                prompt = st.text_area("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.", height=350)
        
        with img_tab:
            input_col1, img_col,refresh_col, _ = st.columns([5,5, 1, 1])

            with input_col1:
                #prompt = st.text_area("Input Text","")
                prompt_img = st.text_area("Input Text","", placeholder="A corgi wearing a top hat as an oil painting.", key="img", height=350)
            
            with img_col:

                uploaded_images = st.file_uploader(
                            "Upload Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "webp"],
                            help="Upload an image which will be used for the image to image generation.",
                        )
                image_holder = st.empty()
                if uploaded_images:
                    image = Image.open(uploaded_images).convert('RGB')
                    new_img = image.resize((200, 200))
                    image_holder.image(new_img)

            with refresh_col:
                st.form_submit_button("Refresh")

        # creating the page layout using columns
        #col1, col2, col3 = st.columns([1,2,1], gap="large")
        gcol1, _ = st.columns([1, 20])
        generate_button = gcol1.form_submit_button("Generate")
        col1, col2, = st.columns([1,2], gap="large")

        with col1:
            #width = st.slider("Width:", min_value=st.session_state['defaults'].txt2img.width.min_value, max_value=st.session_state['defaults'].txt2img.width.max_value,
                              #value=st.session_state['defaults'].txt2img.width.value, step=st.session_state['defaults'].txt2img.width.step)
            #height = st.slider("Height:", min_value=st.session_state['defaults'].txt2img.height.min_value, max_value=st.session_state['defaults'].txt2img.height.max_value,
                               #value=st.session_state['defaults'].txt2img.height.value, step=st.session_state['defaults'].txt2img.height.step)
            #cfg_scale = st.slider("CFG (Classifier Free Guidance Scale):", min_value=st.session_state['defaults'].txt2img.cfg_scale.min_value,
                                  #max_value=st.session_state['defaults'].txt2img.cfg_scale.max_value,
                                  #value=st.session_state['defaults'].txt2img.cfg_scale.value, step=st.session_state['defaults'].txt2img.cfg_scale.step,
                                  #help="How strongly the image should follow the prompt.")
            cfg_scale = 7.5

            st.session_state.sampling_steps = st.number_input("Sampling Steps", 
                                                            value=st.session_state.defaults.txt2img.sampling_steps.value,
                                                            min_value=st.session_state.defaults.txt2img.sampling_steps.min_value,
                                                            step=st.session_state['defaults'].txt2img.sampling_steps.step,
                                                            help="Set the default number of sampling steps to use. Default is: 30 (with k_euler)")

            sampler_name_list = ["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
            sampler_name = st.selectbox("Sampling method", sampler_name_list,
                                        index=sampler_name_list.index(st.session_state['defaults'].txt2img.default_sampler), help="Sampling method to use. Default: k_euler")
            seed = st.text_input("Seed:", value=st.session_state['defaults'].txt2img.seed, help=" The seed to use, if left blank a random seed will be generated.")

            #with st.expander("Batch Options"):
                ##batch_count = st.slider("Batch count.", min_value=st.session_state['defaults'].txt2img.batch_count.min_value, max_value=st.session_state['defaults'].txt2img.batch_count.max_value,
                                        ##value=st.session_state['defaults'].txt2img.batch_count.value, step=st.session_state['defaults'].txt2img.batch_count.step,
                                        ##help="How many iterations or batches of images to generate in total.")

                ##batch_size = st.slider("Batch size", min_value=st.session_state['defaults'].txt2img.batch_size.min_value, max_value=st.session_state['defaults'].txt2img.batch_size.max_value,
                                       ##value=st.session_state.defaults.txt2img.batch_size.value, step=st.session_state.defaults.txt2img.batch_size.step,
                                       ##help="How many images are at once in a batch.\
                                       ##It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes to finish generation as more images are generated at once.\
                                       ##Default: 1")

                #st.session_state["batch_count"] = st.number_input("Batch count.", value=st.session_state['defaults'].txt2img.batch_count.value,
                                                                #help="How many iterations or batches of images to generate in total.")

                #st.session_state["batch_size"] = st.number_input("Batch size", value=st.session_state.defaults.txt2img.batch_size.value,
                                                                   #help="How many images are at once in a batch.\
                                                                   #It increases the VRAM usage a lot but if you have enough VRAM it can reduce the time it takes \
                                                                   #to finish generation as more images are generated at once.\
                                                                   #Default: 1")


            st.session_state["update_preview"] = st.session_state["defaults"].general.update_preview
            st.session_state["update_preview_frequency"] = st.number_input("Update Image Preview Frequency",
                                                                            min_value=1,
                                                                            max_value=30,
                                                                            value=st.session_state['defaults'].txt2img.update_preview_frequency,
                                                                            step=5,
                                                                            help="Frequency in steps at which the the preview image is updated. By default the frequency \
                                                                            is set to 10 step.")
            if uploaded_images:
                st.session_state["strength"] = st.slider("Strenght:", min_value=0.1,
                                                max_value=0.99,
                                                step=0.01,
                                                value=0.8,
                                                help="How strongly the image should follow the prompt.")
            else:
                st.session_state["strength"] = -1
                

        with col2:
            #preview_tab, gallery_tab = st.tabs(["Preview", "Gallery"])
            preview_tab,  = st.tabs(["Preview", ])

            with preview_tab:
                #st.write("Image")
                #Image for testing
                #image = Image.open(requests.get("https://icon-library.com/images/image-placeholder-icon/image-placeholder-icon-13.jpg", stream=True).raw).convert('RGB')
                #new_image = image.resize((175, 240))
                #preview_image = st.image(image)

                # create an empty container for the image, progress bar, etc so we can update it later and use session_state to hold them globally.
                st.session_state["preview_image"] = st.empty()


                st.session_state["progress_bar_text"] = st.empty()
                st.session_state["progress_bar_text"].info("Nothing but crickets here, try generating something first.")

                st.session_state["progress_bar"] = st.empty()

                message = st.empty()
        

        if generate_button:

            with col2:
                with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders,index=[0]):
                    if "model" in server_state:
                        print("Already loaded model")
                        model = server_state["model"]
                    else:
                        model = AppModel()
                        server_state["model"] = model

                seed = seed_to_int(seed)

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
                output_image = model.run_with_prompt(seed, prompt_img if uploaded_images else prompt, 1, 512, 384, 7.5, 
                                                    st.session_state.sampling_steps, st.session_state["strength"], image if uploaded_images else None, 
                                                    generation_callback, st.session_state.update_preview_frequency)
                st.session_state["preview_image"].image(output_image)

                #output_images, seeds, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, st.session_state["batch_count"], st.session_state["batch_size"],
                                                            ##cfg_scale, seed, height, width, separate_prompts, normalize_prompt_weights, save_individual_images,
                                                            #cfg_scale, seed, False, normalize_prompt_weights, save_individual_images,
                                                            #save_grid, group_by_prompt, save_as_jpg, st.session_state["use_GFPGAN"], st.session_state['GFPGAN_model'],
                                                            #use_RealESRGAN=st.session_state["use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"],
                                                            #use_LDSR=st.session_state["use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
                                                            #variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files)

                #message.success('Render Complete: ' + info + '; Stats: ' + stats, icon="✅")

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
                #sdGallery(output_images)


            #except (StopException, KeyError):
                #print(f"Received Streamlit StopException")

                # this will render all the images at the end of the generation but its better if its moved to a second tab inside col2 and shown as a gallery.
                # use the current col2 first tab to show the preview_img and update it as its generated.
                #preview_image.image(output_images)


