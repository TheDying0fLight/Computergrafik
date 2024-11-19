import os
from auto1111sdk import StableDiffusionPipeline
from auto1111sdk import civit_download, download_realesrgan, RealEsrganPipeline, StableDiffusionPipeline, EsrganPipeline
from PIL import Image
import torch

def test():
# if __name__ == '__main__':
    # path = "stable-diffusion"

    # pipe = StableDiffusionPipeline(".\dreamshaper_8.safetensors")

    # prompt = "a picture of a brown dog"
    # output = pipe.generate_txt2img(prompt = prompt, height = 1024, width = 768, steps = 10)

    # output[0].save("image.png")
    print("Torch version:",torch.__version__)
    print("Is CUDA enabled?",torch.cuda.is_available())

    civit_url = 'https://civitai.com/models/4384/dreamshaper'

    model_path = 'dreamshaper.safetensors'
    if not os.path.exists(model_path):
        print(f'downloading {model_path} from {civit_url}')
        civit_download(civit_url, model_path)
    else:
        print(f'using model {model_path}')


    print(f'Text to image, model={model_path}')
    pipe = StableDiffusionPipeline(model_path)

    prompt          = "portrait photo of a beautiful 20 y.o. girl, 8k uhd, high quality, cinematic" #@param{type: 'string'}
    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck" #@param{type: 'string'}
    print(f'{prompt=}')
    print(f'{negative_prompt=}')

    num_images      = 1
    height          = 768
    width           = 512
    steps           = 20
    output_path     = "txt2img.png"
    cfg_scale       = 7.5
    seed            = -1
    sampler_name    = 'Euler'
    # ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a",
    #  "DPM++ 2S a", "DPM++ 2M", "DPM fast", "DPM adaptive",
    #  "LMS Karras", "DPM2 Karras", "DPM2 a Karras",
    #  "DPM++ 2S a Karras", "DPM++ 2M Karras", "DDIM", "PLMS"]

    output = pipe.generate_txt2img(
                        num_images = num_images, cfg_scale = cfg_scale, sampler_name = sampler_name, seed       = seed,
                        prompt     = prompt,     height    = height,    width        = width,
                        negative_prompt = negative_prompt,              steps        = steps)

    output[0].save(output_path)

    if os.path.exists(output_path):
        print(f'Text to Image output generated: {output_path}')
    else:
        print(f'Error: output file not found {output_path}')

    del pipe