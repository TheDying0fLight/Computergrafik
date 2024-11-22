import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from PIL import Image
import numpy as np

class diffuser():
    def __init__(self, model = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.set_pipeline(model)


    def generate(self,
                 prompt,
                 steps = 25,
                 guidance = 7,
                 batch_size = 1,
                 height = None,
                 width = None,
                 seed = None):
        generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)

        self.images = self.pipe(
            prompt * batch_size,
            num_inference_steps = steps,
            guidance_scale = guidance,
            height = height,
            width = width,
            generator = generator
        ).images
        return self.images


    def set_pipeline(self, model):
        self.model = model
        self.pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16)

        self.pipe.enable_vae_slicing()
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()


    def get_grid(self):
        size = int(np.ceil(np.sqrt(len(self.images))))
        return self.image_grid(self.images, size, size)


    def image_grid(self, imgs, rows, cols):
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))

        for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid