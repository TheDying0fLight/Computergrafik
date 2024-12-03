from diffusers import AutoPipelineForText2Image, DiffusionPipeline, StableDiffusion3Pipeline
from PIL import Image
from pathlib import Path
import numpy as np
import os, pandas, torch
from IPython.display import clear_output

class Diffuser():
    def __init__(self, model = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.set_model(model)


    def generate(self,
                 prompt,
                 negative_prompt = "",
                 steps = 25,
                 guidance = 7,
                 batch_size = 1,
                 height = None,
                 width = None,
                 seed = None):
        generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)

        self.images = self.pipe(
            [prompt] * batch_size,
            num_inference_steps = steps,
            guidance_scale = guidance,
            height = height,
            width = width,
            generator = generator,
            negative_prompt = [negative_prompt] * batch_size
        ).images
        return self.images


    def set_model(self, model):
        if issubclass(type(model), DiffusionPipeline): self.pipe = model
        else:
            self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16)

        try: self.pipe.enable_vae_slicing()
        except: pass
        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()


    def optimized_sd3pipeline(self, path):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            path,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch.float16
        )
        return pipe


    def get_grid(self):
        size = int(np.ceil(np.sqrt(len(self.images))))
        return self._image_grid_(self.images, size, size)


    def _image_grid_(self, imgs, rows, cols):
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))

        for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid


    def generate_from_csv(self, csv_name: str, prompt_idx: str, name_idx: str, replace = False):
        split = str.partition(csv_name, ".")
        if split[-1] == "csv": csv_name = ".".join(csv_name[0:-1])
        frame = pandas.read_csv(csv_name + ".csv")
        idx_prompt = list(zip(frame[name_idx].to_list(), frame[prompt_idx].to_list()))
        img_path = os.path.join(csv_name, str.split(self.pipe.config._name_or_path,"/")[-1])
        Path(img_path).mkdir(parents=True, exist_ok=True)
        for idx,prompt in idx_prompt:
            path = os.path.join(img_path, f"{idx}.webp")
            if not replace and os.path.exists(path): continue
            self.generate(prompt, batch_size=4)
            img = self.get_grid()
            clear_output(wait=True)
            try:
                display(img)
            except: pass
            img.save(path)